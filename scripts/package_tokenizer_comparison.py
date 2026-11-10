"""Package a source-only matched tokenizer comparison over the immutable data ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from genie_diep.baseline import implementation_hash, validate_readiness
from genie_diep.data import file_hash
from scripts.build_tokenizer_comparison_notebook import BOUND_FILES, REQUIRED_MEMBERS, build


def validate_comparison(root):
    from scripts.run_tokenizer_comparison import validate_experiment
    return validate_experiment(root, 'configs/tokenizer-comparison-v1.json',
        'configs/tokenizer-comparison-baseline-v1.json', 'configs/tokenizer-comparison-detail-v1.json')


def run(output='exports/tokenizer-comparison-v1'):
    root = Path.cwd()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous comparison exports')
    for specification in ('configs/training-baseline-v1.json', 'configs/tokenizer-comparison-baseline-v1.json',
                          'configs/tokenizer-comparison-detail-v1.json'):
        validate_readiness(root, specification)
    validate_comparison(root)
    sources = {name: root/name for name in REQUIRED_MEMBERS}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > 2 * 1024**2:
            raise ValueError('Source overlay member too large: ' + path.name)
    # Read every bound input before opening the export; no partially selected
    # experiments can be packaged while source/configuration files are missing.
    payloads = {name: path.read_bytes() for name, path in sources.items()}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    for name, digest in hashes.items():
        if file_hash(sources[name]) != digest:
            raise ValueError('Comparison input changed during packaging: ' + name)
    output.mkdir(parents=True)
    archive = output/'genie-tokenizer-comparison-v1.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in sorted(payloads.items()):
            bundle.writestr(name, payload)
        bundle.writestr('overlay-sha256.json', json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(REQUIRED_MEMBERS) | {'overlay-sha256.json'}
        for name, digest in hashes.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest, name
    manifest = dict(overlay_name=archive.name, overlay_sha256=file_hash(archive), overlay_bytes=archive.stat().st_size,
        expected_implementation_sha256=implementation_hash(), member_hashes=hashes,
        contains_private_diagnostic_weights=False, contains_existing_weights=False, contains_dataset=False,
        maximum_compute_units=2, maximum_allocated_seconds=3600,
        maximum_optimizer_calls_per_arm=1000, maximum_unique_updates_per_arm=999,
        final_common_checkpoint_step=999,
        budget_shared_by_both_arms=True)
    manifest.update({field: hashes[name] for field, name in BOUND_FILES.items()})
    manifest_path = output/'overlay-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    build(output/'launcher', manifest_path)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='exports/tokenizer-comparison-v1')
    run(parser.parse_args().out)
