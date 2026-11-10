"""Package verified private step-999 parents without changing prior artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import torch

from genie_diep.baseline import implementation_hash, validate_readiness
from genie_diep.cohort import read
from genie_diep.data import file_hash
from scripts.build_tokenizer_continuation_notebook import BOUND_FILES, MAX_OVERLAY_BYTES, PARENT_MEMBERS, REQUIRED_MEMBERS, build
from scripts.run_tokenizer_comparison import compare_exact
from scripts.validate_gpu_recovery_artifacts import finite_tensors


def validate_inputs(root, baseline_checkpoint, detail_checkpoint):
    from scripts.run_tokenizer_continuation import validate_experiment, validate_parent, validate_continuation_readiness
    plan = validate_experiment(root, 'configs/tokenizer-continuation-v1.json')
    validate_continuation_readiness(root, 'configs/tokenizer-continuation-v1.json')
    for specification in ('configs/training-baseline-v1.json', 'configs/tokenizer-comparison-baseline-v1.json',
                          'configs/tokenizer-comparison-detail-v1.json'):
        validate_readiness(root, specification)
    parent_experiment_path = root/plan['parent_experiment']['path']
    if file_hash(parent_experiment_path) != plan['parent_experiment']['sha256']:
        raise ValueError('Parent experiment binding changed')
    parent_experiment = read(parent_experiment_path)
    states = {}
    checkpoints = dict(baseline=Path(baseline_checkpoint).resolve(), detail=Path(detail_checkpoint).resolve())
    for arm, path in checkpoints.items():
        if plan['parents'][arm]['path'] != PARENT_MEMBERS[arm]:
            raise ValueError('Parent checkpoint archive path differs from the exact allowlist')
        if file_hash(path) != plan['parents'][arm]['sha256']:
            raise ValueError('Parent checkpoint hash changed: ' + arm)
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_parent(state, plan, arm, parent_experiment, root)
        if state['step'] != 999 or state['contract']['stage'] != 'tokenizer' or state['contract']['preflight'] is not False:
            raise ValueError('Only the preserved production step-999 tokenizer parents are allowed')
        finite_tensors(state['model'], arm+'.model')
        finite_tensors(state['optimizer'], arm+'.optimizer')
        states[arm] = state
    compare_exact(states['baseline']['sampler'], states['detail']['sampler'])
    return plan, checkpoints


def run(output, baseline_checkpoint, detail_checkpoint):
    root = Path.cwd()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior continuation exports')
    plan, checkpoints = validate_inputs(root, baseline_checkpoint, detail_checkpoint)
    sources = {name: root/name for name in REQUIRED_MEMBERS}
    for arm, name in PARENT_MEMBERS.items():
        sources[name] = checkpoints[arm]
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        maximum = (32 if name in PARENT_MEMBERS.values() else 2) * 1024**2
        if path.stat().st_size > maximum:
            raise ValueError('Overlay member too large: ' + name)
    payloads = {name: path.read_bytes() for name, path in sources.items()}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    for arm, name in PARENT_MEMBERS.items():
        if hashes[name] != plan['parents'][arm]['sha256']:
            raise ValueError('Parent checkpoint changed while packaging')
    for name, digest in hashes.items():
        if file_hash(sources[name]) != digest:
            raise ValueError('Bound continuation input changed during packaging: ' + name)
    hash_payload = (json.dumps(hashes, indent=2) + '\n').encode('utf-8')
    if sum(map(len, payloads.values())) + len(hash_payload) + 512*len(payloads) > MAX_OVERLAY_BYTES:
        raise ValueError('Expanded private overlay exceeds 64 MiB')
    output.mkdir(parents=True)
    archive = output/'genie-tokenizer-continuation-v1.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as bundle:
        for name, payload in sorted(payloads.items()):
            bundle.writestr(name, payload)
        bundle.writestr('overlay-sha256.json', hash_payload)
    if archive.stat().st_size > MAX_OVERLAY_BYTES:
        raise ValueError('Private overlay archive exceeds 64 MiB')
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(REQUIRED_MEMBERS) | {'overlay-sha256.json'}
        for name, digest in hashes.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest, name
    manifest = dict(overlay_name=archive.name, overlay_sha256=file_hash(archive), overlay_bytes=archive.stat().st_size,
        expected_implementation_sha256=implementation_hash(), member_hashes=hashes,
        contains_private_checkpoint_weights=True, parent_checkpoints_verified=True, contains_dataset=False,
        original_artifacts_modified=False, maximum_compute_units=4, maximum_allocated_seconds=10800,
        minimum_remaining_units=50, parent_step=999, target_step=5000,
        maximum_additional_optimizer_calls_per_arm=4002, maximum_new_unique_updates_per_arm=4001,
        budget_shared_by_both_arms=True)
    manifest.update({field: hashes[name] for field, name in BOUND_FILES.items()})
    manifest_path = output/'overlay-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    build(output/'launcher', manifest_path)
    # Readback includes the caller's original checkpoints, not merely ZIP copies.
    for arm, path in checkpoints.items():
        if file_hash(path) != plan['parents'][arm]['sha256']:
            raise ValueError('Original checkpoint changed during export')
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='exports/tokenizer-continuation-v1')
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--detail-checkpoint', required=True)
    args = parser.parse_args()
    run(args.out, args.baseline_checkpoint, args.detail_checkpoint)
