"""Build a disabled, jointly bounded tokenizer-loss comparison launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256, notebook as trial_notebook


REQUIRED_MEMBERS = frozenset({
    'genie_diep/baseline.py', 'genie_diep/cohort.py', 'genie_diep/model.py',
    'genie_diep/baseline_evaluation.py', 'genie_diep/gpu_precision.py', 'genie_diep/detail_loss.py',
    'scripts/run_tokenizer_comparison.py', 'scripts/diagnose_tokenizer_precision.py',
    'configs/training-baseline-v1.json', 'configs/training-readiness-v1.json',
    'configs/tokenizer-comparison-v1.json',
    'configs/tokenizer-comparison-baseline-v1.json', 'configs/tokenizer-comparison-detail-v1.json',
    'configs/tokenizer-comparison-baseline-readiness-v1.json',
    'configs/tokenizer-comparison-detail-readiness-v1.json',
})
SOURCE_ALLOWLIST = REQUIRED_MEMBERS
BOUND_FILES = {
    'expected_experiment_sha256': 'configs/tokenizer-comparison-v1.json',
    'expected_baseline_spec_sha256': 'configs/tokenizer-comparison-baseline-v1.json',
    'expected_detail_spec_sha256': 'configs/tokenizer-comparison-detail-v1.json',
    'expected_training_spec_sha256': 'configs/training-baseline-v1.json',
    'expected_readiness_sha256': 'configs/training-readiness-v1.json',
    'expected_baseline_readiness_sha256': 'configs/tokenizer-comparison-baseline-readiness-v1.json',
    'expected_detail_readiness_sha256': 'configs/tokenizer-comparison-detail-readiness-v1.json',
    'expected_model_sha256': 'genie_diep/model.py',
}
HASH_FIELDS = tuple(BOUND_FILES) + ('expected_implementation_sha256',)


def validate_manifest(manifest):
    name = manifest['overlay_name']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*\.zip', name):
        raise ValueError('Overlay must use a simple ZIP filename')
    if name == 'genie-baseline-v1-v9.zip':
        raise ValueError('Preserve the original data archive')
    for field in ('overlay_sha256',) + HASH_FIELDS:
        if not isinstance(manifest[field], str) or not re.fullmatch(r'[0-9a-f]{64}', manifest[field]):
            raise ValueError('Invalid SHA256: ' + field)
    if type(manifest['overlay_bytes']) is not int or not 0 < manifest['overlay_bytes'] <= 16 * 1024**2:
        raise ValueError('Source overlay must be at most 16 MiB')
    return {field: manifest[field] for field in ('overlay_name', 'overlay_sha256', 'overlay_bytes') + HASH_FIELDS}


def overlay_code(manifest):
    manifest = validate_manifest(manifest)
    return '''    # Bind every source/configuration member before replacing any file.
    overlay_contract = OVERLAY_CONTRACT
    required_overlay = set(REQUIRED_OVERLAY)
    bound_files = BOUND_FILES
    overlay_source = pathlib.Path('/content/drive/MyDrive') / overlay_contract['overlay_name']
    assert overlay_source.stat().st_size == overlay_contract['overlay_bytes'], 'Overlay size mismatch'
    overlay_path = pathlib.Path('/content') / overlay_contract['overlay_name']
    shutil.copyfile(overlay_source, overlay_path)
    with overlay_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == overlay_contract['overlay_sha256'], 'Overlay hash mismatch'
    with zipfile.ZipFile(overlay_path) as overlay:
        names = overlay.namelist()
        assert len(names) == len(set(names)), 'Duplicate overlay member'
        assert 'overlay-sha256.json' in names, 'Missing overlay hashes'
        overlay_hashes = json.loads(overlay.read('overlay-sha256.json'))
        assert isinstance(overlay_hashes, dict), 'Invalid overlay hashes'
        assert set(names) == set(overlay_hashes) | {'overlay-sha256.json'}, 'Unbound overlay member'
        assert set(overlay_hashes) == required_overlay, 'Overlay must contain exactly the source/configuration allowlist'
        payloads = {}
        for name, digest in overlay_hashes.items():
            assert (root/name).resolve().is_relative_to(root.resolve()), 'Unsafe overlay path'
            member = overlay.getinfo(name)
            assert not member.is_dir() and (member.external_attr >> 16) & 0o170000 != 0o120000, 'Overlay symlinks/directories denied'
            assert member.file_size <= 2 * 1024**2, 'Source overlay member too large'
            payload = overlay.read(name)
            assert hashlib.sha256(payload).hexdigest() == digest, 'Overlay member hash mismatch: ' + name
            payloads[name] = payload
        for field, name in bound_files.items():
            assert overlay_hashes[name] == overlay_contract[field], 'Input binding mismatch: ' + name
        implementation = hashlib.sha256()
        for name in ['baseline.py', 'cohort.py', 'model.py', 'baseline_evaluation.py']:
            implementation.update(name.encode())
            implementation.update(payloads['genie_diep/' + name].replace(b'\\r\\n', b'\\n'))
        assert implementation.hexdigest() == overlay_contract['expected_implementation_sha256'], 'Implementation binding mismatch'
        # No writes happen until the complete overlay and experiment are verified.
        for name, payload in payloads.items():
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    status['overlay_sha256'] = overlay_contract['overlay_sha256']
    status['experiment_sha256'] = overlay_contract['expected_experiment_sha256']
    status['overlay_applied'] = True
'''.replace('OVERLAY_CONTRACT', repr(manifest)).replace('REQUIRED_OVERLAY', repr(sorted(REQUIRED_MEMBERS))).replace('BOUND_FILES', repr(BOUND_FILES))


def notebook(manifest):
    manifest = validate_manifest(manifest)
    document = trial_notebook()
    document['cells'][0]['source'] = ['''# Diep-Genie: matched tokenizer-loss comparison

Reuse the immutable data ZIP already in My Drive. Upload only the small source
overlay before allocating a GPU. Both arms start fresh from identical weights
and use frozen matched sampling. No previous model weights are uploaded.

The whole experiment permits at most 1,000 optimizer calls per arm, including
one exact-resume replay: at most 999 unique updates per arm, with the final
common checkpoint at 999. Both arms share a COMBINED 2 compute unit / one
allocated hour ceiling, preserving 50 units. Setup, idle,
checkpoint export and evaluation all count. If interrupted, compare only common
saved boundaries. This screen cannot establish convergence, G1 or controllability.

After authorization, enter the live balance/rate, actual allocation start and any
prior debits for this experiment below. A new allocation cannot reset its budget.
No purchases. The watchdog starts before Drive access; success, bounded stops,
failures and deadlines release the runtime. The exported notebook is disabled.
''']
    source = ''.join(document['cells'][2]['source'])
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Trial template changed: ' + repr(old))
        source = source.replace(old, new)
    replace('Authorize only this bounded measurement first', 'Authorize only this paired tokenizer comparison first')
    # Once the authorization flag is enabled, invalid accounting also releases
    # the allocation instead of failing before the template's cleanup block.
    start = source.index('assert math.isfinite(LIVE_BALANCE)')
    end = source.index('finished = threading.Event()', start)
    validation = source[start:end]
    source = source[:start] + 'try:\n' + ''.join('    ' + line if line.strip() else line for line in validation.splitlines(keepends=True)) + 'except BaseException:\n    runtime.unassign()\n    raise\n' + source[end:]
    replace("'purpose': 'GPU execution measurement only'", "'purpose': 'Matched tokenizer loss comparison; combined budget'")
    replace("'quality_gates': 'unmeasured', 'training_stage_promotion': False}",
            "'quality_gates': 'unmeasured', 'training_stage_promotion': False,\n          'maximum_optimizer_calls_per_arm': 1000, 'maximum_unique_updates_per_arm': 999,\n          'final_common_checkpoint_step': 999, 'budget_shared_by_both_arms': True}")
    replace("pathlib.Path('/content/drive/MyDrive/Diep-Genie/baseline-v1-v9')", "pathlib.Path('/content/drive/MyDrive/Diep-Genie/tokenizer-comparison-v1')")
    replace("time.strftime('pilot-%Y%m%d-%H%M%S'", "time.strftime('comparison-%Y%m%d-%H%M%S'")
    replace("pathlib.Path('/content/genie-baseline-v1-v9')", "pathlib.Path('/content/genie-tokenizer-comparison-v1')")
    marker = "    assert time.time() + 180 < stop_at, 'Setup exhausted the measurement budget'"
    replace(marker, overlay_code(manifest) + marker)
    start = source.index('    # The pilot requires a new directory')
    end = source.index('\nexcept BaseException as error:', start)
    command = '''    command = [sys.executable, '-u', '-m', 'scripts.run_tokenizer_comparison',
               '--root', '.', '--baseline-spec', 'configs/tokenizer-comparison-baseline-v1.json',
               '--detail-spec', 'configs/tokenizer-comparison-detail-v1.json',
               '--experiment', 'configs/tokenizer-comparison-v1.json',
               '--out', str(result/'comparison'), '--device', 'cuda', '--authorized',
               '--rate', str(DISPLAYED_UNITS_PER_HOUR), '--balance', str(LIVE_BALANCE),
               '--allocation-start', str(accounting_start)]
    status['status'] = 'comparison'
    (result/'launch-status.json').write_text(json.dumps(status, indent=2) + '\\n')
    with (result/'comparison.log').open('w', buffering=1) as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=max(1, stop_at-time.time()-180))
    report = json.loads((result/'comparison/comparison-report.json').read_text())
    assert report['status'] in {'completed', 'budget_exhausted'}, 'Comparison failed; inspect preserved reports'
    status['status'] = report['status']
    status['quality_gates'] = 'unmeasured'
    print('Saved bounded comparison to Drive:', result, '\\n', json.dumps(report, indent=2), flush=True)
'''
    source = source[:start] + command + source[end:]
    document['cells'][2]['source'] = source.splitlines(keepends=True)
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{index}', 'exec')
    return document


def build(output, overlay_manifest):
    manifest = validate_manifest(json.loads(Path(overlay_manifest).read_text(encoding='utf-8')))
    document = notebook(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output/'tokenizer-comparison.ipynb'
    target.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    result = dict(notebook=target.name, notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        original_archive='genie-baseline-v1-v9.zip', original_archive_sha256=ARCHIVE_SHA256,
        original_archive_bytes=ARCHIVE_BYTES, overlay=manifest, execution_authorized_in_file=False,
        maximum_compute_units=2, maximum_allocated_seconds=3600, minimum_remaining_units=50,
        maximum_optimizer_calls_per_arm=1000, maximum_unique_updates_per_arm=999,
        final_common_checkpoint_step=999, budget_shared_by_both_arms=True,
        contains_existing_weights=False, training_stage_promotion=False)
    (output/'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--overlay-manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.overlay_manifest), indent=2))
