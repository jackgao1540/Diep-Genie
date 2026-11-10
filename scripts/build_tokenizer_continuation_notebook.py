"""Build a disabled paired continuation launcher with a private checkpoint overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256, notebook as trial_notebook
from scripts.build_tokenizer_comparison_notebook import REQUIRED_MEMBERS as COMPARISON_MEMBERS, BOUND_FILES as COMPARISON_BOUND_FILES


PARENT_MEMBERS = {
    'baseline': 'continuation-inputs/baseline-step999.pt',
    'detail': 'continuation-inputs/detail-step999.pt',
}
REQUIRED_MEMBERS = COMPARISON_MEMBERS | frozenset({
    'genie_diep/coverage_sampler.py', 'scripts/run_tokenizer_continuation.py',
    'configs/tokenizer-continuation-v1.json', 'configs/tokenizer-continuation-readiness-v1.json', *PARENT_MEMBERS.values(),
})
BOUND_FILES = dict(COMPARISON_BOUND_FILES,
    expected_parent_experiment_sha256='configs/tokenizer-comparison-v1.json',
    expected_experiment_sha256='configs/tokenizer-continuation-v1.json',
    expected_continuation_readiness_sha256='configs/tokenizer-continuation-readiness-v1.json',
    expected_baseline_parent_sha256=PARENT_MEMBERS['baseline'],
    expected_detail_parent_sha256=PARENT_MEMBERS['detail'],
    expected_sampler_source_sha256='genie_diep/coverage_sampler.py',
    expected_runner_sha256='scripts/run_tokenizer_continuation.py')
HASH_FIELDS = tuple(BOUND_FILES) + ('expected_implementation_sha256',)
MAX_OVERLAY_BYTES = 64 * 1024**2


def validate_manifest(manifest):
    name = manifest['overlay_name']
    if not isinstance(name, str) or not re.fullmatch(r'genie-tokenizer-continuation-[A-Za-z0-9_-]+\.zip', name):
        raise ValueError('Use a versioned tokenizer-continuation ZIP filename')
    for field in ('overlay_sha256',) + HASH_FIELDS:
        if not isinstance(manifest[field], str) or not re.fullmatch(r'[0-9a-f]{64}', manifest[field]):
            raise ValueError('Invalid SHA256: ' + field)
    if type(manifest['overlay_bytes']) is not int or not 0 < manifest['overlay_bytes'] <= MAX_OVERLAY_BYTES:
        raise ValueError('Private overlay must be at most 64 MiB')
    return {field: manifest[field] for field in ('overlay_name', 'overlay_sha256', 'overlay_bytes') + HASH_FIELDS}


def overlay_code(manifest):
    manifest = validate_manifest(manifest)
    return '''    # Verify all private inputs before replacing any source or adding weights.
    overlay_contract = OVERLAY_CONTRACT
    required_overlay = set(REQUIRED_OVERLAY)
    parent_members = set(PARENT_MEMBERS)
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
        assert set(overlay_hashes) == required_overlay, 'Overlay must contain exactly the approved input allowlist'
        assert sum(member.file_size for member in overlay.infolist()) <= 64 * 1024**2, 'Expanded overlay too large'
        payloads = {}
        for name, digest in overlay_hashes.items():
            target = (root/name).resolve()
            assert target.is_relative_to(root.resolve()), 'Unsafe overlay path'
            member = overlay.getinfo(name)
            assert not member.is_dir() and (member.external_attr >> 16) & 0o170000 != 0o120000, 'Overlay symlinks/directories denied'
            maximum = (32 if name in parent_members else 2) * 1024**2
            assert member.file_size <= maximum, 'Overlay member too large'
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
        # Parent checkpoints are added only at fresh, explicitly allowlisted paths.
        for name in parent_members:
            assert not (root/name).exists(), 'Parent checkpoint destination already exists'
        for name, payload in payloads.items():
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in parent_members:
                with target.open('xb') as stream: stream.write(payload)
            else:
                target.write_bytes(payload)
    status['overlay_sha256'] = overlay_contract['overlay_sha256']
    status['experiment_sha256'] = overlay_contract['expected_experiment_sha256']
    status['overlay_applied'] = True
'''.replace('OVERLAY_CONTRACT', repr(manifest)).replace('REQUIRED_OVERLAY', repr(sorted(REQUIRED_MEMBERS))).replace('PARENT_MEMBERS', repr(sorted(PARENT_MEMBERS.values()))).replace('BOUND_FILES', repr(BOUND_FILES))


def notebook(manifest):
    manifest = validate_manifest(manifest)
    document = trial_notebook()
    document['cells'][0]['source'] = ['''# Diep-Genie: paired tokenizer continuation

Reuse the immutable original data ZIP. Upload the separate PRIVATE continuation
overlay before allocating a GPU. It contains exactly the two preserved step-999
parent checkpoints plus bound source and configuration. Do not publish the overlay.
Original checkpoints and prior exports remain unchanged.

Continue both arms to step 5,000: at most 4,001 new unique updates and 4,002
optimizer calls per arm, including one resume replay. Both arms share a COMBINED
4 compute unit / three allocated hour ceiling, preserving 50 units. Setup, idle,
validation and checkpoint export all count. Compare only common saved boundaries.
This continuation cannot establish convergence, G1 or controllability by itself.

After authorization, enter the live balance/rate, actual allocation start and
prior debits for this continuation. A new allocation cannot reset its budget.
No purchases. The watchdog starts before Drive access. Completion, bounded stops,
failures and deadlines release the runtime. This exported notebook is disabled.
''']
    source = ''.join(document['cells'][2]['source'])
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Trial template changed: ' + repr(old))
        source = source.replace(old, new)
    replace('Authorize only this bounded measurement first', 'Authorize only this bounded paired continuation first')
    replace('deadline = accounting_start + min(3600, 2 / DISPLAYED_UNITS_PER_HOUR * 3600,',
            'deadline = accounting_start + min(10800, 4 / DISPLAYED_UNITS_PER_HOUR * 3600,')
    start = source.index('assert math.isfinite(LIVE_BALANCE)')
    end = source.index('finished = threading.Event()', start)
    validation = source[start:end]
    source = source[:start] + 'try:\n' + ''.join('    ' + line if line.strip() else line for line in validation.splitlines(keepends=True)) + 'except BaseException:\n    runtime.unassign()\n    raise\n' + source[end:]
    replace("'purpose': 'GPU execution measurement only'", "'purpose': 'Paired tokenizer continuation; combined budget'")
    replace("'maximum_compute_units': 2, 'maximum_allocated_seconds': 3600,", "'maximum_compute_units': 4, 'maximum_allocated_seconds': 10800,")
    replace("'quality_gates': 'unmeasured', 'training_stage_promotion': False}",
            "'quality_gates': 'unmeasured', 'training_stage_promotion': False,\n          'parent_step': 999, 'target_step': 5000, 'maximum_additional_optimizer_calls_per_arm': 4002,\n          'maximum_new_unique_updates_per_arm': 4001, 'budget_shared_by_both_arms': True}")
    replace("pathlib.Path('/content/drive/MyDrive/Diep-Genie/baseline-v1-v9')", "pathlib.Path('/content/drive/MyDrive/Diep-Genie/tokenizer-continuation-v1')")
    replace("time.strftime('pilot-%Y%m%d-%H%M%S'", "time.strftime('continuation-%Y%m%d-%H%M%S'")
    replace("pathlib.Path('/content/genie-baseline-v1-v9')", "pathlib.Path('/content/genie-tokenizer-continuation-v1')")
    marker = "    assert time.time() + 180 < stop_at, 'Setup exhausted the measurement budget'"
    replace(marker, overlay_code(manifest) + marker)
    start = source.index('    # The pilot requires a new directory')
    end = source.index('\nexcept BaseException as error:', start)
    command = '''    command = [sys.executable, '-u', '-m', 'scripts.run_tokenizer_continuation',
               '--experiment', 'configs/tokenizer-continuation-v1.json',
               '--out', str(result/'continuation'), '--device', 'cuda', '--authorized',
               '--rate', str(DISPLAYED_UNITS_PER_HOUR), '--balance', str(LIVE_BALANCE),
               '--allocation-start', str(accounting_start)]
    status['status'] = 'continuation'
    (result/'launch-status.json').write_text(json.dumps(status, indent=2) + '\\n')
    with (result/'continuation.log').open('w', buffering=1) as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=max(1, stop_at-time.time()-180))
    report = json.loads((result/'continuation/continuation-report.json').read_text())
    assert report['status'] in {'completed', 'budget_exhausted'}, 'Continuation failed; inspect preserved evidence'
    status['status'] = report['status']
    print('Saved bounded continuation to Drive:', result, '\\n', json.dumps(report, indent=2), flush=True)
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
    target = output/'tokenizer-continuation.ipynb'
    target.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    result = dict(notebook=target.name, notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        original_archive='genie-baseline-v1-v9.zip', original_archive_sha256=ARCHIVE_SHA256,
        original_archive_bytes=ARCHIVE_BYTES, overlay=manifest, execution_authorized_in_file=False,
        maximum_compute_units=4, maximum_allocated_seconds=10800, minimum_remaining_units=50,
        parent_step=999, target_step=5000, maximum_additional_optimizer_calls_per_arm=4002,
        maximum_new_unique_updates_per_arm=4001, budget_shared_by_both_arms=True,
        contains_private_checkpoint_weights=True, training_stage_promotion=False)
    (output/'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--overlay-manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.overlay_manifest), indent=2))
