"""Build a disabled GPU recovery check using the existing data ZIP and a small overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256, notebook as trial_notebook


SOURCE_ALLOWLIST = frozenset({
    'genie_diep/baseline.py', 'genie_diep/cohort.py', 'genie_diep/model.py',
    'genie_diep/baseline_evaluation.py', 'genie_diep/gpu_precision.py',
    'configs/training-baseline-v1.json', 'configs/training-readiness-v1.json',
    'scripts/gpu_recovery_verification.py', 'scripts/diagnose_tokenizer_precision.py',
    'scripts/gpu_tokenizer_pilot.py', 'recovery-inputs/last.pt',
    'recovery-inputs/original-spec.json',
})
REQUIRED_MEMBERS = SOURCE_ALLOWLIST - {'scripts/gpu_tokenizer_pilot.py'}
HASH_FIELDS = ('expected_checkpoint_sha256', 'expected_implementation_sha256',
               'expected_spec_sha256', 'expected_model_sha256')


def validate_manifest(manifest):
    name = manifest['overlay_name']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*\.zip', name):
        raise ValueError('Overlay must use a simple ZIP filename')
    if name == 'genie-baseline-v1-v9.zip':
        raise ValueError('Overlay cannot replace the original archive')
    for field in ('overlay_sha256',) + HASH_FIELDS:
        if not isinstance(manifest[field], str) or not re.fullmatch(r'[0-9a-f]{64}', manifest[field]):
            raise ValueError('Invalid SHA256: ' + field)
    if type(manifest['overlay_bytes']) is not int or not 0 < manifest['overlay_bytes'] <= 128 * 1024**2:
        raise ValueError('Overlay must be a positive size of at most 128 MiB')
    return {key: manifest[key] for key in ('overlay_name', 'overlay_sha256', 'overlay_bytes') + HASH_FIELDS}


def overlay_code(manifest):
    """This exact block is tested before embedding it into the notebook."""
    manifest = validate_manifest(manifest)
    return '''    # Verify the separately uploaded overlay before replacing any source.
    overlay_contract = OVERLAY_CONTRACT
    allowed_overlay = set(ALLOWED_OVERLAY)
    required_overlay = set(REQUIRED_OVERLAY)
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
        assert set(overlay_hashes) <= allowed_overlay, 'Overlay member not allowed'
        assert required_overlay <= set(overlay_hashes), 'Missing required overlay member'
        payloads = {}
        for name, digest in overlay_hashes.items():
            target = (root / name).resolve()
            assert target.is_relative_to(root.resolve()), 'Unsafe overlay path'
            member = overlay.getinfo(name)
            assert not member.is_dir() and (member.external_attr >> 16) & 0o170000 != 0o120000, 'Overlay symlinks/directories denied'
            assert member.file_size <= 64 * 1024**2, 'Overlay member too large'
            payload = overlay.read(name)
            assert hashlib.sha256(payload).hexdigest() == digest, 'Overlay member hash mismatch: ' + name
            payloads[name] = payload
        assert overlay_hashes['recovery-inputs/last.pt'] == overlay_contract['expected_checkpoint_sha256'], 'Checkpoint binding mismatch'
        assert overlay_hashes['configs/training-baseline-v1.json'] == overlay_contract['expected_spec_sha256'], 'Specification binding mismatch'
        assert hashlib.sha256(payloads['genie_diep/model.py'].replace(b'\\r\\n', b'\\n')).hexdigest() == overlay_contract['expected_model_sha256'], 'Model source binding mismatch'
        # No writes happen until every member and input binding has passed.
        for name, payload in payloads.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    status['overlay_sha256'] = overlay_contract['overlay_sha256']
    status['overlay_applied'] = True
'''.replace('OVERLAY_CONTRACT', repr(manifest)).replace('ALLOWED_OVERLAY', repr(sorted(SOURCE_ALLOWLIST))).replace('REQUIRED_OVERLAY', repr(sorted(REQUIRED_MEMBERS)))


def notebook(manifest):
    manifest = validate_manifest(manifest)
    document = trial_notebook()
    document['cells'][0]['source'] = ['''# Diep-Genie: bounded GPU recovery verification

Reuse the original immutable data archive already uploaded to My Drive. Upload
the separate small recovery overlay before allocating a GPU. The overlay is
bound to revised source/configuration and the preserved unfinished checkpoint;
it cannot modify the dataset. Original artifacts stay preserved.

This check compares the recovered failing batch at FP32 and FP16, verifies
failure rollback/resume, and performs at most 300 diagnostic optimizer calls
with fixed held-out evaluation. It does not promote weights or authorize longer
training, actions or dynamics. Model quality gates remain unmeasured.

Enter the live balance/rate and actual allocation start only after authorization.
Stop at 2 compute units or one allocated hour, preserving 50 units. Setup, export
and idle time count. Prior debits apply only to this authorized experiment.
No purchases. The watchdog starts before Drive setup; completion, failure and
deadline paths release the runtime. Keep the export disabled until use.
''']
    source = ''.join(document['cells'][2]['source'])

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Trial template changed: ' + repr(old))
        source = source.replace(old, new)

    replace('Authorize only this bounded measurement first', 'Authorize this bounded recovery verification first')
    replace("'purpose': 'GPU execution measurement only'", "'purpose': 'Bounded GPU recovery verification'")
    replace("pathlib.Path('/content/drive/MyDrive/Diep-Genie/baseline-v1-v9')", "pathlib.Path('/content/drive/MyDrive/Diep-Genie/recovery-verification-v1')")
    replace("time.strftime('pilot-%Y%m%d-%H%M%S'", "time.strftime('verification-%Y%m%d-%H%M%S'")
    replace("pathlib.Path('/content/genie-baseline-v1-v9')", "pathlib.Path('/content/genie-recovery-verification-v1')")
    marker = "    assert time.time() + 180 < stop_at, 'Setup exhausted the measurement budget'"
    replace(marker, overlay_code(manifest) + marker)
    start = source.index('    # The pilot requires a new directory')
    end = source.index('\nexcept BaseException as error:', start)
    command = '''    command = [sys.executable, '-u', '-m', 'scripts.gpu_recovery_verification',
               '--checkpoint', 'recovery-inputs/last.pt',
               '--original-spec', 'recovery-inputs/original-spec.json',
               '--out', str(result/'verification'), '--updates', '300',
               '--rate', str(DISPLAYED_UNITS_PER_HOUR), '--balance', str(LIVE_BALANCE),
               '--allocation-start', str(accounting_start)]
    for key in EXPECTED_FIELDS:
        command.extend(['--' + key.replace('_', '-'), overlay_contract[key]])
    status['status'] = 'verification'
    (result/'launch-status.json').write_text(json.dumps(status, indent=2) + '\\n')
    with (result/'verification.log').open('w', buffering=1) as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=max(1, stop_at-time.time()-120))
    report = json.loads((result/'verification/verification-report.json').read_text())
    assert report['status'] == 'passed', 'GPU recovery verification did not pass'
    status['status'] = 'passed'
    print('Saved to Drive:', result, '\\n', json.dumps(report, indent=2), flush=True)
'''.replace('EXPECTED_FIELDS', repr(HASH_FIELDS))
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
    target = output/'gpu-recovery-verification.ipynb'
    target.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    result = dict(notebook=target.name, notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  original_archive='genie-baseline-v1-v9.zip', original_archive_sha256=ARCHIVE_SHA256,
                  original_archive_bytes=ARCHIVE_BYTES, overlay=manifest,
                  execution_authorized_in_file=False, maximum_compute_units=2,
                  maximum_allocated_seconds=3600, minimum_remaining_units=50,
                  maximum_optimizer_calls=300, training_stage_promotion=False)
    (output/'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--overlay-manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.overlay_manifest), indent=2))
