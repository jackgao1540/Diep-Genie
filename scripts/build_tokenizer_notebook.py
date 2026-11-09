"""Build a disabled, bounded tokenizer launcher using the verified archive."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_drive_trial_notebook import notebook as trial_notebook


def notebook(pilot):
    required = ('status', 'implementation_sha256', 'spec_sha256', 'binding_sha256')
    evidence = {key: pilot[key] for key in required}
    if evidence['status'] != 'passed':
        raise ValueError('Passing GPU measurement required')
    document = trial_notebook()
    document['cells'][0]['source'] = ['''# Diep-Genie: bounded fresh tokenizer training

Authorized limits: 8 compute units / 6 allocated hours / 20,000 updates,
whichever comes first, preserving 50 units. Setup, validation, export and idle
allocation count. No purchases. No action or dynamics training is authorized.
Use the live balance/rate and actual allocation start below. Prior debits apply
only to attempts belonging to this experiment. Start from fresh weights.
The immutable archive and passing GPU evidence are checked before training.
Checkpoints and fixed held-out validation diagnostics are saved to Drive.
These diagnostics do not establish G1 object retention or controllability.
The runtime releases itself on completion, failure or budget expiry.
''']
    source = ''.join(document['cells'][2]['source'])

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'Trial template changed: {old!r}')
        source = source.replace(old, new)

    replace('Authorize only this bounded measurement first', 'Authorize this bounded tokenizer run first')
    replace('min(3600, 2 /', 'min(21600, 8 /')
    replace("'purpose': 'GPU execution measurement only'", "'purpose': 'Fresh bounded tokenizer training'")
    replace("'maximum_compute_units': 2, 'maximum_allocated_seconds': 3600", "'maximum_compute_units': 8, 'maximum_allocated_seconds': 21600, 'maximum_updates': 20000")
    replace("time.strftime('pilot-%Y%m%d-%H%M%S'", "time.strftime('tokenizer-%Y%m%d-%H%M%S'")
    replace('Setup exhausted the measurement budget', 'Setup exhausted the training budget')
    start = source.index('    # The pilot requires a new directory')
    end = source.index('\nexcept BaseException as error:', start)
    source = source[:start] + '''    pilot = PILOT_EVIDENCE
    pilot_path = result/'verified-pilot.json'
    pilot_path.write_text(json.dumps(pilot, indent=2) + '\\n')
    # The frozen trainer validates pilot/implementation/config hashes itself.
    command = [sys.executable, '-u', '-m', 'genie_diep.baseline', '--stage', 'tokenizer',
               '--out', str(result/'training'), '--device', 'cuda', '--precision', 'fp16',
               '--microbatch', '4', '--accumulation', '4', '--updates', '20000',
               '--authorized', '--pilot-report', str(pilot_path),
               '--rate', str(DISPLAYED_UNITS_PER_HOUR), '--balance', str(LIVE_BALANCE),
               '--allocation-start', str(accounting_start)]
    status.update(status='training', command=command)
    (result/'launch-status.json').write_text(json.dumps(status, indent=2) + '\\n')
    print('Training outputs:', result, flush=True)
    with (result/'training.log').open('w', buffering=1) as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.time() >= stop_at-60:
                    raise TimeoutError('Export reserve reached')
                time.sleep(min(30, max(.1, stop_at-60-time.time())))
                with (result/'training.log').open() as progress:
                    progress.seek(max(0, (result/'training.log').stat().st_size-4096))
                    lines = progress.read().splitlines()
                if lines:
                    print(lines[-1], flush=True)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    assert (result/'training/last.pt').is_file(), 'Missing final checkpoint'
    status['status'] = 'completed'
    print('Saved checkpoints and validation diagnostics to Drive:', result, flush=True)
''' .replace('PILOT_EVIDENCE', repr(evidence)) + source[end:]
    document['cells'][2]['source'] = source.splitlines(keepends=True)
    return document


def build(output, pilot_path):
    output = Path(output)
    document = notebook(json.loads(Path(pilot_path).read_text(encoding='utf-8')))
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{index}', 'exec')
    output.mkdir(parents=True, exist_ok=False)
    target = output/'tokenizer-v9-bounded-v1.ipynb'
    target.write_text(json.dumps(document, indent=2)+'\n', encoding='utf-8')
    manifest = dict(notebook=target.name, notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    pilot_report_sha256=hashlib.sha256(Path(pilot_path).read_bytes()).hexdigest(),
                    archive_modified=False, execution_authorized_in_file=False,
                    max_compute_units=8, max_allocated_hours=6, max_updates=20000, reserve_units=50)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pilot-report', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.pilot_report), indent=2))
