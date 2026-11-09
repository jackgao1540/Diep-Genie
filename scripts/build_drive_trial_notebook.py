"""Create a versioned Drive launcher for the immutable v9 GPU trial archive."""
import argparse
import hashlib
import json
from pathlib import Path


ARCHIVE_SHA256 = '2ff978a9676e87f003929c5727158279113f7e4426febc3ea550950a01ac1c6f'
ARCHIVE_BYTES = 2915470473


def notebook():
    cells = []

    def cell(kind, source):
        item = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''# Diep-Genie: bounded v9 GPU measurement from Drive

Upload the immutable v9 ZIP to My Drive before connecting a GPU. This launcher
reuses that archive without changing its data, implementation or configuration.
It measures GPU memory, throughput, FP16 stability and checkpoint continuation;
it does not establish model quality or authorize further training.

After authorization, enter the live balance, displayed hourly usage rate and
actual allocation start timestamp below. All allocated setup and idle time counts.
Stop at 2 compute units or one allocated hour; preserve 50 units. No purchases.
The watchdog begins before Drive authentication and setup. Results go to Drive,
and the runtime is unassigned on success or failure. Do not run the historical pilot.
''')
    cell('code', '''AUTHORIZED_TO_RUN = False
LIVE_BALANCE = 0.0
DISPLAYED_UNITS_PER_HOUR = 0.0
ALLOCATION_START_UTC = ""  # Actual allocation start, ISO 8601 with UTC offset
''')
    cell('code', '''import hashlib, json, math, pathlib, shutil, subprocess, sys, threading, time, zipfile
from datetime import datetime
from google.colab import drive, runtime

assert AUTHORIZED_TO_RUN, "Authorize only this bounded measurement first"
assert math.isfinite(LIVE_BALANCE) and LIVE_BALANCE > 50
assert math.isfinite(DISPLAYED_UNITS_PER_HOUR) and DISPLAYED_UNITS_PER_HOUR > 0
allocation_time = datetime.fromisoformat(ALLOCATION_START_UTC)
assert allocation_time.utcoffset() is not None, "Allocation timestamp needs a timezone"
allocation_start = allocation_time.timestamp()
assert allocation_start <= time.time(), "Allocation start cannot be in the future"
deadline = allocation_start + min(3600, 2 / DISPLAYED_UNITS_PER_HOUR * 3600,
                                  (LIVE_BALANCE - 50) / DISPLAYED_UNITS_PER_HOUR * 3600)
# Begin shutdown 30 seconds early to allow the unassign request to finish.
stop_at = deadline - 30
finished = threading.Event()

def watchdog():
    while not finished.wait(min(5, max(0.1, stop_at - time.time()))):
        if time.time() >= stop_at:
            runtime.unassign()
            return

threading.Thread(target=watchdog, daemon=True).start()
result = None
status = {'status': 'setup', 'purpose': 'GPU execution measurement only',
          'allocation_start_utc': ALLOCATION_START_UTC,
          'live_balance': LIVE_BALANCE, 'displayed_units_per_hour': DISPLAYED_UNITS_PER_HOUR,
          'maximum_compute_units': 2, 'maximum_allocated_seconds': 3600,
          'quality_gates': 'unmeasured', 'training_stage_promotion': False}
try:
    assert time.time() + 180 < stop_at, "Insufficient budget for setup/export"
    drive.mount('/content/drive')
    result = pathlib.Path('/content/drive/MyDrive/Diep-Genie/baseline-v1-v9') / time.strftime('pilot-%Y%m%d-%H%M%S', time.gmtime())
    result.mkdir(parents=True, exist_ok=False)
    (result/'launch-status.json').write_text(json.dumps(status, indent=2) + '\\n')
    source = pathlib.Path('/content/drive/MyDrive/genie-baseline-v1-v9.zip')
    assert source.stat().st_size == 2915470473, 'Archive size mismatch'
    archive_path = pathlib.Path('/content/genie-baseline-v1-v9.zip')
    shutil.copyfile(source, archive_path)
    with archive_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == '2ff978a9676e87f003929c5727158279113f7e4426febc3ea550950a01ac1c6f', 'Archive hash mismatch'
    root = pathlib.Path('/content/genie-baseline-v1-v9')
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            assert (root/member.filename).resolve().is_relative_to(root.resolve()), 'Unsafe archive path'
        archive.extractall(root)
    expected = json.loads((root/'bundle-sha256.json').read_text())
    for name, digest in expected.items():
        with (root/name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    assert time.time() + 180 < stop_at, 'Setup exhausted the measurement budget'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', str(root)],
                   check=True, timeout=max(1, stop_at-time.time()-180))
    # The pilot requires a new directory and writes its own report even on failure.
    with (result/'pilot.log').open('w', buffering=1) as log:
        subprocess.run([sys.executable, '-u', '-m', 'scripts.gpu_tokenizer_pilot', '--out', str(result/'measurement'),
                        '--rate', str(DISPLAYED_UNITS_PER_HOUR), '--balance', str(LIVE_BALANCE),
                        '--allocation-start', str(allocation_start)], cwd=root, check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=max(1, stop_at-time.time()-120))
    report = json.loads((result/'measurement/pilot-report.json').read_text())
    status['status'] = report['status']
    print('Saved to Drive:', result, '\\n', json.dumps(report, indent=2), flush=True)
except BaseException as error:
    status.update(status='failed', error=repr(error))
    raise
finally:
    try:
        status['allocated_seconds_before_unassign'] = time.time()-allocation_start
        status['estimated_units_before_unassign'] = status['allocated_seconds_before_unassign']*DISPLAYED_UNITS_PER_HOUR/3600
        if result is not None and result.exists():
            (result/'launch-status.json').write_text(json.dumps(status, indent=2) + '\\n')
    finally:
        # Keep the watchdog armed if unassign itself fails.
        runtime.unassign()
        finished.set()
''')
    return dict(nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(name='python3', display_name='Python 3')),
                cells=cells)


def build(output):
    output = Path(output)
    document = notebook()
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{index}', 'exec')
    output.mkdir(parents=True, exist_ok=False)
    target = output/'first-gpu-trial-drive-v1.ipynb'
    target.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    manifest = dict(notebook=target.name, notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    archive='genie-baseline-v1-v9.zip', archive_sha256=ARCHIVE_SHA256,
                    archive_bytes=ARCHIVE_BYTES, archive_modified=False, execution_authorized_in_file=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='exports/baseline-v1-v9-drive-launch-v1')
    print(json.dumps(build(parser.parse_args().out), indent=2))
