"""Package the prepared baseline and train/val proxies only; never allocate compute."""
import hashlib
import json
from pathlib import Path
import zipfile
from scripts.build_stationary_recovery import read,rows,write
from genie_diep.data import file_hash


def run():
    root=Path.cwd();out=root/'exports/baseline-v1-v9'
    if out.exists():raise FileExistsError('Preserve the previous export; choose a new version')
    out.mkdir(parents=True)
    binding=read('configs/baseline-data-v9.json');cohort=Path(binding['cohort']);manifest=rows(cohort/'manifest.jsonl')
    files=list(Path('genie_diep').glob('*.py'))+[Path('pyproject.toml'),Path('scripts/gpu_tokenizer_pilot.py')]
    files += [Path('configs')/n for n in ['training-baseline-v1.json','baseline-data-v9.json','baseline-evaluation-v1.json','baseline-fidelity-review-v1.json','training-readiness-v1.json','longform-ninth-checkpoint-v1.json','sprayer-class-resolution-v1.json']]
    files += [cohort/n for n in ['manifest.jsonl','summary.json','sequences.json']]
    files += [cohort/r['proxy_path'] for r in manifest if r['split'] in {'train','val'}]
    files=sorted(set(files));hashes={};archive=out/'genie-baseline-v1-v9.zip'
    expected={str(cohort/r['proxy_path']).replace('\\','/'):r['proxy_sha256'] for r in manifest if r['split'] in {'train','val'}}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,p in enumerate(files):
            digest=file_hash(p);name=p.as_posix()
            if name in expected and expected[name]!=digest:raise ValueError('Proxy changed before packaging')
            hashes[name]=digest;z.write(p,name)
            if (i+1)%1000==0:print(json.dumps(dict(packaged=i+1,total=len(files))),flush=True)
        z.writestr('bundle-sha256.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(hashes)|{'bundle-sha256.json'}
        for name,digest in hashes.items():
            with z.open(name) as stream:
                h=hashlib.file_digest(stream,'sha256').hexdigest()
            assert h==digest,name
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),cells=[])
    def cell(kind,source):
        c=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        notebook['cells'].append(c)
    cell('markdown','''# Diep-Genie: first bounded GPU trial

Local data and execution checks are complete. This notebook measures tokenizer GPU memory, speed, FP16 stability and resume. It does not run the long training stages or demonstrate model quality.

Use the locally prepared `genie-baseline-v1-v9.zip`. It contains train/validation proxies and frozen split metadata; no reserved-test images or original videos. Upload and setup time count toward the 2-unit / 1-hour cap. Keep at least 50 units in reserve. No purchases.

After authorization, enter the live balance, displayed hourly rate and actual allocation start time below. The last cell mounts your Drive, saves results there, runs a watchdog, and unassigns the runtime even on failure. A failed trial stops progression. No automatic tokenizer/action/dynamics training follows.
''')
    cell('code','''AUTHORIZED_TO_RUN = False
LIVE_BALANCE = 0.0
DISPLAYED_UNITS_PER_HOUR = 0.0
ALLOCATION_START_UTC = ""  # e.g. 2026-11-08T12:00:00+00:00; actual allocation start
assert AUTHORIZED_TO_RUN, "Authorize this bounded trial first"
assert LIVE_BALANCE > 50 and DISPLAYED_UNITS_PER_HOUR > 0
from datetime import datetime
allocation_start = datetime.fromisoformat(ALLOCATION_START_UTC).timestamp()
''')
    cell('code','''import time, threading, pathlib, json, hashlib, zipfile, subprocess, sys
from google.colab import files, drive, runtime
deadline = allocation_start + min(3600, 2 / DISPLAYED_UNITS_PER_HOUR * 3600, (LIVE_BALANCE - 50) / DISPLAYED_UNITS_PER_HOUR * 3600)
assert time.time() + 180 < deadline, "Insufficient budget remains for setup/export"
finished = threading.Event()
def watchdog():
    while not finished.wait(min(30, max(0.1, deadline - time.time()))):
        if time.time() >= deadline:
            runtime.unassign()
            return
threading.Thread(target=watchdog, daemon=True).start()
try:
    uploaded = files.upload()  # Select genie-baseline-v1-v9.zip
    archive_path = next(pathlib.Path(n) for n in uploaded if n.endswith('.zip'))
    uploaded.clear()
    root = pathlib.Path('/content/genie-baseline-v1-v9')
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            assert (root / member.filename).resolve().is_relative_to(root.resolve()), 'Unsafe archive path'
        archive.extractall(root)
    expected = json.loads((root/'bundle-sha256.json').read_text())
    for name, digest in expected.items():
        with (root/name).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == digest, name
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','-e',str(root)],check=True)
    drive.mount('/content/drive')
    result = pathlib.Path('/content/drive/MyDrive/Diep-Genie/baseline-v1-v9') / time.strftime('pilot-%Y%m%d-%H%M%S')
    subprocess.run([sys.executable,'-m','scripts.gpu_tokenizer_pilot','--out',str(result),
        '--rate',str(DISPLAYED_UNITS_PER_HOUR),'--balance',str(LIVE_BALANCE),
        '--allocation-start',str(allocation_start)],cwd=root,check=True)
    report = json.loads((result/'pilot-report.json').read_text())
    print('Saved to Drive:', result, '\\nTrial status:', report['status'])
finally:
    finished.set()
    runtime.unassign()
''')
    write(out/'first-gpu-trial.ipynb',notebook)
    report=dict(archive=archive.as_posix(),archive_sha256=file_hash(archive),bytes=archive.stat().st_size,files=len(files),
        proxy_files=len(expected),reserved_test_proxy_files=0,all_archive_members_hash_verified=True,
        notebook=(out/'first-gpu-trial.ipynb').as_posix(),training_started=False)
    write(out/'manifest.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
