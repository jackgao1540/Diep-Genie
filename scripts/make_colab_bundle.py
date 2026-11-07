"""Package reviewed local inputs; creates a notebook but never starts a runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/pilot-clean")
    p.add_argument("--out",default="exports"); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; out=root/a.out; out.mkdir(parents=True,exist_ok=True)
    files=[*sorted((root/"genie_diep").glob("*.py")),root/"pyproject.toml",root/"README.md"]
    data=root/a.data
    files+=sorted(data.glob("*.jsonl"))+sorted(data.glob("*.npy"))
    # Continue the representative local tokenizer, never the single-clip overfit model.
    checkpoint=root/"runs/real-tokenizer-detail/last.pt"
    if checkpoint.exists(): files.append(checkpoint)
    manifest={str(path.relative_to(root)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with zipfile.ZipFile(out/"genie-pilot.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(root))
        z.writestr("bundle-sha256.json",json.dumps(manifest,indent=2))
    def md(text):return dict(cell_type="markdown",metadata={},source=text.splitlines(keepends=True))
    def code(text):return dict(cell_type="code",metadata={},source=text.splitlines(keepends=True),execution_count=None,outputs=[])
    cells=[md("# Genie / diep.io: bounded GPU pilot\n\nPrepared locally. Upload only the bundled source and preprocessed clips. Do not scrape or preprocess videos here. This notebook does not establish playability.\n\nBefore connecting a GPU: inspect the Colab unit balance and displayed rate, set the verified rate below, and confirm local gates in HISTORY.md. No purchases. The local task has not yet approved this pilot as ready."),
      code("""# Local evidence must be reviewed before changing these values.
LOCAL_GATES_PASSED = False
DISPLAYED_UNITS_PER_HOUR = 0.0  # Read from Colab; do not guess.
PILOT_UNIT_CAP = 2.0
PILOT_WALL_CAP_SECONDS = 600
assert LOCAL_GATES_PASSED, 'Local validation is required before this paid pilot.'
assert DISPLAYED_UNITS_PER_HOUR > 0, 'Enter the actual displayed rate.'
TRAIN_SECONDS = min(PILOT_WALL_CAP_SECONDS, int(PILOT_UNIT_CAP / DISPLAYED_UNITS_PER_HOUR * 3600 * .8))
assert TRAIN_SECONDS >= 60
print('Training-loop limit:', TRAIN_SECONDS, 'seconds. Setup/export/idle also consume units.')
"""),
      code("""from google.colab import files
uploaded = files.upload()  # Select genie-pilot.zip built locally.
"""),
      code("""from pathlib import Path
import hashlib, json, zipfile, sys, subprocess
root = Path('/content/genie-pilot')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('genie-pilot.zip') as archive:
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        assert target.is_relative_to(root.resolve()), 'Unsafe archive path'
    archive.extractall(root)
manifest = json.loads((root/'bundle-sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-e',str(root)],check=True)
import torch
assert torch.cuda.is_available(), 'Use a GPU only after local verification.'
print(torch.cuda.get_device_name(), 'PyTorch',torch.__version__)
"""),
      code("""# Tokenizer throughput/quality pilot only. Real-video action learning is still under repair locally.
import time
started = time.monotonic()
def run(args):
    subprocess.run([sys.executable,'-m','genie_diep.train',*args],cwd=root,check=True)
run(['--stage','tokenizer','--data','data/pilot-clean','--init','runs/real-tokenizer-detail/last.pt',
     '--batch','16','--steps','100000','--max-seconds',str(TRAIN_SECONDS),
     '--device','cuda','--out','runs/pilot-tokenizer','--log-every','100'])
subprocess.run([sys.executable,'-m','genie_diep.evaluate','runs/pilot-tokenizer/last.pt',
                '--data','data/pilot-clean','--out','runs/pilot-tokenizer/evaluation','--device','cuda'],cwd=root,check=True)
"""),
      code("""# Export before ending the runtime. Download can take time; verify locally.
import shutil
result = shutil.make_archive('/content/genie-pilot-results','zip',root/'runs')
files.download(result)
"""),
      md("After the results file has downloaded successfully, run the last cell immediately to end GPU billing. If an earlier cell errors, save the latest checkpoints and end the runtime. The loop time limit is not a hard billing cap and does not stop idle billing."),
      code("""from google.colab import runtime
runtime.unassign()
""")]
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name="python3",display_name="Python 3")),cells=cells)
    (out/"genie-pilot.ipynb").write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    print(json.dumps(dict(bundle=str(out/"genie-pilot.zip"),files=len(files),bytes=(out/"genie-pilot.zip").stat().st_size,
                          notebook=str(out/"genie-pilot.ipynb"),ready_for_paid_run=False)))


if __name__=="__main__":main()
