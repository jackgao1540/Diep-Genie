"""Package the prepared baseline and train/val proxies only; never allocate compute."""
import argparse
import hashlib
import re
import json
from pathlib import Path
import zipfile
from scripts.build_stationary_recovery import read,rows,write
from genie_diep.data import file_hash
from genie_diep.baseline import validate_readiness
from scripts.build_drive_trial_notebook import notebook as drive_notebook


def run(version='baseline-v1-v9'):
    if not re.fullmatch(r'baseline-v[1-9][0-9]*-v9', version):
        raise ValueError('Use a new baseline-vN-v9 export version')
    root=Path.cwd();out=root/'exports'/version
    if out.exists():raise FileExistsError('Preserve the previous export; choose a new version')
    validate_readiness(root,'configs/training-baseline-v1.json')
    out.mkdir(parents=True)
    binding=read('configs/baseline-data-v9.json');cohort=Path(binding['cohort']);manifest=rows(cohort/'manifest.jsonl')
    files=list(Path('genie_diep').glob('*.py'))+[Path('pyproject.toml'),Path('scripts/gpu_tokenizer_pilot.py')]
    files += [Path('configs')/n for n in ['training-baseline-v1.json','baseline-data-v9.json','baseline-evaluation-v1.json','baseline-fidelity-review-v1.json','training-readiness-v1.json','longform-ninth-checkpoint-v1.json','sprayer-class-resolution-v1.json']]
    files += [cohort/n for n in ['manifest.jsonl','summary.json','sequences.json']]
    files += [cohort/r['proxy_path'] for r in manifest if r['split'] in {'train','val'}]
    files=sorted(set(files));hashes={};archive=out/f'genie-{version}.zip'
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
    notebook=drive_notebook(archive.name,file_hash(archive),archive.stat().st_size)
    for index,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell-{index}','exec')
    write(out/'gpu-measurement-drive.ipynb',notebook)
    report=dict(archive=archive.as_posix(),archive_sha256=file_hash(archive),bytes=archive.stat().st_size,files=len(files),
        proxy_files=len(expected),reserved_test_proxy_files=0,all_archive_members_hash_verified=True,
        notebook=(out/'gpu-measurement-drive.ipynb').as_posix(),
        notebook_sha256=file_hash(out/'gpu-measurement-drive.ipynb'),
        export_version=version,execution_authorized_in_file=False,training_started=False)
    write(out/'manifest.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',default='baseline-v1-v9')
    run(parser.parse_args().version)
