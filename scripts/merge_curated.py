"""Combine approved immutable cohorts without changing source split assignments."""
import argparse
import json
import shutil
from pathlib import Path
from genie_diep.data import ClipDataset,file_hash


def run(inputs,out):
    roots=[Path(p).resolve() for p in inputs];out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve existing datasets')
    if len({p.name for p in roots})!=len(roots):raise ValueError('Cohort names must be unique')
    rows=[];copies=[];sources={};hashes={}
    for root in roots:
        manifest=root/'clips.jsonl'
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('training_approved') is not True:raise ValueError('Unapproved cohort clip')
            src=(root/row['path']).resolve();relative=Path(root.name)/row['path'];dst=(out/relative).resolve()
            if not src.is_relative_to(root) or not dst.is_relative_to(out):raise ValueError('Unsafe path')
            checksum=file_hash(src)
            if checksum!=row['clip_sha256']:raise ValueError('Curated content changed')
            if sources.setdefault(row['source_id'],row['split'])!=row['split']:raise ValueError('Source split leakage')
            if checksum in hashes:raise ValueError('Duplicate clip across cohorts')
            hashes[checksum]=row['split']
            rows.append({**row,'path':relative.as_posix(),'cohort':root.name,
                         'cohort_manifest_sha256':file_hash(manifest)})
            copies.append((src,dst))
    if not {'train','val'}.issubset({r['split'] for r in rows}):raise ValueError('Need train and validation')
    out.mkdir(parents=True)
    for src,dst in copies:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    (out/'clips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    result=dict(inputs=[str(p) for p in roots],train=len(ClipDataset(out,'train')),val=len(ClipDataset(out,'val')))
    (out/'merge-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('inputs',nargs='+')
    run(**vars(p.parse_args()))
