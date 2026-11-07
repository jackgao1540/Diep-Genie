"""Reproducible local coverage experiment; never change the original pilot.

Six additional training clips were reviewed in all 16 frames. They contain
ordinary player tanks and colored food. Some lower-edge text remains; this is
a narrow coverage experiment, not a declaration that bulk data is fully clean.
"""
import json
from pathlib import Path
import shutil
from genie_diep.data import ClipDataset,file_hash


def main():
    project=Path(__file__).resolve().parents[1]
    target=project/'data/pilot-plus-food'
    if target.exists():raise FileExistsError('Preserve existing experiment datasets')
    selected=[]
    for folder in ('pilot-clean','ordinary-addition'):
        root=project/'data'/folder
        for manifest in sorted(root.glob('*.jsonl')):
            for line in manifest.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if folder=='ordinary-addition':
                    if not(93.5<=row['start'] and row['end']<=103.11):continue
                    assert row['split']=='train' and row['source_id']=='35gc8uMbkiQ'
                    row={**row,'curation_note':'All 16 frames reviewed: ordinary tank and colored food; occasional lower-edge text. No large caption/cut in this interval.'}
                selected.append((root,row))
    assert sum(root.name=='ordinary-addition' for root,_ in selected)==6
    assert len({row['path'] for _,row in selected})==len(selected)
    target.mkdir(parents=True)
    records=[]
    for root,row in selected:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe clip path')
        shutil.copy2(path,target/row['path'])
        records.append({**row,'clip_sha256':file_hash(path)})
    (target/'clips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    train=ClipDataset(target,'train',5);val=ClipDataset(target,'val',5)
    baseline=ClipDataset(project/'data/pilot-clean','val',5)
    assert {r['path'] for r in val.rows}=={r['path'] for r in baseline.rows}
    for row in val.rows:
        assert file_hash(target/row['path'])==file_hash(project/'data/pilot-clean'/row['path'])
    print(json.dumps(dict(train=len(train),val=len(val),added_training_clips=6,validation_unchanged=True)))


if __name__=='__main__':main()
