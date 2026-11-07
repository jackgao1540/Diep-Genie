"""Build an immutable dataset from explicit, content-bound visual decisions."""
import argparse
import json
import shutil
from pathlib import Path
from genie_diep.data import ClipDataset,file_hash


def build(candidate_root,decisions,out,screen):
    root=Path(candidate_root).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Do not overwrite a curated dataset')
    rows={}
    for manifest in sorted(root.glob('*.jsonl')):
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if not line:continue
            row=json.loads(line)
            if row['path'] in rows:raise ValueError('Duplicate candidate path')
            rows[row['path']]=row
    screen_path=Path(screen);screen_report=json.loads(screen_path.read_text(encoding='utf-8'))
    if not screen_report['arguments'].get('every_frame') or screen_report['arguments'].get('padding',0)<24:
        raise ValueError('Final caption screen must cover every frame with a 24-pixel margin')
    caption={r['path']:r for r in screen_report['rows']}
    selected=[];seen=set();hash_splits={};caption_rejected=[]
    for decision_path in map(Path,decisions):
        review=json.loads(decision_path.read_text(encoding='utf-8'))
        for decision in review['clips']:
            path=decision['path']
            if path in seen:raise ValueError('A clip has multiple review decisions')
            seen.add(path)
            if decision['decision'] not in ('approve','reject'):raise ValueError('Unrecognized review decision')
            if decision['decision']!='approve':continue
            if path not in caption:raise ValueError('Approved clip lacks the final caption-screen result')
            if caption[path]['suspicious']:
                caption_rejected.append(path);continue
            row=rows[path];src=(root/path).resolve();dst=(out/path).resolve()
            if caption[path]['sampled_frames']!=row['shape'][0]:raise ValueError('Incomplete caption coverage')
            if not src.is_relative_to(root) or not dst.is_relative_to(out):raise ValueError('Unsafe clip path')
            for key in ('source_id','split','start','end'):
                if decision[key]!=row[key]:raise ValueError('Review metadata differs from candidate')
            checksum=file_hash(src)
            if checksum!=decision['sha256']:raise ValueError('Reviewed clip content changed')
            previous=hash_splits.setdefault(checksum,row['split'])
            if previous!=row['split']:raise ValueError('Identical clip leakage across splits')
            selected.append((src,dst,{**row,'training_approved':True,'reviewed':True,'clip_sha256':checksum,
                'review_file':decision_path.as_posix(),'review_sha256':file_hash(decision_path),'review_reason':decision['reason']}))
    if not {'train','val'}.issubset({row['split'] for _,_,row in selected}):raise ValueError('Need approved training and held-out clips')
    out.mkdir(parents=True)
    for src,dst,_ in selected:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    (out/'clips.jsonl').write_text(''.join(json.dumps(row)+'\n' for _,_,row in selected),encoding='utf-8')
    train=ClipDataset(out,'train');val=ClipDataset(out,'val')
    result=dict(train=len(train),val=len(val),reviewed_candidates=len(seen),approved=len(selected),
                caption_screen_sha256=file_hash(screen_path),rejected_by_final_caption_screen=caption_rejected)
    (out/'curation-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate_root');p.add_argument('out');p.add_argument('decisions',nargs='+')
    p.add_argument('--screen',required=True)
    print(json.dumps(build(**vars(p.parse_args()))))
