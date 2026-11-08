"""Report contiguous candidate time without bridging rejected blocks or cuts.

These are potential sequences, not approval: metadata adjacency cannot establish
that an edited source has no cut. Optional pixel boundary checks are also triage.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def sequence_runs(rows,boundary_check=None):
    groups=defaultdict(list)
    for row in rows:
        if not row['reasons']:groups[row['source_id']].append(row)
    runs=[]
    for source,blocks in sorted(groups.items()):
        blocks.sort(key=lambda r:r['start']);current=[];previous=None
        for row in blocks:
            adjacent=False
            if previous:
                pa,pb=map(int,previous['block_id'].split('_'));ra,rb=map(int,row['block_id'].split('_'))
                adjacent=(pa==ra and rb==pb+1 and previous['video_sha256']==row['video_sha256']
                          and previous['crop']==row['crop'] and previous['fps']==row['fps']
                          and abs(row['start']-previous['end'])<=.6/row['source_fps']+1e-6)
                if adjacent and boundary_check is not None:adjacent=boundary_check(previous,row)
            if current and not adjacent:
                runs.append(make_run(source,current));current=[]
            current.append(row);previous=row
        if current:runs.append(make_run(source,current))
    return runs


def make_run(source,rows):
    return dict(source_id=source,split=rows[0]['split'],start=rows[0]['start'],end=rows[-1]['end'],
                seconds=sum(r['frames']/r['fps'] for r in rows),blocks=[r['block_id'] for r in rows],
                training_approved=False)


def run(args):
    rows=[];seen=set();class_decisions={}
    for p in args.class_reviews:
        for r in json.loads(Path(p).read_text(encoding='utf-8'))['rows']:
            key=(r['source_id'],r['block_id']);old=class_decisions.get(key)
            if old and old!=r:raise ValueError('Conflicting class decisions')
            class_decisions[key]=r
    for root in args.screens:
        for p in sorted(Path(root).glob('*/blocks.jsonl')):
            for line in p.read_text(encoding='utf-8').splitlines():
                r=json.loads(line);key=(r['source_id'],r['block_id'])
                if key in seen:raise ValueError('Duplicate source block across screens')
                seen.add(key)
                review=class_decisions.get(key)
                if review:
                    if r.get('proxy_sha256')!=review['proxy_sha256']:raise ValueError('Class review hash mismatch')
                    if not review['class_candidate']:continue
                elif args.require_class_review:continue
                if not r['reasons']:r['proxy_path']=str(p.parent/r['proxy'])
                rows.append(r)
    boundary_rejections=[]
    def check(a,b):
        first=np.load(a['proxy_path'],allow_pickle=False)['frames'][-1]
        second=np.load(b['proxy_path'],allow_pickle=False)['frames'][0]
        delta=float(np.abs(first.astype(np.float32)-second.astype(np.float32)).mean()/255)
        if delta>.12:
            boundary_rejections.append(dict(source_id=a['source_id'],before=a['block_id'],after=b['block_id'],delta=delta));return False
        return True
    runs=sequence_runs(rows,check if args.check_boundaries else None)
    durations=np.array([r['seconds'] for r in runs])
    summary=dict(candidate_hours=float(durations.sum()/3600),sources=len(set(r['source_id'] for r in runs)),
                 runs=len(runs),longest_seconds=float(durations.max()) if len(durations) else 0,
                 seconds_in_runs_at_least={str(n):float(durations[durations>=n].sum()) for n in (4,10,20,30,60)},
                 run_counts_at_least={str(n):int((durations>=n).sum()) for n in (4,10,20,30,60)},
                 boundary_checks=args.check_boundaries,require_class_review=args.require_class_review,
                 warning=__doc__,training_approved=False)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'sequences.json').write_text(json.dumps(dict(summary=summary,runs=runs,boundary_rejections=boundary_rejections),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('screens',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--class-reviews',nargs='*',default=[]);p.add_argument('--require-class-review',action='store_true')
    p.add_argument('--check-boundaries',action='store_true');run(p.parse_args())
