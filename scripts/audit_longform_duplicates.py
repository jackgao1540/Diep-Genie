"""Exact clips and offset-aware near-duplicate triage, never automatic approval.

Frame hashes find possible reuse across block boundaries. Consecutive matches
must agree on source-time offset and RGB foreground. Sampling and hash lookup
can miss rescaled, recolored, speed-changed or heavily edited reuse.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.review_longform import candidates


def fingerprint(frame):
    small=cv2.resize(frame,(32,32),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY).astype(np.float32)
    freq=cv2.dct(gray)[:8,:8].reshape(-1)
    bits=freq>np.median(freq[1:]);bits[0]=False
    code=int.from_bytes(np.packbits(bits).tobytes(),'big')
    return code,small


def near_frames(a,b):
    # Empty gray areas must not hide disagreement on the small game objects.
    af=a.astype(np.float32)/255;bf=b.astype(np.float32)/255
    foreground=(np.ptp(a,axis=2)>30)|(np.ptp(b,axis=2)>30)
    if foreground.sum()<8:return False
    diff=(af-bf)**2
    return bool(diff.mean()<.0015 and diff[foreground].mean()<.012)


def matching_runs(matches,step=.5):
    groups=defaultdict(list)
    for m in matches:
        # Bin offsets at the sampling resolution; rounding allows source FPS
        # timestamp jitter, not arbitrary gaps or speed changes.
        groups[(m['source_a'],m['source_b'],round((m['time_b']-m['time_a'])/step))].append(m)
    runs=[]
    def emit(chain):
        if len(chain)>=4 and chain[-1]['time_a']-chain[0]['time_a']>=1.5-1e-4:
            runs.append(dict(source_a=chain[0]['source_a'],source_b=chain[0]['source_b'],
                start_a=chain[0]['time_a'],end_a=chain[-1]['time_a'],
                start_b=chain[0]['time_b'],end_b=chain[-1]['time_b'],matches=len(chain),
                cross_split=chain[0]['split_a']!=chain[0]['split_b'],review_required=True))
    for group in groups.values():
        chain=[]
        for m in sorted(group,key=lambda x:(x['time_a'],x['time_b'])):
            if chain and not (0<m['time_a']-chain[-1]['time_a']<=step*1.1 and
                              0<m['time_b']-chain[-1]['time_b']<=step*1.1):
                emit(chain);chain=[]
            chain.append(m)
        emit(chain)
    return runs


def audit(rows,sample_fps=2,bucket_limit=5000):
    exact={};exact_pairs=[];frames=[];buckets=defaultdict(list);matches=[];saturated=set();low_info=0
    for ri,row in enumerate(rows):
        with np.load(row['proxy_path'],allow_pickle=False) as data:video=data['frames']
        digest=hashlib.sha256(str(video.shape).encode()+video.tobytes()).hexdigest()
        if digest in exact:exact_pairs.append(dict(a=exact[digest],b=ri,review_required=True))
        else:exact[digest]=ri
        stride=round(row['fps']/sample_fps)
        if stride<1 or abs(row['fps']/stride-sample_fps)>1e-6:raise ValueError('Sampling rate must divide retained FPS')
        for index in range(0,len(video),stride):
            code,small=fingerprint(video[index]);time=row['source_frame_indices'][index]/row['source_fps']
            if (np.ptp(small,axis=2)>30).sum()<8:low_info+=1;continue
            keys=[(band,(code>>(band*16))&65535) for band in range(4)]
            possible=set()
            for key in keys:
                if len(buckets[key])>=bucket_limit:saturated.add(key)
                else:possible.update(buckets[key])
            for old_index in sorted(possible):
                old=frames[old_index];other=rows[old['row']]
                if other['source_id']==row['source_id'] and abs(time-old['time'])<8:continue
                if (code^old['hash']).bit_count()>6 or not near_frames(small,old['pixels']):continue
                matches.append(dict(source_a=other['source_id'],source_b=row['source_id'],
                    time_a=old['time'],time_b=time,split_a=other['split'],split_b=row['split']))
            fi=len(frames);frames.append(dict(row=ri,time=time,hash=code,pixels=small))
            for key in keys:
                if len(buckets[key])<bucket_limit:buckets[key].append(fi)
        if (ri+1)%200==0:print(json.dumps(dict(clips_scanned=ri+1,frame_matches=len(matches))),flush=True)
    return dict(exact_pairs=exact_pairs,near_runs=matching_runs(matches,1/sample_fps),
        clips=len(rows),sampled_informative_frames=len(frames),low_information_frames=low_info,
        raw_frame_matches=len(matches),saturated_hash_buckets=len(saturated),
        sample_fps=sample_fps,limitations=__doc__,training_approved=False)


def run(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new immutable duplicate report directory')
    rows=candidates(args.screens,args.class_reviews)
    report=audit(rows,args.sample_fps,args.bucket_limit)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'index.json').write_text(json.dumps(dict(arguments=vars(args),rows=rows),indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('exact_pairs','near_runs')},indent=2))
    print(json.dumps(dict(exact_pairs=len(report['exact_pairs']),near_runs=len(report['near_runs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('screens',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--class-reviews',nargs='+',required=True);p.add_argument('--sample-fps',type=float,default=2)
    p.add_argument('--bucket-limit',type=int,default=5000);run(p.parse_args())
