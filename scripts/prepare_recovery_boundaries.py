"""Prepare unique native-rate joins for a selected recovery set."""
import argparse
from pathlib import Path
from scripts.prepare_stationary_recovery import read, rows, write, native_sheet
from scripts.build_stationary_recovery import key


def run(selection,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    selected=set(read(selection)); original=read('runs/remaining-low-motion-v1/candidates-native.json')
    evidence=[e for e in original['rows'] if e['queue_index'] in selected]
    base=rows('data/curated-longform-v7/manifest.jsonl'); added=[e['row'] for e in evidence]
    seen=set(); count=0
    for e in evidence:
        r=e['row']; e['boundaries']=[]
        for other in base+added:
            if key(r)==key(other) or other['source_id']!=r['source_id']: continue
            if abs(other['end']-r['start'])>=.6/r['source_fps'] and abs(r['end']-other['start'])>=.6/r['source_fps']: continue
            before,after=(other,r) if other['start']<r['start'] else (r,other)
            identity=(r['source_id'],before['block_id'],after['block_id'])
            if identity in seen: continue
            seen.add(identity); join=round(before['end']*r['source_fps'])
            ev=native_sheet(r['media_path'],r['crop'],join-6,join+6,out,f'join-{count:03}')
            e['boundaries'].append(dict(source_id=r['source_id'],before=before['block_id'],after=after['block_id'],
                                      before_sha256=before['proxy_sha256'],after_sha256=after['proxy_sha256'],evidence=ev))
            count+=1
    write(out/'native-index.json',dict(base_manifest_sha256=original['base_manifest_sha256'],rows=evidence))
    print(dict(clips=len(evidence),joins=count))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('selection');p.add_argument('out');a=p.parse_args();run(a.selection,a.out)
