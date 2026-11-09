"""Compare selected recoveries against the complete parent and one another."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from genie_diep.data import file_hash
from scripts.prepare_stationary_recovery import read, rows, write
from scripts.audit_longform_duplicates import fingerprint, near_frames, matching_runs


def run(selection, out, parent='data/curated-longform-v7'):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    selected = read(selection)
    index = read('runs/remaining-low-motion-v1/candidates-native.json')
    added = [e['row'] for e in index['rows'] if e['queue_index'] in selected]
    assert len(added) == len(selected)
    base = rows(Path(parent)/'manifest.jsonl')
    for r in base:
        r['proxy_path'] = (Path(parent) / r['proxy_path']).as_posix()
    targets = []; exact = {}; matches = []; pairs = []
    for i, r in enumerate(added):
        with np.load(r['proxy_path'], allow_pickle=False) as d: video = d['frames']
        digest = hashlib.sha256(str(video.shape).encode()+video.tobytes()).hexdigest()
        exact.setdefault(digest, []).append(i)
        for f in range(0, 40, 5):
            code, small = fingerprint(video[f])
            targets.append((i, code, small, r['source_frame_indices'][f]/r['source_fps']))
    for bi, b in enumerate(base+added):
        if file_hash(b['proxy_path']) != b['proxy_sha256']: raise ValueError('Proxy changed')
        with np.load(b['proxy_path'], allow_pickle=False) as d: video = d['frames']
        digest = hashlib.sha256(str(video.shape).encode()+video.tobytes()).hexdigest()
        for ai in exact.get(digest, []):
            if bi != len(base)+ai: pairs.append(dict(base_index=bi, addition_index=ai))
        for f in range(0, 40, 5):
            code, small = fingerprint(video[f]); time = b['source_frame_indices'][f]/b['source_fps']
            for ai, target, pixels, t in targets:
                a = added[ai]
                if bi == len(base)+ai: continue
                if b['source_id'] == a['source_id'] and abs(time-t) < 8: continue
                if (code^target).bit_count() > 6 or not near_frames(small, pixels): continue
                matches.append(dict(source_a=b['source_id'], source_b=a['source_id'], time_a=time, time_b=t, split_a=b['split'], split_b=a['split']))
        if (bi+1) % 1000 == 0: print(json.dumps(dict(compared=bi+1)), flush=True)
    report = dict(base_manifest_sha256=file_hash(Path(parent)/'manifest.jsonl'), parent_cohort=parent, base_clips=len(base), additions=len(added),
                  selection=selection, selection_sha256=file_hash(selection), exact_pairs=pairs,
                  near_runs=matching_runs(matches), raw_matches=len(matches), sample_fps=2,
                  method='Every addition versus every parent block and other additions; whole-clip exact pixel hashes and 2 FPS foreground-aware near matching. Same-source offsets under 8 seconds skipped as adjacent gameplay; no bucket truncation.',
                  limitations='Sampled checks can miss transformed reuse. Reserved-test pixels are machine-compared only, never opened for development review.')
    write(out/'duplicates.json', report); write(out/'duplicate-index.json', dict(rows=base+added))
    print(json.dumps(dict(additions=len(added), exact_pairs=len(pairs), near_runs=len(report['near_runs']))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('selection'); p.add_argument('out')
    p.add_argument('--parent',default='data/curated-longform-v7')
    a=p.parse_args(); run(a.selection,a.out,a.parent)
