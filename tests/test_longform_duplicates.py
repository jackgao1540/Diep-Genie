import json
import cv2
import numpy as np
import pytest
from genie_diep.data import file_hash
from scripts.audit_longform_duplicates import audit,near_frames,matching_runs
from scripts.review_longform import candidates
from scripts.build_longform_cohort import indices


def test_review_coverage_ranges_cannot_hide_gaps_or_overlap():
    assert indices([[0,2],[4,5]],6)=={0,1,2,4,5}
    with pytest.raises(ValueError,match='Overlapping'):indices([[0,2],[2,4]],5)
    with pytest.raises(ValueError,match='outside'):indices([[0,5]],5)


def test_offset_reuse_crosses_clip_boundaries(tmp_path):
    rng=np.random.default_rng(42);video=[]
    for i in range(80):
        frame=np.full((64,64,3),180,np.uint8)
        for j in range(8):
            x,y=rng.integers(4,60,2);color=tuple(int(v) for v in rng.integers(20,240,3))
            cv2.circle(frame,(int(x),int(y)),3,color,-1)
        video.append(frame)
    video=np.array(video);rows=[]
    # Reused content starts three seconds into a block and crosses the next
    # boundary, with changed encoding brightness and unrelated source timing.
    for k,(source,start,frames) in enumerate([('a',0,video[:40]),('a',4,video[40:]),('b',100,video[30:70]+1)]):
        path=tmp_path/f'{k}.npz';np.savez_compressed(path,frames=frames)
        rows.append(dict(source_id=source,split='train' if source=='a' else 'test',
            proxy_path=str(path),fps=10,source_fps=10,source_frame_indices=list(range(start*10,start*10+40))))
    report=audit(rows)
    assert not report['exact_pairs']
    assert any(r['cross_split'] and r['end_a']-r['start_a']>=3.5 for r in report['near_runs'])
    gray=np.full((32,32,3),180,np.uint8)
    assert not near_frames(gray,gray)
    # Isolated matching scenes must not be joined over missing footage.
    hits=[dict(source_a='a',source_b='b',time_a=t,time_b=t+100,split_a='train',split_b='test') for t in [0,.5,5,5.5]]
    assert not matching_runs(hits)


def test_audit_candidates_require_matching_class_evidence(tmp_path):
    root=tmp_path/'screen';folder=root/'a';folder.mkdir(parents=True)
    proxy=folder/'clip.npz';proxy.write_bytes(b'pixels')
    row=dict(source_id='a',block_id='000_000000',reasons=[],proxy=proxy.name,proxy_sha256=file_hash(proxy))
    (folder/'blocks.jsonl').write_text(json.dumps(row)+'\n')
    review=tmp_path/'review.json'
    decision=dict(source_id='a',block_id=row['block_id'],class_candidate=False,proxy_sha256=row['proxy_sha256'])
    review.write_text(json.dumps({'rows':[decision]}))
    assert candidates([root],[review])==[]
    decision.update(class_candidate=True,proxy_sha256='changed');review.write_text(json.dumps({'rows':[decision]}))
    with pytest.raises(ValueError,match='hash mismatch'):candidates([root],[review])
    decision['proxy_sha256']=row['proxy_sha256'];review.write_text(json.dumps({'rows':[decision]}))
    assert len(candidates([root],[review]))==1
    with pytest.raises(ValueError,match='Repeated source block'):candidates([root,root],[review])
