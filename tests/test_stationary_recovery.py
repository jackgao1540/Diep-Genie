import copy
import pytest
from scripts.build_stationary_recovery import validate_additions
from scripts.serve_footage_review import Library


def sample(block,start):
    return dict(source_id='source',block_id=block,video_sha256='hash',crop=[0,0,200,200],fps=10,source_fps=30,
                split='train',start=start,end=start+4,frames=40,source_frame_indices=list(range(start*30,(start+4)*30,3)),
                reasons=['low_motion_review'],ocr_frames=40,detections=[])


def test_recovery_requires_clean_bound_nonoverlapping_source_and_preserves_hold():
    base=[sample('000_000000',0)];addition=sample('000_000001',4)
    validate_additions(base,[addition],[])
    for change,message in [({'reasons':['low_motion_review','text']},'Unresolved screening'),
                           ({'ocr_frames':39},'Unresolved screening'),({'split':'test'},'Reserved test'),
                           ({'video_sha256':'other'},'Source binding'),({'block_id':'000_000000'},'Duplicate'),
                           ({'start':2},'Overlapping time'),({'source_frame_indices':[0]*40},'Overlapping frames')]:
        with pytest.raises(ValueError,match=message):validate_additions(base,[{**addition,**change}],[])
    with pytest.raises(ValueError,match='Existing exclusion'):validate_additions(base,[addition],[dict(source_id='source',start=0,end=100)])
    with pytest.raises(ValueError,match='Duplicate'):validate_additions(base,[addition,copy.deepcopy(addition)],[])


def test_recovery_keeps_original_frame_notes_but_rejects_changed_content():
    old = dict(sample('block', 0), proxy_sha256=None)
    digest = Library.content_hash(old)
    recovered = dict(old, proxy_sha256='reviewed', recovery_recipe='recipe')
    assert Library.feedback_matches(recovered, digest)
    assert not Library.feedback_matches(dict(recovered, crop=[1,0,200,200]), digest)
    assert not Library.feedback_matches(dict(old, proxy_sha256='unreviewed'), digest)
