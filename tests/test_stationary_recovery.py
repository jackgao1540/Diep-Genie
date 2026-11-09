import copy
import pytest
from scripts.build_stationary_recovery import validate_additions, validate_boundary, resolve_native_names
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


def test_recovery_joins_new_candidates_only_when_adjacent_and_content_bound():
    a=dict(sample('a',4),proxy_sha256='a-hash')
    b=dict(sample('b',8),proxy_sha256='b-hash')
    join=dict(source_id='source',before='a',after='b',before_sha256='a-hash',after_sha256='b-hash')
    assert validate_boundary(join,[a,b])==('source','a','b')
    with pytest.raises(ValueError,match='missing'):validate_boundary(join,[a])
    with pytest.raises(ValueError,match='changed'):validate_boundary(join,[a,dict(b,proxy_sha256='changed')])
    with pytest.raises(ValueError,match='not adjacent'):validate_boundary(join,[a,dict(b,start=12,end=16)])
    with pytest.raises(ValueError,match='source mismatch'):validate_boundary(join,[a,dict(b,split='test')])


def test_native_name_override_is_explicit_complete_and_preserves_evidence():
    screen=dict(reasons=['low_motion_review','text'],detections=[dict(frame=1,text='Player',native_name_allowed=False)])
    review=dict(native_name_overrides=[dict(detection_index=0,frame=1,text='Player',reason='Native player name verified visually')])
    result=resolve_native_names(screen,review)
    assert result['reasons']==['low_motion_review'] and result['detections'][0]['native_name_allowed']
    assert screen['reasons']==['low_motion_review','text'] and not screen['detections'][0]['native_name_allowed']
    with pytest.raises(ValueError,match='evidence mismatch'):
        resolve_native_names(screen,dict(native_name_overrides=[dict(review['native_name_overrides'][0],text='Other')]))
    with pytest.raises(ValueError,match='Unexpected screening'):
        resolve_native_names(dict(screen,reasons=['low_motion_review','text','cut']),review)
