from genie_diep.screen_captions import text_candidate,intersects_crop,score_caption_candidate


def test_numeric_caption_misses_without_reintroducing_known_food_misreads():
    box=[[73,133],[145,134],[144,167],[73,165]]
    for text,score in [('980k',.99989),('994k°',.92814),('1000',.95)]:
        assert not text_candidate(text,score,box)
        assert score_caption_candidate(text,score,box)
    for text in ('44','70'):
        assert not score_caption_candidate(text,.99,box)
    assert not score_caption_candidate('980k',.6,box)


def test_stronger_caption_policy_cannot_reuse_legacy_negatives(tmp_path):
    from types import SimpleNamespace
    import pytest
    from scripts.screen_longform import screen_source
    args=SimpleNamespace(caption_policy='score-captions',previous_screen=['old'])
    with pytest.raises(ValueError,match='cannot reuse'):
        screen_source({},args,None,'recipe')
import json
import pytest
from genie_diep.data import ClipDataset


def test_food_misread_as_digits_does_not_trigger_caption_rejection():
    box=[[0,0],[30,0],[30,20],[0,20]]
    assert not text_candidate('70',.99,box)
    assert not text_candidate('44',.99,box)
    assert text_candidate('Hmmm',.99,box)
    assert text_candidate('Must hide',.8,box)
    assert not text_candidate('Hmmm',.3,box)


def test_unapproved_candidate_cannot_enter_training(tmp_path):
    row=dict(source_id='candidate',split='train',shape=[16,64,64,3],path='candidate.npy',training_approved=False)
    (tmp_path/'clips.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='Unapproved candidate'):ClipDataset(tmp_path,'train')


def test_padded_ocr_rejects_intersecting_caption_but_ignores_outside_hud():
    assert intersects_crop([[-20,190],[223,190],[223,220],[-20,220]],200,200)
    assert not intersects_crop([[10,201],[100,201],[100,220],[10,220]],200,200)
