from genie_diep.screen_captions import text_candidate,intersects_crop
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
