import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from genie_diep.data import file_hash
from scripts.screen_longform import cheap_reasons, screen_source
from genie_diep.data import extract, ClipDataset


def test_low_motion_is_review_not_proof_of_frozen_recording():
    frames=[]
    for i in range(40):
        frame=np.full((200,200,3),180,np.uint8)
        cv2.circle(frame,(100,100),12,(220,170,20),-1)
        cv2.line(frame,(100,100),(119,100+i//20),(70,70,70),4)
        frames.append(frame)
    reasons,metrics=cheap_reasons(frames)
    assert 'low_motion_review' in reasons and 'freeze' not in reasons
    assert metrics['exact_repeat_fraction']<1 and metrics['longest_exact_repeat_frames']==20
    assert 'freeze' in cheap_reasons(frames,'legacy')[0]
    still=[frames[0]]*40
    assert 'low_motion_review' in cheap_reasons(still)[0]
    assert cheap_reasons(still)[1]['longest_exact_repeat_frames']==40


def test_review_preserves_all_rejects_and_checks_late_overlay(tmp_path):
    video=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,160))
    assert writer.isOpened()
    for i in range(80):
        frame=np.full((160,160,3),180,np.uint8)
        if i<40:
            cv2.circle(frame,(80,80),10,(220,170,20),-1)
            cv2.line(frame,(80,80),(97,80),(70,70,70),4)
        writer.write(frame)
    writer.release()
    source=dict(id='source',media_path=str(video),video_sha256=file_hash(video),split='train',crop=[16,16,128,128],intervals=[[0,8]])
    args=SimpleNamespace(out=str(tmp_path/'screen'),fps=10,block_seconds=4,padding=24,proxy_size=64)
    class Engine:
        calls=0
        def __call__(self,frame):
            self.calls+=1
            if self.calls in (1,40):
                return SimpleNamespace(boxes=np.array([[[30,30],[65,30],[65,45],[30,45]]]),txts=['caption'],scores=[.99])
            return SimpleNamespace(boxes=None)
    engine=Engine();screen_source(source,args,engine,'recipe')
    rows=[json.loads(s) for s in (tmp_path/'screen/source/blocks.jsonl').read_text().splitlines()]
    assert rows[0]['reasons']==['low_motion_review','text']
    assert rows[0]['ocr_frames']==40 and len(rows[0]['detections'])==2
    assert 'no_center_body' in rows[1]['reasons']
    for row in rows:
        assert not row['training_approved'] and not row['quality_approved']
        assert row['preservation_required'] and row['disposition']=='manual_review'
        assert file_hash(tmp_path/'screen/source'/row['proxy'])==row['proxy_sha256']
    screen_source(source,args,engine,'recipe');assert engine.calls==40
    (tmp_path/'screen/source'/rows[1]['proxy']).write_bytes(b'changed rejected proxy')
    with pytest.raises(ValueError,match='missing or changed'):screen_source(source,args,engine,'recipe')


def test_small_extractor_also_preserves_flags_outside_training_manifest(tmp_path):
    video=tmp_path/'still.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
    assert writer.isOpened()
    for _ in range(20):writer.write(np.full((64,64,3),180,np.uint8))
    writer.release()
    out=tmp_path/'extract'
    assert extract(video,out,'still',seconds=2,length=10)==[]
    records=[json.loads(s) for s in (out/'rejected/still.jsonl').read_text().splitlines()]
    assert len(records)==2
    assert all(r['reasons']==['low_motion_review'] and not r['training_approved'] for r in records)
    assert all(file_hash(out/'rejected'/r['path'])==r['proxy_sha256'] for r in records)
    with pytest.raises(ValueError,match='No train clips'):ClipDataset(out,'train')
