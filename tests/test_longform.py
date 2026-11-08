import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from genie_diep.data import file_hash
from scripts.screen_longform import crop_frame, player_present, screen_source
from scripts.report_sequence_lengths import sequence_runs
from genie_diep.native_text import native_name
from genie_diep.curation_reviews import load_class_review


def test_compact_review_is_bound_to_index_and_rejects_overlap(tmp_path):
    index=tmp_path/'index.json'
    index.write_text(json.dumps({'rows':[dict(source_id='a',block_id='000_000000',proxy_sha256='pixels')]}))
    review=tmp_path/'review.json'
    config=dict(review=str(tmp_path),index_sha256=file_hash(index),accepted_ranges=[[0,0]],rejected_indices={})
    review.write_text(json.dumps(config))
    rows=load_class_review(review)
    assert len(rows)==1 and rows[0]['proxy_sha256']=='pixels' and not rows[0]['training_approved']
    review.write_text(json.dumps({**config,'accepted_ranges':[[0,0],[0,0]]}))
    with pytest.raises(ValueError,match='Overlapping'):load_class_review(review)
    review.write_text(json.dumps(config));index.write_text('{}')
    with pytest.raises(ValueError,match='Changed'):load_class_review(review)


def test_native_name_requires_small_outlined_font_and_body():
    frame=np.full((360,640,3),185,np.uint8)
    cv2.circle(frame,(320,190),12,(220,170,20),-1)
    for color,thickness in (((30,30,30),3),((255,255,255),1)):
        cv2.putText(frame,'Player',(303,171),cv2.FONT_HERSHEY_SIMPLEX,.3,color,thickness,cv2.LINE_AA)
    box=np.array([[300,158],[337,158],[337,176],[300,176]])
    assert native_name(frame,box)
    without_body=frame.copy();without_body[178:210,300:340]=185
    assert not native_name(without_body,box)
    caption=frame.copy();caption[155:178,298:340]=185
    cv2.putText(caption,'Player',(303,171),cv2.FONT_HERSHEY_SIMPLEX,.3,(220,50,50),1,cv2.LINE_AA)
    assert not native_name(caption,box)
    assert not native_name(frame,np.array([[200,120],[440,120],[440,160],[200,160]]))


def test_contiguous_runs_do_not_bridge_missing_rejected_or_new_intervals():
    rows=[dict(source_id='a',split='train',block_id=f'000_{i:06}',start=i*4,end=(i+1)*4,
               frames=40,fps=10,source_fps=30,crop=[0,0,128,128],video_sha256='sha',reasons=[])
          for i in range(7)]
    rows[2]['reasons']=['text']
    rows[6]['block_id']='001_000000'
    runs=sequence_runs(rows)
    assert [r['seconds'] for r in runs]==[8,12,4]
    assert all(not r['training_approved'] for r in runs)
    runs=sequence_runs(rows,lambda a,b:b['start']!=16)
    assert [r['seconds'] for r in runs]==[8,4,8,4]


def test_player_palette_and_crop_padding():
    for color in ((220,170,20),(40,200,40),(40,40,220),(220,80,180)):
        frame=np.full((128,128,3),180,np.uint8)
        cv2.circle(frame,(64,64),9,color,-1)
        assert player_present(frame)
    frame[:]=180;cv2.rectangle(frame,(58,58),(70,70),(30,220,240),-1)
    assert not player_present(frame)  # yellow square is not a player
    crop,padded,origin=crop_frame(frame,[4,5,100,100],24)
    assert crop.shape==(100,100,3) and padded.shape==(128,128,3)
    assert origin.tolist()==[4,5]
    with pytest.raises(ValueError,match='outside'):crop_frame(frame,[60,60,100,100],24)


def test_every_frame_ocr_rejection_and_resume_integrity(tmp_path,monkeypatch):
    video=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,160))
    assert writer.isOpened()
    for i in range(80):
        frame=np.full((160,160,3),180,np.uint8)
        cv2.circle(frame,(80,80),10,(220,170,20),-1)
        cv2.rectangle(frame,(18+i%30,30),(28+i%30,40),(30,220,240),-1)
        writer.write(frame)
    writer.release()
    source=dict(id='source',media_path=str(video),video_sha256=file_hash(video),
                split='train',crop=[16,16,128,128],intervals=[[0,8]])
    args=SimpleNamespace(out=str(tmp_path/'screen'),fps=10,block_seconds=4,padding=24,proxy_size=64)
    class Engine:
        calls=0
        def __call__(self,frame):
            self.calls+=1
            if self.calls==79:
                return SimpleNamespace(boxes=np.array([[[30,30],[65,30],[65,45],[30,45]]]),txts=['caption'],scores=[.99])
            return SimpleNamespace(boxes=None)
    engine=Engine();screen_source(source,args,engine,'recipe-sha')
    rows=[json.loads(s) for s in (tmp_path/'screen/source/blocks.jsonl').read_text().splitlines()]
    assert len(rows)==2 and rows[0]['ocr_frames']==40 and rows[0]['reasons']==[]
    assert rows[1]['ocr_frames']==39 and rows[1]['reasons']==['text']
    assert not any(r['training_approved'] for r in rows)
    screen_source(source,args,engine,'recipe-sha')
    assert engine.calls==79  # no duplicate work or duplicate accepted seconds
    recovered=SimpleNamespace(**vars(args));recovered.out=str(tmp_path/'recovered')
    recovered.text_policy='native-names';recovered.previous_screen=[args.out]
    second_engine=Engine();screen_source(source,recovered,second_engine,'recipe-sha')
    recovered_rows=[json.loads(s) for s in (tmp_path/'recovered/source/blocks.jsonl').read_text().splitlines()]
    assert second_engine.calls==40  # strict negative reused, text-positive block rescreened in FULL
    assert recovered_rows[0]['proxy_sha256']==rows[0]['proxy_sha256']
    assert recovered_rows[0]['reused_strict_screen']
    class LateCaption:
        calls=0
        def __call__(self,frame):
            self.calls+=1
            if self.calls in (1,39):
                x=20 if self.calls==1 else 80
                return SimpleNamespace(boxes=np.array([[[x,30],[x+35,30],[x+35,45],[x,45]]]),txts=['label'],scores=[.99])
            return SimpleNamespace(boxes=None)
    late_args=SimpleNamespace(**vars(args));late_args.out=str(tmp_path/'late-caption');late_args.text_policy='native-names'
    monkeypatch.setattr('scripts.screen_longform.native_name',lambda frame,box:box[0,0]<60)
    late_engine=LateCaption();screen_source({**source,'intervals':[[0,4]]},late_args,late_engine,'late-recipe')
    late_row=json.loads((tmp_path/'late-caption/source/blocks.jsonl').read_text())
    assert late_row['reasons']==['text'] and late_row['ocr_frames']==39
    assert late_row['detections'][0]['native_name_allowed']
    assert not late_row['detections'][1]['native_name_allowed']
    with pytest.raises(ValueError,match='configuration changed'):screen_source(source,args,engine,'different-recipe')
    (tmp_path/'screen/source'/rows[0]['proxy']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='missing or changed'):screen_source(source,args,engine,'recipe-sha')
