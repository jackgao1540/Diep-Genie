import json
from pathlib import Path
import pytest
from genie_diep.data import file_hash
from scripts.preserve_screened_candidates import run, content_id, snapshot_class_evidence


def fixture(root):
    (root/'data').mkdir();(root/'configs').mkdir()
    media=root/'data/source.mp4';media.write_bytes(b'original retained, no decoding requested')
    row=dict(source_id='source',block_id='000_000000',split='train',video_sha256=file_hash(media),
             crop=[0,0,200,200],source_frame_indices=list(range(0,120,3)),frames=40,fps=10,start=0,reasons=['freeze'])
    for name in ['legacy','newer']:
        folder=root/'runs'/name/'source';folder.mkdir(parents=True)
        (folder/'configuration.json').write_text(json.dumps(dict(source=dict(id='source',media_path='data/source.mp4'),video_sha256=row['video_sha256'],screen_version=1)))
        (folder/'blocks.jsonl').write_text(json.dumps(row)+'\n')
    return row,media


def test_registry_deduplicates_content_preserves_decisions_and_excludes_test_pixels(tmp_path):
    row,_=fixture(tmp_path)
    active=tmp_path/'data/curated-longform-v6';active.mkdir()
    (active/'manifest.jsonl').write_text(json.dumps({**row,'split':'test'})+'\n')
    out=tmp_path/'data/preserved'
    summary=run(tmp_path,out,render=True)
    assert summary['screen_records']==2 and summary['unique_content_blocks']==1
    assert summary['originals_verified']==1 and summary['low_motion_previews']==0
    queue=json.loads((out/'low-motion-review.json').read_text())['rows'][0]
    assert queue['reserved_test'] and queue['in_current_cohort']
    assert not queue['training_approved'] and not queue['quality_approved']
    assert queue['id']==content_id(row)
    assert len(json.loads((out/'registry.jsonl').read_text())['observations'])==2
    for evidence in json.loads((out/'evidence.json').read_text()):
        assert file_hash(out/evidence['snapshot'])==evidence['sha256']
    with pytest.raises(FileExistsError):run(tmp_path,out,False)


def test_registry_reports_missing_original_and_never_claims_recovery(tmp_path):
    _,media=fixture(tmp_path);media.unlink()
    out=tmp_path/'data/preserved'
    with pytest.raises(ValueError,match='missing/changed'):run(tmp_path,out,False)
    report=json.loads((out/'summary.json').read_text())
    assert report['integrity_errors'] and report['originals_verified']==0
    assert report['low_motion_previews']==0 and not report['training_approved']


def test_class_snapshot_preserves_bound_index_without_changing_decisions(tmp_path):
    fixture(tmp_path);review=tmp_path/'runs/class-review';review.mkdir()
    index=review/'index.json';index.write_text('{"rows": []}')
    decision=tmp_path/'configs/longform-class-example.json'
    decision.write_text(json.dumps(dict(review='runs/class-review',index_sha256=file_hash(index),rejected_indices={'0':'future class'})))
    out=tmp_path/'data/class-evidence';result=snapshot_class_evidence(tmp_path,out)
    assert {r['original'] for r in result['evidence']}=={'runs/class-review/index.json','configs/longform-class-example.json'}
    for row in result['evidence']:assert file_hash(out/row['snapshot'])==row['sha256']
    index.write_text('{"rows": ["changed"]}')
    with pytest.raises(ValueError,match='index changed'):snapshot_class_evidence(tmp_path,tmp_path/'data/changed-class-evidence')
