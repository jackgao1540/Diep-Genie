"""Validate saved preservation evidence and the four explicit recovery reviews."""
import json
from pathlib import Path
import numpy as np
from genie_diep.data import file_hash
from scripts.preserve_screened_candidates import content_id


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(root):
    root=Path(root);checkpoint=read(root/'configs/preservation-checkpoint-v1.json')
    bank=root/checkpoint['preservation_bank'];summary=read(bank/'summary.json')
    assert file_hash(bank/'summary.json')==checkpoint['summary_sha256']
    assert not summary['integrity_errors'] and not summary['training_approved']
    for name,sha in summary['hashes'].items():assert file_hash(bank/name)==sha,name
    for name,sha in checkpoint['unchanged_cohort_hashes'].items():assert file_hash(root/checkpoint['cohort']/name)==sha,name
    registry={r['id']:r for r in [json.loads(s) for s in (bank/'registry.jsonl').read_text().splitlines()]}
    assert len(registry)==summary['unique_content_blocks']
    for entry in registry.values():
        assert entry['preservation_required'] and not entry['training_approved']
        assert all(content_id(o['record'])==entry['id'] for o in entry['observations'])
    for evidence in read(bank/'evidence.json'):
        assert file_hash(bank/evidence['snapshot'])==evidence['sha256']
    class_index=root/checkpoint['class_evidence_index']
    assert file_hash(class_index)==checkpoint['class_evidence_sha256']
    class_evidence=read(class_index)['evidence']
    for evidence in class_evidence:assert file_hash(class_index.parent/evidence['snapshot'])==evidence['sha256']
    class_paths={e['original'] for e in class_evidence}
    recipe=read(root/'configs/longform-cohort-v6.json')
    assert set(recipe['class_reviews'])<=class_paths
    for path in recipe['class_reviews']:
        decision=read(root/path)
        if 'review' in decision:
            index=Path(decision['review'])/'index.json'
            assert index.as_posix().replace('\\','/') in class_paths
            assert file_hash(root/index)==decision['index_sha256']
    queue=read(bank/'low-motion-review.json')['rows'];by_id={r['id']:r for r in queue}
    assert len(by_id)==len(queue)==summary['low_motion_blocks']
    for row in queue:
        assert row['id'] in registry and not row['training_approved'] and not row['quality_approved']
        if row['reserved_test']:assert 'review_artifacts' not in row
        for name,artifact in row.get('review_artifacts',{}).items():
            assert file_hash(bank/artifact['path'])==artifact['sha256']
            if name=='frames.npz':
                with np.load(bank/artifact['path'],allow_pickle=False) as data:
                    assert data['frames'].shape==(len(row['source_frame_indices']),128,128,3)
                    assert data['frames'].dtype==np.uint8
    material=root/checkpoint['rescreen_material']
    assert file_hash(material)==checkpoint['rescreen_material_sha256']
    reviewed=read(material)['rows']
    assert len(reviewed)==len(checkpoint['priority_reviews'])==4
    for decision in checkpoint['priority_reviews']:
        r=next(r for r in reviewed if r['source_id']==decision['source_id'] and r['original_block_id']==decision['block_id'])
        q=by_id[decision['preservation_id']]
        assert q['source_id']==r['source_id'] and q['source_frame_indices']==r['source_frame_indices']
        assert r['ocr_frames']==len(r['source_frame_indices'])==40
        assert file_hash(root/r['rescreen'])==r['rescreen_sha256']
        record=json.loads((root/r['rescreen']).read_text())
        proxy=(root/r['rescreen']).parent/record['proxy']
        assert file_hash(proxy)==r['proxy_sha256']
        with np.load(proxy,allow_pickle=False) as a,np.load(bank/q['review_artifacts']['frames.npz']['path'],allow_pickle=False) as b:
            assert np.array_equal(a['frames'],b['frames'])
        assert decision['training_approved'] is False and decision['cohort_action']=='none'
        if decision['recovery_status']=='visual_candidate':assert r['reasons']==['low_motion_review']
        else:assert decision['recovery_status']=='preserved_overlay' and 'text' in r['reasons']
        original=r['visual_evidence'];directory=Path(checkpoint['focused_index']).parent
        for name in ['sheet','context']:assert file_hash(root/directory/original[name])==original[name+'_sha256']
    result=dict(validation='passed',preserved_blocks=len(registry),historical_rejects=summary['historically_rejected_blocks'],
                low_motion=len(queue),previews=summary['low_motion_previews'],priority_reviews=4,cohort_changed=False)
    print(json.dumps(result));return result


if __name__=='__main__':validate(Path(__file__).resolve().parents[1])
