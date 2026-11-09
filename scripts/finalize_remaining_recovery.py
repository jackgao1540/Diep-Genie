"""Bind the completed, explicitly reviewed remaining-candidate audit to v8."""
from collections import Counter
from pathlib import Path
from genie_diep.data import file_hash
from scripts.build_stationary_recovery import read, write, rows


def run():
    folder=Path('runs/remaining-low-motion-v1')
    index=read(folder/'final-native-index.json')
    groups=read(folder/'duplicate-groups.json')
    report=read(folder/'duplicates-final.json')
    review=read('configs/remaining-low-motion-review-v1.json')
    accepted={r['queue_index'] for r in index['rows']}
    for d in review['decisions']:
        if d['queue_index'] in accepted:
            d['decision']='accepted_v8'
            d['reason']='Clean ordinary shooting player; all native crop frames and original context reviewed. Fresh full retained-frame OCR, duplicate review and adjacent joins passed. Live surrounding activity distinguishes stationary gameplay from a frozen recording.'
        if d['queue_index'] in range(27,32):
            d['decision']='held_class_uncertain'
            d['reason']='Live clean footage preserved. Trapper-like muzzle and low-resolution class HUD do not establish ordinary-shooter eligibility confidently; retain separately for class verification/future scope.'
    review['decision_counts']=dict(Counter(d['decision'] for d in review['decisions']))
    review['parent_class_followup']={'source_id':'8f7hehx5w5g','start':3071.5,'end':3073.5,
        'reason':'Duplicate comparison exposed the same uncertain player class in inherited footage. Recheck the surrounding parent class interval before training; v8 does not alter immutable parent rows.'}
    write('configs/remaining-low-motion-review-v1.json',review)
    decisions=[]
    for i,g in enumerate(groups['groups']):
        if i in [0,1,2,3,4]:
            reason='Continuous same-scene idle gameplay at different times: original context shows evolving food, health and/or other-player positions; no repeated recording interval.'
        elif i==16:
            reason='Same purple Destroyer at different times, but different score/level and surrounding player/food positions; no reused recording.'
        elif i==19:
            reason='Different source scenes and player geometry: maze Destroyer versus Spread Shot; opponents, food, HUD and grid differ.'
        elif i==20:
            reason='Same basic player later in the session: level 11 versus level 13, score and food layout differ; no reused recording.'
        else:
            reason='Different player barrel geometry and/or score, level and arena/player/food layout in original context distinguish these same-color low-motion matches.'
        g.update(decision='distinct_gameplay',reason=reason)
        decisions.extend(dict(index=n,decision='distinct_gameplay',group=i,reason=reason) for n in g['flag_indices'])
    assert sorted(d['index'] for d in decisions)==list(range(len(report['near_runs'])))
    groups['review_method']='All 13 start/middle/end full-context comparison sheets inspected; 21 interval-pair groups cover every flag. This is grouped sampled duplicate review, supplemented by all native frames of candidate additions, not an exhaustive review of parent footage.'
    write('configs/remaining-low-motion-duplicate-review-v1.json',groups)
    evidence=[dict(path=s['path'],sha256=s['sha256']) for s in groups['segment_evidence']]
    evidence.append(dict(path='configs/remaining-low-motion-duplicate-review-v1.json',sha256=file_hash('configs/remaining-low-motion-duplicate-review-v1.json')))
    recipe=dict(schema_version=1,parent_cohort='data/curated-longform-v7',parent_checkpoint='configs/longform-seventh-checkpoint-v1.json',
        output_checkpoint='configs/longform-eighth-checkpoint-v1.json',native_index=(folder/'final-native-index.json').as_posix(),
        native_index_sha256=file_hash(folder/'final-native-index.json'),duplicate_report=(folder/'duplicates-final.json').as_posix(),
        duplicate_report_sha256=file_hash(folder/'duplicates-final.json'),accepted=[],duplicate_decisions=sorted(decisions,key=lambda d:d['index']),
        duplicate_review_evidence=evidence,training_approved=False,
        review='configs/remaining-low-motion-review-v1.json',review_sha256=file_hash('configs/remaining-low-motion-review-v1.json'))
    for e in index['rows']:
        recipe['accepted'].append(dict(source_id=e['row']['source_id'],block_id=e['row']['block_id'],
            queue_index=e['queue_index'],rescreen_sha256=e['rescreen_sha256'],all_native_frames_reviewed=True,
            ordinary_shooter=True,clean_gameplay=True,notes='All native crop frames, original context and all listed native-rate joins reviewed.',
            approved_boundaries=[{k:b[k] for k in ['source_id','before','after']} for b in e['boundaries']]))
        if e['queue_index']==25:
            screen=rows(e['row']['recovery_screen'])[0]
            recipe['accepted'][-1]['native_name_overrides']=[dict(detection_index=i,text=d['text'],frame=d['frame'],
                reason='Native DB Yagatov player label at the crop edge, confirmed against all native frames and original context; not an added caption.')
                for i,d in enumerate(screen['detections']) if not d.get('native_name_allowed')]
    write('configs/longform-cohort-v8.json',recipe)
    print(review['decision_counts'])


if __name__=='__main__':run()
