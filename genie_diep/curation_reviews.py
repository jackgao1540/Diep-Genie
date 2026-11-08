"""Read explicit or compact content-bound human class review decisions."""
import json
from pathlib import Path
from genie_diep.data import file_hash


def load_class_review(path):
    review=json.loads(Path(path).read_text(encoding='utf-8'))
    if review.get('invalidated_reason'):raise ValueError('Invalidated class review: '+review['invalidated_reason'])
    if 'rows' in review:return review['rows']
    index_path=Path(review['review'])/'index.json'
    if file_hash(index_path)!=review['index_sha256']:raise ValueError('Changed class review index')
    rows=json.loads(index_path.read_text(encoding='utf-8'))['rows'];decisions={}
    for start,end in review['accepted_ranges']:
        if not (0<=start<=end<len(rows)):raise ValueError('Class review range outside index')
        for i in range(start,end+1):
            if i in decisions:raise ValueError('Overlapping class review ranges')
            decisions[i]=(True,'Ordinary shooter in displayed class samples')
    for key,reason in review.get('rejected_indices',{}).items():
        i=int(key)
        if i in decisions or not 0<=i<len(rows):raise ValueError('Conflicting class decision')
        decisions[i]=(False,reason)
    return [dict(source_id=rows[i]['source_id'],block_id=rows[i]['block_id'],
                 proxy_sha256=rows[i]['proxy_sha256'],review_index=i,class_candidate=decision,
                 reason=reason,training_approved=False) for i,(decision,reason) in sorted(decisions.items())]
