import json
import pytest
from scripts.review_original_classes import unreviewed_rows


def test_prior_rejections_are_not_reintroduced(tmp_path):
    rows=[dict(source_id='source',block_id=str(i),proxy_sha256=str(i)) for i in range(3)]
    review=tmp_path/'review.json'
    review.write_text(json.dumps({'rows':[dict(rows[0],class_candidate=True),dict(rows[1],class_candidate=False)]}))
    assert unreviewed_rows(rows,[review])==[rows[2]]
    changed=[dict(rows[0],proxy_sha256='changed')]
    with pytest.raises(ValueError,match='evidence changed'):
        unreviewed_rows(changed,[review])
    with pytest.raises(ValueError,match='Repeated prior'):
        unreviewed_rows(rows,[review,review])
