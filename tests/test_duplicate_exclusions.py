import pytest
from scripts.build_longform_cohort import validate_duplicate_exclusions


def test_confirmed_reuse_requires_a_complete_withheld_side():
    rows=[dict(source_id=s,block_id=str(i),start=i*4,end=(i+1)*4)
          for s in ('original','copy') for i in range(2)]
    report=dict(exact_pairs=[],near_runs=[dict(source_a='original',source_b='copy',
        start_a=3,end_a=5,start_b=3,end_b=5)])
    for excluded in [set(),{('copy','0')},{('copy','0'),('original','1')}]:
        with pytest.raises(ValueError,match='both sides'):
            validate_duplicate_exclusions(rows,report,[0],excluded)
    validate_duplicate_exclusions(rows,report,[0],{('copy','0'),('copy','1')})
    with pytest.raises(ValueError,match='Invalid'):
        validate_duplicate_exclusions(rows,report,[1],set())
    with pytest.raises(ValueError,match='Repeated'):
        validate_duplicate_exclusions(rows,report,[0,0],set())


def test_missing_source_is_not_a_resolved_duplicate():
    report=dict(exact_pairs=[],near_runs=[dict(source_a='absent',source_b='copy',
        start_a=0,end_a=2,start_b=0,end_b=2)])
    rows=[dict(source_id='copy',block_id='0',start=0,end=4)]
    with pytest.raises(ValueError,match='both sides'):
        validate_duplicate_exclusions(rows,report,[0],set())
    report['exact_pairs']=[dict(a=0,b=1)]
    with pytest.raises(ValueError,match='Exact'):
        validate_duplicate_exclusions(rows,report,[],{('copy','0')})
