import copy

import pytest
import torch

from genie_diep.baseline import Trainer
from genie_diep.cohort import read
from scripts.gpu_recovery_verification import compare_states, migrate_legacy_failure, verification_budget
from test_baseline_pipeline import project


def test_diagnostic_migration_recovers_consumed_batch_and_preserves_source(project):
    source=Trainer(project,'spec.json','tokenizer',preflight=True)
    source.update();before=source._update_boundary()
    expected_windows=[]
    for _ in range(source.microbatch*source.accumulation):expected_windows.append(source.sampler.draw())
    source.save(project/'legacy.pt')
    state=torch.load(project/'legacy.pt',weights_only=True);unchanged=copy.deepcopy(state)
    migrated=Trainer(project,'spec.json','tokenizer',preflight=True)
    report=migrate_legacy_failure(migrated,state,read(project/'spec.json'))
    assert compare_states(migrated._update_boundary(),before,rtol=0,atol=0)==0
    assert report['recovered_sampler_draws']==before['sampler']['draws']
    assert report['recovered_windows']==[source.data.descriptor(*w) for w in expected_windows]
    assert report['preflight'] and not report['promotion_approved']
    assert compare_states(state,unchanged,rtol=0,atol=0)==0
    migrated.save(project/'migrated.pt')
    expected_loss=migrated.update();expected=migrated._update_boundary()
    resumed=Trainer(project,'spec.json','tokenizer',preflight=True,resume=project/'migrated.pt')
    assert resumed.update()==expected_loss
    assert compare_states(resumed._update_boundary(),expected,rtol=0,atol=0)==0


def test_migration_refuses_production_and_wrong_boundary(project):
    trainer=Trainer(project,'spec.json','tokenizer',preflight=True);trainer.save(project/'state.pt')
    state=torch.load(project/'state.pt',weights_only=True)
    with pytest.raises(ValueError,match='fully consumed'):
        migrate_legacy_failure(trainer,state,read(project/'spec.json'))
    trainer.preflight=False
    with pytest.raises(ValueError,match='diagnostic tokenizer'):
        migrate_legacy_failure(trainer,state,read(project/'spec.json'))


@pytest.mark.parametrize('rate,balance,start',[(0,100,1),(1,50,1),(1,100,201),(float('nan'),100,1),(1,100,0)])
def test_verification_rejects_invalid_allocation(rate,balance,start):
    with pytest.raises(ValueError):verification_budget(rate,balance,start,now=200)


def test_verification_budget_counts_setup_and_preserves_export_and_units(monkeypatch):
    monkeypatch.setattr('genie_diep.baseline.time.monotonic',lambda:10)
    budget=verification_budget(4,100,100,now=700)
    assert budget.spent()==(600,pytest.approx(2/3))
    # Two CU at four CU/hour binds before the one-hour wall-clock limit.
    assert budget.remaining()==pytest.approx(1020)
    low=verification_budget(4,50.5,100,now=700)
    assert not low.permits()


def test_state_comparison_rejects_discrete_rng_changes():
    with pytest.raises(AssertionError,match='Discrete state'):
        compare_states({'rng':torch.tensor([1,2])},{'rng':torch.tensor([1,3])})
