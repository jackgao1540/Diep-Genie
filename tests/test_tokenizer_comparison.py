import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from genie_diep.baseline import implementation_hash, rng_state, restore_rng
from genie_diep.cohort import read
from genie_diep.data import file_hash
from scripts.run_tokenizer_comparison import ComparisonTrainer, allocation_budget, compare_exact, paired_update, run, source_hash, validate_experiment
from test_baseline_pipeline import project, write


@pytest.fixture
def comparison(project):
    code_root=Path(__file__).resolve().parents[1]
    for name in ('genie_diep/model.py','genie_diep/detail_loss.py','scripts/run_tokenizer_comparison.py'):
        target=project/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((code_root/name).read_bytes())
    original=read(project/'spec.json')
    next(s for s in original['stages'] if s['name']=='tokenizer')['max_updates']=20000
    write(project/'reference.json',original)
    arms={}
    for arm,objective in [('baseline',dict(name='baseline')),('detail',dict(name='baseline_plus_detail',weight=1.,loss_spec={}))]:
        spec=copy.deepcopy(original);spec['comparison_objective']=objective
        filename=f'{arm}.json';write(project/filename,spec)
        arms[arm]=dict(spec=filename,spec_sha256=file_hash(project/filename),objective=objective)
    experiment=dict(arms=arms,reference_spec='reference.json',reference_spec_sha256=file_hash(project/'reference.json'),
        source_hashes=dict(implementation_sha256=implementation_hash(),
        model_sha256=source_hash(project/'genie_diep/model.py'),detail_loss_sha256=source_hash(project/'genie_diep/detail_loss.py'),
        runner_sha256=source_hash(project/'scripts/run_tokenizer_comparison.py')),
        limits=dict(optimizer_calls_per_arm=1000,unique_updates_per_arm=999,maximum_compute_units=2,maximum_allocated_hours=1,minimum_remaining_units=50),
        save_steps=[250,500,750,999],schedule_max_updates=20000)
    write(project/'experiment.json',experiment)
    return project


def make_arms(root):
    return {arm:ComparisonTrainer(root,f'{arm}.json','experiment.json',arm,preflight=True) for arm in ('baseline','detail')}


def test_fresh_matching_initialization_and_isolated_rng(comparison,monkeypatch):
    trainers=make_arms(comparison)
    compare_exact(trainers['baseline'].model.state_dict(),trainers['detail'].model.state_dict())
    first=copy.deepcopy(trainers['baseline'].arm_rng)
    # Simulate stochastic objectives in both arms, despite the current loss being deterministic.
    draws={arm:[] for arm in trainers}
    for arm,trainer in trainers.items():
        original=trainer.loss
        def stochastic(x,arm=arm,original=original):
            draws[arm].append(torch.rand(()).item())
            return original(x)
        monkeypatch.setattr(trainer,'loss',stochastic)
    paired_update(trainers,None)
    assert draws['baseline']==draws['detail']
    compare_exact(trainers['baseline'].sampler.state_dict(),trainers['detail'].sampler.state_dict())
    assert trainers['baseline'].step==trainers['detail'].step==1
    assert not torch.equal(first['torch'],trainers['baseline'].arm_rng['torch'])


def test_midpair_budget_preserves_partial_and_restores_common_boundary(comparison):
    trainers=make_arms(comparison);before={a:t.snapshot() for a,t in trainers.items()};partials=[]
    class Budget:
        calls=0
        def permits(self,*args):
            self.calls+=1
            return self.calls<=2  # baseline's two microbatches pass; detail stops.
    with pytest.raises(TimeoutError,match='paired update'):
        paired_update(trainers,Budget(),on_partial=lambda arm,t:partials.append((arm,t.step)))
    assert partials==[('baseline',1)]
    for arm,trainer in trainers.items():compare_exact(trainer.snapshot(),before[arm])


def test_nonfinite_second_arm_preserves_failure_and_rolls_back_first(comparison,monkeypatch):
    trainers=make_arms(comparison);before={a:t.snapshot() for a,t in trainers.items()}
    monkeypatch.setattr(trainers['detail'],'loss',lambda x:torch.tensor(float('nan')))
    with pytest.raises(FloatingPointError):paired_update(trainers,None)
    for arm,trainer in trainers.items():compare_exact(trainer.snapshot(),before[arm])
    assert trainers['detail'].last_failure['restored_pre_update_boundary'] is True


def test_detail_exact_resume_rejects_changed_objective(comparison):
    trainer=make_arms(comparison)['detail'];trainer.update();trainer.save(comparison/'saved.pt')
    expected_loss=trainer.update();expected=trainer.snapshot()
    restored=ComparisonTrainer(comparison,'detail.json','experiment.json','detail',preflight=True,resume=comparison/'saved.pt')
    assert restored.update()==expected_loss;compare_exact(restored.snapshot(),expected)
    experiment=read(comparison/'experiment.json');experiment['arms']['detail']['objective']['weight']=2.
    spec=read(comparison/'detail.json');spec['comparison_objective']['weight']=2.;write(comparison/'detail.json',spec)
    experiment['arms']['detail']['spec_sha256']=file_hash(comparison/'detail.json');write(comparison/'experiment.json',experiment)
    with pytest.raises(ValueError,match='resume contract'):
        ComparisonTrainer(comparison,'detail.json','experiment.json','detail',preflight=True,resume=comparison/'saved.pt')


def test_local_preflight_completes_both_exact_measurements(comparison,monkeypatch):
    # Exercise the saved-boundary evaluation path with this one-window fixture.
    def local_boundaries(*args):
        experiment=validate_experiment(*args);experiment['save_steps']=[2];return experiment
    monkeypatch.setattr('scripts.run_tokenizer_comparison.validate_experiment',local_boundaries)
    args=SimpleNamespace(root=str(comparison),out=str(comparison/'run'),experiment='experiment.json',baseline_spec='baseline.json',
                         detail_spec='detail.json',device='cpu',preflight=True,authorized=False,updates=2,threads=2)
    result=run(args)
    assert result['status']=='completed' and result['latest_common_step']==2
    assert result['optimizer_calls']==dict(baseline=3,detail=3)
    assert result['replay_optimizer_calls']==dict(baseline=1,detail=1)
    assert result['measurement']['both_arms_passed'] is True
    assert len(result['evaluations'])==2 and all(item['report']['complete'] for item in result['evaluations'])
    assert all(item['unchanged'] for item in result['frozen_modules'].values())
    for arm in ('baseline','detail'):
        state=torch.load(comparison/'run'/arm/'last-common.pt',weights_only=True)
        assert state['step']==2 and state['contract']['preflight'] is True
        assert state['contract']['comparison']['arm']==arm


def test_spec_difference_cannot_change_sampling(comparison):
    spec=read(comparison/'detail.json');spec['data']['source_balanced_sampling_fraction']=.9;write(comparison/'detail.json',spec)
    experiment=read(comparison/'experiment.json');experiment['arms']['detail']['spec_sha256']=file_hash(comparison/'detail.json')
    write(comparison/'experiment.json',experiment)
    with pytest.raises(ValueError,match='outside objective'):
        validate_experiment(comparison,'experiment.json','baseline.json','detail.json')


def test_source_edit_invalidates_bound_experiment(comparison):
    with (comparison/'genie_diep/detail_loss.py').open('a') as stream:stream.write('\n# changed\n')
    with pytest.raises(ValueError,match='source changed'):
        ComparisonTrainer(comparison,'detail.json','experiment.json','detail',preflight=True)


def test_both_arms_cannot_drift_from_bound_original_baseline(comparison):
    experiment=read(comparison/'experiment.json')
    for arm in ('baseline','detail'):
        spec=read(comparison/f'{arm}.json');spec['optimization']['seed']=123;write(comparison/f'{arm}.json',spec)
        experiment['arms'][arm]['spec_sha256']=file_hash(comparison/f'{arm}.json')
    write(comparison/'experiment.json',experiment)
    with pytest.raises(ValueError,match='original specification'):
        validate_experiment(comparison,'experiment.json','baseline.json','detail.json')


def test_shared_budget_and_explicit_authorization(monkeypatch):
    monkeypatch.setattr('scripts.run_tokenizer_comparison.time.time',lambda:700)
    monkeypatch.setattr('genie_diep.baseline.time.monotonic',lambda:10)
    args=SimpleNamespace(preflight=False,authorized=True,device='cuda',rate=4.,balance=100.,allocation_start=100.)
    budget=allocation_budget(args)
    assert budget.remaining()==pytest.approx(1020)
    args.authorized=False
    with pytest.raises(ValueError,match='authorization'):allocation_budget(args)


def test_cpu_preflight_cannot_run_extended_training(comparison):
    args=SimpleNamespace(root=str(comparison),out=str(comparison/'run'),experiment='experiment.json',baseline_spec='baseline.json',
                         detail_spec='detail.json',device='cpu',preflight=True,authorized=False,updates=3,threads=2)
    with pytest.raises(ValueError,match='limit'):run(args)
