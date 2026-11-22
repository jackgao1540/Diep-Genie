import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from genie_diep.baseline import implementation_hash
from genie_diep.cohort import Cohort
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.run_tokenizer_comparison import compare_exact, source_hash
from scripts.run_staged_tokenizer import (
    StagedTrainer, BOUND_SOURCES, BACKEND_POLICY, PILOT_LIMITS,
    allocation_budget, validate_output, run,
)
from test_baseline_pipeline import project


@pytest.fixture
def staged(project, monkeypatch):
    original_window = Cohort.window
    monkeypatch.setattr(Cohort, 'window', lambda self, *a: original_window(self, *a)[:, :, ::8, ::8])
    spec = json.loads((project / 'spec.json').read_text())
    spec['model'].update(size=16, context=16, downsample=4, vocab=16,
        tokenizer_arch='st_transformer', tokenizer_transformer=dict(
            encoder_width=8, encoder_depth=1, encoder_heads=2,
            decoder_width=8, decoder_depth=1, decoder_heads=2))
    spec['stages'][0].update(max_updates=20000, microbatch_sequences=4, gradient_accumulation=4)
    train = Cohort(project, 'binding.json', 'train', 16)
    val = Cohort(project, 'binding.json', 'val', 16)
    (project / 'evaluation.json').write_text(json.dumps(dict(binding_sha256=train.binding_sha256,
        validation=[val.descriptor(0, 0)], training_calibration=[train.descriptor(0, 0)])))
    entries = {}
    for phase in ('joint', 'decoder_only'):
        current = copy.deepcopy(spec)
        if phase == 'decoder_only':
            current['model']['tokenizer_transformer'].update(decoder_width=12, decoder_heads=3)
        model = WorldModel(Config(**current['model']))
        current['parameter_counts'] = {n: sum(p.numel() for p in getattr(model.tokenizer, n).parameters())
                                       for n in ('encoder','decoder','vq')}
        path = project / (phase + '.json'); path.write_text(json.dumps(current))
        entries[phase] = dict(spec=path.name, spec_sha256=file_hash(path))
    plan = dict(schema_version=1, phases=['joint','decoder_only'], stages=entries,
        implementation_sha256=implementation_hash(), runner_sha256=source_hash('scripts/run_staged_tokenizer.py'),
        backend_policy=BACKEND_POLICY, pilot_limits=PILOT_LIMITS, full_updates_per_phase=5000,
        timing_updates=5, microbatch_candidates=[4,2,1], sequences_per_update=16,
        coverage_after_update=999, coverage_seed=20261110, maximum_fp16_gradient_norm_relative_error=.05,
        source_sha256={name:source_hash(name) for name in BOUND_SOURCES})
    for name in BOUND_SOURCES:
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
    (project / 'plan.json').write_text(json.dumps(plan))
    return project


def test_finite_stages_exact_handoff_and_resume(staged):
    a = StagedTrainer(staged, 'plan.json', 'joint', microbatch=4, local_check=True)
    assert a.microbatch * a.accumulation * 16 == 256
    assert torch.isfinite(torch.tensor(a.update()))
    a.save(staged / 'joint.pt'); reference = a.freeze_reference()
    b = StagedTrainer(staged, 'plan.json', 'decoder_only', microbatch=4,
                      handoff=staged / 'joint.pt', local_check=True)
    compare_exact(b.freeze_reference(), reference)
    assert set(id(p) for p in b.parameters) == set(id(p) for p in b.module.decoder.parameters())
    assert b.scheduler.optimizer is b.optimizer
    assert b.stage_spec['max_updates'] == 20000 and b.scheduler.lr_lambdas[0](5000) > .8
    assert torch.isfinite(torch.tensor(b.update()))
    b.verify_inactive(); compare_exact(b.freeze_reference(), reference)
    b.save(staged / 'decoder.pt'); expected = b.update()
    resumed = StagedTrainer(staged, 'plan.json', 'decoder_only', microbatch=4,
        handoff=staged / 'joint.pt', resume=staged / 'decoder.pt', local_check=True)
    assert resumed.update() == expected
    for key in ('model','optimizer','scheduler','scaler','sampler','rng','step'):
        compare_exact(resumed._update_boundary()[key], b._update_boundary()[key])


def test_partial_optimizer_failure_rolls_back_staged_state(staged, monkeypatch):
    trainer = StagedTrainer(staged, 'plan.json', 'joint', local_check=True)
    trainer.update(); before = trainer._update_boundary()
    original = trainer.optimizer.step
    def fail(*a, **kw):
        original(*a, **kw)
        raise RuntimeError('injected partial step')
    monkeypatch.setattr(trainer.optimizer, 'step', fail)
    with pytest.raises(RuntimeError, match='partial step'): trainer.update()
    compare_exact(trainer._update_boundary(), before)
    assert trainer.last_failure['restored_pre_update_boundary']


def test_local_orchestrator_and_finished_resume(staged):
    args = SimpleNamespace(root=str(staged), plan='plan.json', out=str(staged / 'runs' / 'check'),
        mode='pilot', device='cpu', local_check=True, authorized=False, rate=0, balance=None,
        allocation_start=None, prior_seconds=0, prior_units=0, threads=2, resume=False,
        full_budget=None, private_output_root=None)
    report = run(args)
    assert report['status'] == 'local_check_complete'
    assert report['phases']['decoder_only']['handoff_ids_embeddings_exact']
    args.resume = True
    assert run(args)['phases']['joint']['latest_step'] == 1
    reference = staged / 'runs/check/joint/handoff-reference.pt'
    reference.write_bytes(b'changed')
    with pytest.raises(ValueError, match='reference changed'): run(args)


def test_full_requires_independent_budget_before_cuda_allocation():
    args = SimpleNamespace(mode='full', full_budget=None)
    with pytest.raises(ValueError, match='independent budget'): allocation_budget(args)


def test_output_root_allows_only_explicit_private_cuda_directory(tmp_path):
    root = tmp_path / 'source'; private = tmp_path / 'private'
    assert validate_output(root, root / 'runs/check', 'cpu') == root / 'runs/check'
    assert validate_output(root, private / 'run/experiment', 'cuda', private) == private / 'run/experiment'
    with pytest.raises(ValueError): validate_output(root, private / 'run', 'cpu', private)
    with pytest.raises(ValueError): validate_output(root, private.parent / 'public', 'cuda', private)


def test_real_plan_is_content_bound():
    from scripts.run_staged_tokenizer import validate_plan
    validate_plan(Path.cwd(), 'configs/staged-tokenizer-run-v1.json')


def test_full_gate_binds_pilot_conditions_debits_and_forecast(tmp_path, monkeypatch):
    from scripts.run_staged_tokenizer import full_pilot_gate
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'pilot GPU')
    plan_path = tmp_path / 'plan.json'; plan_path.write_text('{}')
    entry = dict(complete=True, pilot_passed=True, precision=dict(passed=True), replay=dict(passed=True),
        microbatch=2, accumulation=8, timing=dict(unique_updates=5, seconds=[10.]*5, mean_seconds=10.),
        checkpoint_seconds=[2.])
    pilot = dict(status='pilot_complete', identity=dict(local_check=False, mode='pilot',
        plan_sha256=file_hash(plan_path), rate=1.15), gpu='pilot GPU',
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        spent_seconds=900., spent_units=.3, phases={p:copy.deepcopy(entry) for p in ('joint','decoder_only')})
    path = tmp_path / 'pilot.json'; path.write_text(json.dumps(pilot))
    receipt = dict(pilot=dict(report_path=str(path), sha256=file_hash(path)),
                   minimum_prior_seconds=900., minimum_prior_units=.3)
    receipt_path = tmp_path / 'budget.json'; receipt_path.write_text(json.dumps(receipt))
    args = SimpleNamespace(full_budget=str(receipt_path), plan='plan.json', rate=1.15,
                           prior_seconds=900., prior_units=.3)
    plan = dict(microbatch_candidates=[4,2,1])
    budget = SimpleNamespace(permits=lambda seconds: seconds < 200000)
    result = full_pilot_gate(tmp_path, args, plan, budget)
    assert result['microbatches'] == dict(joint=2, decoder_only=2)
    assert result['estimated_remaining_seconds_with_margin'] > 130000
    with pytest.raises(ValueError, match='cannot fit'):
        full_pilot_gate(tmp_path, args, plan, SimpleNamespace(permits=lambda seconds: False))
    args.prior_units = .2
    with pytest.raises(ValueError, match='prior allocation'):
        full_pilot_gate(tmp_path, args, plan, budget)
    args.prior_units = .3; args.rate = 2.
    with pytest.raises(ValueError, match='rate/mode'):
        full_pilot_gate(tmp_path, args, plan, budget)
