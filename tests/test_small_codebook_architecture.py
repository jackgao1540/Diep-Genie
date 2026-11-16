import copy
import json
from types import SimpleNamespace

import pytest
import torch

from genie_diep.baseline import Budget, implementation_hash, rng_state
from genie_diep.cohort import Cohort, MixedSampler
from genie_diep.coverage_sampler import CoverageSampler
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.run_small_codebook_architecture import ArchitectureTrainer, run, BACKEND_POLICY, measure_training_updates
from scripts.run_tokenizer_comparison import source_hash, compare_exact
from test_baseline_pipeline import project


@pytest.fixture
def experiment(project):
    root = project
    spec = json.loads((root/'spec.json').read_text())
    spec['model'].update(context=16, downsample=4, vocab=512)
    data = Cohort(root, 'binding.json', 'train', 16)
    val = Cohort(root, 'binding.json', 'val', 16)
    (root/'evaluation.json').write_text(json.dumps(dict(binding_sha256=file_hash(root/'binding.json'),
        validation=[val.descriptor(0, 0)])))
    torch.manual_seed(42)
    reference = WorldModel(Config(**spec['model']))
    sampler = MixedSampler(data, 42, .5)
    with torch.no_grad():
        reference.tokenizer.vq.initialize(reference.tokenizer.features(sampler.batch(1)))
    torch.save(dict(step=0, model=reference.state_dict(), optimizer=dict(state={}), sampler=sampler.state_dict(),
        contract=dict(binding_sha256=data.binding_sha256, stage='tokenizer', preflight=False), rng=rng_state()), root/'fresh.pt')
    arms = {}
    for arm in ('cnn', 'transformer'):
        s = copy.deepcopy(spec); s['model']['vocab'] = 32
        s['stages'][0].update(microbatch_sequences=1, gradient_accumulation=1)
        if arm == 'transformer':
            s['model'].update(tokenizer_arch='st_transformer', tokenizer_transformer=dict(
                encoder_width=8, encoder_depth=1, encoder_heads=2,
                decoder_width=8, decoder_depth=1, decoder_heads=2))
        path = root/f'{arm}.json'; path.write_text(json.dumps(s))
        arms[arm] = dict(spec=path.name, spec_sha256=file_hash(path))
    plan = dict(arms=arms, backend_policy=BACKEND_POLICY, initial_checkpoint=dict(path='fresh.pt', sha256=file_hash(root/'fresh.pt')),
        implementation_sha256=implementation_hash(), runner_sha256=source_hash('scripts/run_small_codebook_architecture.py'),
        coverage_after_update=1, coverage_seed=20261110)
    (root/'plan.json').write_text(json.dumps(plan))
    return root, reference


def test_cnn_starts_from_exact_untrained_encoder_decoder_and_book_prefix(experiment):
    root, original = experiment
    trainer = ArchitectureTrainer(root, 'plan.json', 'cnn', preflight=True)
    for name, tensor in trainer.model.state_dict().items():
        if name.startswith(('tokenizer.encoder.', 'tokenizer.decoder.')):
            torch.testing.assert_close(tensor, original.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(trainer.model.tokenizer.vq.embedding.weight,
        original.tokenizer.vq.embedding.weight[:32], rtol=0, atol=0)
    assert trainer.step == 0 and not trainer.optimizer.state and trainer.sampler.draws == 1
    assert trainer.contract['backend_policy']==BACKEND_POLICY
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()


def test_both_arms_have_identical_mixed_coverage_and_fallback_descriptors(experiment):
    root, _ = experiment
    a = ArchitectureTrainer(root, 'plan.json', 'cnn', preflight=True)
    b = ArchitectureTrainer(root, 'plan.json', 'transformer', preflight=True)
    assert [a.sampler.draw() for _ in range(1)] == [b.sampler.draw() for _ in range(1)]
    a._attach_coverage(); b._attach_coverage()
    assert a.sampler.plan_sha256 == b.sampler.plan_sha256
    compare_exact(a.sampler.state_dict(), b.sampler.state_dict())
    for _ in range(len(a.sampler._plan)+9):
        assert a.sampler.draw() == b.sampler.draw()
    assert a.sampler.phase == b.sampler.phase == 'mixed'
    compare_exact(a.sampler.state_dict(), b.sampler.state_dict())


@pytest.mark.parametrize('arm', ['cnn', 'transformer'])
def test_finite_update_exact_resume_across_coverage_switch_and_frozen_modules(experiment, arm):
    root, _ = experiment
    a = ArchitectureTrainer(root, 'plan.json', arm, preflight=True)
    assert torch.isfinite(torch.tensor(a.update()))
    a.save(root/'mixed.pt')
    expected = a.update(); assert isinstance(a.sampler, CoverageSampler)
    a.save(root/'coverage.pt')
    b = ArchitectureTrainer(root, 'plan.json', arm, preflight=True, resume=root/'mixed.pt')
    assert b.update() == expected
    compare_exact(a.model.state_dict(), b.model.state_dict())
    compare_exact(a.sampler.state_dict(), b.sampler.state_dict())
    c = ArchitectureTrainer(root, 'plan.json', arm, preflight=True, resume=root/'coverage.pt')
    expected = a.update(); assert c.update() == expected
    compare_exact(a.model.state_dict(), c.model.state_dict()); a.verify_inactive(); c.verify_inactive()


@pytest.mark.parametrize('arm', ['cnn', 'transformer'])
def test_partial_optimizer_failure_restores_model_optimizer_and_coverage_cursor(experiment, arm, monkeypatch):
    root, _ = experiment
    trainer = ArchitectureTrainer(root, 'plan.json', arm, preflight=True)
    trainer.update(); trainer._attach_coverage(); before = trainer._update_boundary()
    original = trainer.optimizer.step
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected partial optimizer step')
    monkeypatch.setattr(trainer.optimizer, 'step', fail)
    with pytest.raises(RuntimeError, match='injected'):
        trainer.update()
    after = trainer._update_boundary()
    for name in ('model', 'optimizer', 'scheduler', 'scaler', 'sampler', 'rng', 'step'):
        compare_exact(after[name], before[name])
    assert trainer.last_failure['restored_pre_update_boundary']
    assert trainer.last_failure['sampled_windows']
    trainer.save(root/'safe-after-failure.pt')


def test_orchestrator_resumes_interrupted_second_arm_without_retraining_first(experiment, monkeypatch):
    root, _ = experiment
    plan = json.loads((root/'plan.json').read_text())
    plan.update(limits=dict(maximum_allocated_hours=4, maximum_compute_units=4, minimum_remaining_units=50),
        unique_updates_per_arm=5000, save_steps=[1000,2000,3000,4000,5000], progress_validation_windows=1)
    (root/'plan.json').write_text(json.dumps(plan))
    args = SimpleNamespace(root=str(root), plan='plan.json', out=str(root/'run'), device='cpu', preflight=True,
        authorized=False, rate=0, balance=None, allocation_start=None, threads=2, resume=False)
    original = ArchitectureTrainer.update
    def interrupted(self, budget=None):
        if self.arm == 'transformer' and self.step == 1:
            raise KeyboardInterrupt('simulated disconnect before second-arm training')
        return original(self, budget)
    monkeypatch.setattr(ArchitectureTrainer, 'update', interrupted)
    with pytest.raises(KeyboardInterrupt): run(args)
    saved = json.loads((root/'run/experiment-report.json').read_text())
    assert saved['arms']['cnn']['completed'] and saved['arms']['cnn']['latest_step']==2
    first_sha = saved['arms']['cnn']['checkpoint']['sha256']
    assert saved['arms']['transformer']['pilot_passed'] and saved['arms']['transformer']['latest_step']==1
    monkeypatch.setattr(ArchitectureTrainer, 'update', original); args.resume=True
    run(args)
    final = json.loads((root/'run/experiment-report.json').read_text())
    assert final['status']=='completed' and final['arms']['cnn']['checkpoint']['sha256']==first_sha
    assert final['arms']['transformer']['latest_step']==2 and final['arms']['transformer']['completed']


@pytest.mark.parametrize('arm', ['cnn', 'transformer'])
def test_steady_timing_counts_five_updates_and_continues_saved_state(experiment, arm):
    root, _ = experiment
    trainer = ArchitectureTrainer(root, 'plan.json', arm, preflight=True)
    trainer.update()  # Exact-replay boundary; the measured updates are steps2..6.
    budget = Budget(dict(max_allocated_hours=4, max_compute_units=4), reserve_seconds=0)
    timing = measure_training_updates(trainer, budget, 5)
    assert timing['first_step']==2 and timing['last_step']==6 and timing['unique_updates']==5
    assert len(timing['seconds'])==5 and all(t>0 for t in timing['seconds'])
    assert timing['mean_seconds']==sum(timing['seconds'])/5 and trainer.step==6
    trainer.save(root/'timed.pt')
    resumed = ArchitectureTrainer(root, 'plan.json', arm, preflight=True, resume=root/'timed.pt')
    assert resumed.step==6 and resumed.update()==trainer.update()
    assert resumed.step==trainer.step==7
    compare_exact(resumed.model.state_dict(), trainer.model.state_dict())
    compare_exact(resumed.sampler.state_dict(), trainer.sampler.state_dict())


def test_steady_timing_budget_stop_preserves_completed_update_boundary(experiment):
    root, _ = experiment
    trainer = ArchitectureTrainer(root, 'plan.json', 'cnn', preflight=True)
    trainer.update()
    class StopAfterTwo:
        def permits(self, seconds=0): return trainer.step<3
    with pytest.raises(TimeoutError, match='steady timing'):
        measure_training_updates(trainer, StopAfterTwo(), 5)
    assert trainer.step==3
    trainer.save(root/'timing-stopped.pt')
    resumed=ArchitectureTrainer(root, 'plan.json', 'cnn', preflight=True, resume=root/'timing-stopped.pt')
    assert resumed.step==3
    compare_exact(resumed.model.state_dict(), trainer.model.state_dict())
    compare_exact(resumed.sampler.state_dict(), trainer.sampler.state_dict())
