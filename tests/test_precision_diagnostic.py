import copy
from argparse import Namespace

import pytest
import torch

from genie_diep.cohort import MixedSampler
from scripts.diagnose_tokenizer_precision import backward_case, gradient_report, recover_consumed_batch, run


class TinyCohort:
    split = 'train'
    sources = ['a', 'b']
    source_cumulative = {'a': [10], 'b': [20]}
    cumulative = [10, 30]
    by_source = {'a': [0], 'b': [1]}


def test_failure_batch_replay_requires_complete_matching_sampler_boundary():
    data = TinyCohort()
    sampler = MixedSampler(data, 42, .5)
    for _ in range(1 + 3 * 4):
        sampler.draw()
    expected = [sampler.draw() for _ in range(4)]
    state = dict(contract=dict(stage='tokenizer', microbatch=2, accumulation=2), step=3,
                 sampler=sampler.state_dict())
    assert recover_consumed_batch(data, state, 42, .5) == expected
    changed = copy.deepcopy(state)
    changed['sampler']['draws'] -= 1
    with pytest.raises(ValueError, match='fully consumed'):
        recover_consumed_batch(data, changed, 42, .5)
    with pytest.raises(ValueError, match='replay'):
        recover_consumed_batch(data, state, 7, .5)


@pytest.mark.parametrize('backend', ['native', 'mkldnn-forward'])
def test_gradient_diagnostic_preserves_parameters_and_exposes_overflow(backend):
    torch.set_num_threads(2)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    batches = [torch.ones(2, 2)]
    before = model.weight.detach().clone()
    original_backend = torch.backends.mkldnn.enabled
    loss = lambda module, x: module(x).square().mean()
    full = backward_case(model, batches, 'fp32', 1, loss, fp16_backend=backend)
    overflow = backward_case(model, batches, 'fp16', 65536, loss, fp16_backend=backend)
    conservative = backward_case(model, batches, 'fp16', 128, loss, fp16_backend=backend)
    assert full['status'] == conservative['status'] == 'finite'
    assert full['unscaled_gradient_norm'] == pytest.approx(conservative['unscaled_gradient_norm'])
    assert overflow['status'] == 'nonfinite'
    assert overflow['microbatches'][0]['nonfinite_gradient_parameters'] == ['weight']
    assert torch.equal(model.weight, before)
    assert torch.backends.mkldnn.enabled == original_backend


def test_runtime_error_restores_backend_and_is_reported():
    model = torch.nn.Linear(2, 1)
    backend = torch.backends.mkldnn.enabled
    def failure(module, x):
        raise RuntimeError('unsupported kernel')
    result = backward_case(model, [torch.ones(1, 2)], 'fp16', 128, failure)
    assert result['status'] == 'runtime_error'
    assert result['error'] == 'unsupported kernel'
    assert result['model_state_unchanged']
    assert torch.backends.mkldnn.enabled == backend


def test_nonfinite_gradient_statistics_count_nan_and_infinity():
    model = torch.nn.Linear(3, 1, bias=False)
    model.weight.grad = torch.tensor([[float('nan'), float('inf'), 8.]])
    record, = gradient_report(model, 4)
    assert record['nonfinite'] == 2 and record['nan'] == record['infinity'] == 1
    assert record['max_finite_unscaled'] == 2


def test_previous_diagnostic_evidence_cannot_be_overwritten(tmp_path):
    existing = tmp_path / 'evidence'
    existing.mkdir()
    report = existing / 'report.json'
    report.write_text('preserved evidence')
    with pytest.raises(FileExistsError, match='immutable'):
        run(Namespace(threads=2, root=str(tmp_path), checkpoint=str(tmp_path / 'absent.pt'), out=str(existing)))
    assert report.read_text() == 'preserved evidence'


def test_state_mutation_during_diagnosis_is_rejected():
    model = torch.nn.Linear(2, 1)
    def mutating_loss(module, x):
        with torch.no_grad():
            module.weight.add_(1)
        return module(x).square().mean()
    with pytest.raises(AssertionError, match='changed model state'):
        backward_case(model, [torch.ones(1, 2)], 'fp32', 1, mutating_loss)
