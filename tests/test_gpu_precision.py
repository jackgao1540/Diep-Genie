import pytest
import torch

from genie_diep.gpu_precision import _backward_case, compare_recovered_tokenizer_batch


class Budget:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls = []

    def permits(self, seconds):
        self.calls.append(seconds)
        return self.allowed


def test_gpu_compare_refuses_unbounded_or_cpu_execution(tmp_path):
    with pytest.raises(ValueError, match='budget'):
        compare_recovered_tokenizer_batch(tmp_path, 'missing.pt', 'missing.json')
    with pytest.raises(ValueError, match='requires CUDA'):
        compare_recovered_tokenizer_batch(tmp_path, 'missing.pt', 'missing.json', device='cpu', budget=Budget())
    with pytest.raises(TimeoutError, match='budget'):
        compare_recovered_tokenizer_batch(tmp_path, 'missing.pt', 'missing.json', budget=Budget(False))


def test_device_native_comparison_exposes_overflow_without_updates_or_backend_changes():
    torch.set_num_threads(2)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    before = model.weight.detach().clone()
    backend = torch.backends.mkldnn.enabled
    budget = Budget()
    batches = [torch.ones(1, 2)] * 4
    loss = lambda module, x: module(x).square().mean()
    full = _backward_case(model, batches, 'fp32', 1, loss, budget)
    large = _backward_case(model, batches, 'fp16', 65536, loss, budget)
    small = _backward_case(model, batches, 'fp16', 1024, loss, budget)
    assert full['status'] == small['status'] == 'finite'
    assert large['status'] == 'nonfinite'
    assert large['microbatches'][0]['nonfinite_gradient_parameters'] == ['weight']
    assert small['unscaled_gradient_norm'] == pytest.approx(full['unscaled_gradient_norm'])
    assert full['model_state_unchanged'] and large['model_state_unchanged'] and small['model_state_unchanged']
    assert torch.equal(model.weight, before) and torch.backends.mkldnn.enabled == backend
    assert set(budget.calls) == {60}


def test_budget_stop_before_second_microbatch_preserves_weights_and_reports_boundary():
    model = torch.nn.Linear(2, 1)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    budget = Budget()
    reports = []
    def progress(result):
        reports.append(result.copy())
        budget.allowed = False
    with pytest.raises(TimeoutError):
        _backward_case(model, [torch.ones(1, 2)] * 4, 'fp32', 1,
                       lambda module, x: module(x).square().mean(), budget, progress)
    assert reports[-1]['status'] == 'budget_exhausted'
    assert len(reports[-1]['microbatches']) == 1
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())


def test_nonfinite_loss_and_runtime_error_cannot_pass():
    model = torch.nn.Linear(2, 1)
    batch = [torch.ones(1, 2)]
    bad = _backward_case(model, batch, 'fp32', 1, lambda module, x: module(x).mean() * float('nan'), Budget())
    assert bad['status'] == 'nonfinite' and bad['microbatches'][0]['loss'] is None
    def fail(module, x):
        raise RuntimeError('kernel failure')
    error = _backward_case(model, batch, 'fp32', 1, fail, Budget())
    assert error['status'] == 'runtime_error' and error['model_state_unchanged']
