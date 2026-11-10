"""Bounded CUDA comparison of the preserved failing tokenizer batch.

No optimizer is constructed or stepped. The caller owns allocation authorization,
the deadline watchdog, result export and runtime release.
"""
import gc
import json
import math
from pathlib import Path
import time

import torch

from .cohort import Cohort, read
from .data import file_hash
from .model import Config, WorldModel, image_loss
from scripts.diagnose_tokenizer_precision import recover_consumed_batch, gradient_report


def _require_budget(budget):
    if budget is None:
        raise ValueError('CUDA precision comparison requires an allocation budget')
    if not budget.permits(60):
        raise TimeoutError('Insufficient allocation budget for precision comparison')


def _backward_case(model, batches, precision, scale, loss_fn, budget, progress=None):
    """Device-native forward/backward, also usable with tiny CPU test models."""
    if precision not in {'fp32', 'fp16'} or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid precision or loss scale')
    _require_budget(budget)
    device = next(model.parameters()).device
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    result = dict(precision=precision, scale=scale, device=str(device), backend='device_native',
                  status='running', microbatches=[], optimizer_updates=0)
    started = time.monotonic()
    try:
        for index, batch in enumerate(batches):
            _require_budget(budget)
            x = batch.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=precision == 'fp16'):
                loss = loss_fn(model, x)
            item = dict(index=index, loss=float(loss.detach()) if torch.isfinite(loss) else None,
                        finite_loss=bool(torch.isfinite(loss)))
            if item['finite_loss']:
                (loss / len(batches) * scale).backward()
                item['gradients'] = gradient_report(model, scale)
                item['nonfinite_gradient_parameters'] = [g['name'] for g in item['gradients'] if g['nonfinite']]
            result['microbatches'].append(item)
            if progress:
                progress(result)
            if not item['finite_loss'] or item['nonfinite_gradient_parameters']:
                result['status'] = 'nonfinite'
                break
        else:
            result['status'] = 'finite'
            result['unscaled_gradient_norm'] = float(torch.linalg.vector_norm(torch.stack([
                parameter.grad.double().norm() / scale for parameter in model.parameters()
                if parameter.grad is not None])))
    except RuntimeError as error:
        result['status'] = 'runtime_error'
        result['error'] = str(error)
    except TimeoutError:
        result['status'] = 'budget_exhausted'
        raise
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        result['seconds'] = time.monotonic() - started
        result['model_state_unchanged'] = all(torch.equal(value, original[name])
            for name, value in model.state_dict().items())
        if progress:
            progress(result)
        if not result['model_state_unchanged']:
            raise AssertionError('Precision diagnostic changed model state')
    return result


def compare_recovered_tokenizer_batch(root, checkpoint, original_spec, output=None, device='cuda',
                                     scales=(65536., 1024.), threads=2, budget=None):
    """Return a comparison report; ``passed`` means only finite conservative backward.

The archived checkpoint/specification are read-only diagnostic inputs, not an
exact-resume migration or a production training authorization.
    """
    _require_budget(budget)
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Recovered-batch GPU comparison requires CUDA')
    if tuple(scales) != (65536., 1024.):
        raise ValueError('Verification requires original 65536 and conservative 1024 scales')
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    output = Path(output) if output is not None else None
    if output is not None and output.exists():
        raise FileExistsError('Preserve previous precision evidence; choose a new output file')
    root = Path(root).resolve()
    checkpoint = Path(checkpoint).resolve()
    original_spec = (root / original_spec).resolve()
    source_hash = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    spec = read(original_spec)
    if file_hash(original_spec) != state['contract']['spec_sha256']:
        raise ValueError('Original checkpoint specification hash changed')
    if state['config'] != spec['model']:
        raise ValueError('Checkpoint architecture differs from original specification')
    if state['contract']['microbatch'] != 4 or state['contract']['accumulation'] != 4:
        raise ValueError('Recovered verification requires the original 4 x 4 batch')
    data = Cohort(root, spec['data_binding'], 'train', state['config']['context'])
    if data.binding_sha256 != state['contract']['binding_sha256']:
        raise ValueError('Checkpoint data binding changed')
    windows = recover_consumed_batch(data, state, spec['optimization']['seed'],
                                    spec['data']['source_balanced_sampling_fraction'])
    batches = [torch.stack([data.window(*window) for window in windows[start:start + 4]])
               for start in range(0, len(windows), 4)]
    report = dict(schema_version=1, status='running', complete=False, optimizer_updates=0,
        checkpoint_sha256=source_hash, specification_sha256=file_hash(original_spec),
        binding_sha256=data.binding_sha256, checkpoint_implementation_sha256=state['contract']['implementation_sha256'],
        diagnostic_source_sha256=file_hash(__file__), model_source_sha256=file_hash(Path(__file__).with_name('model.py')),
        recovery_helper_sha256=file_hash(Path(__file__).resolve().parents[1] / 'scripts/diagnose_tokenizer_precision.py'),
        failed_update=state['step'] + 1, samples=[data.descriptor(*window) for window in windows],
        failed_samples_recovered_exactly=True, device=str(device), gpu=torch.cuda.get_device_name(device),
        torch_version=str(torch.__version__), original_torch_version=state['contract']['torch_version'],
        cudnn_version=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        quality_gates='unmeasured', interpretation='Recovered-batch numerical comparison only; no training or promotion.', cases=[])
    def save():
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + '.tmp')
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            temporary.replace(output)
    stage = next(stage for stage in spec['stages'] if stage['name'] == 'tokenizer')
    def loss_fn(model, x):
        prediction, _, vq = model.tokenizer(x)
        return 10 * image_loss(prediction, x, color_balance=stage['color_balance']) + vq
    torch.set_num_threads(threads)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        for precision, scale in [('fp32', 1.)] + [('fp16', scale) for scale in scales]:
            _require_budget(budget)
            model = WorldModel(Config(**state['config'])).to(device)
            model.load_state_dict(state['model'])
            model.requires_grad_(False)
            model.tokenizer.requires_grad_(True)
            model.eval()
            model.tokenizer.train()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            report['cases'].append({})
            def progress(case):
                report['cases'][-1] = case
                save()
            case = _backward_case(model, batches, precision, scale, loss_fn, budget, progress)
            report['cases'][-1] = case
            save()
            print(json.dumps({key: value for key, value in case.items() if key != 'microbatches'}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if case['status'] == 'runtime_error':
                break
        report['complete'] = len(report['cases']) == 3 and all(case['status'] in {'finite', 'nonfinite'} for case in report['cases'])
        report['conservative_fp16_finite'] = report['complete'] and report['cases'][2]['status'] == 'finite'
        report['fp32_finite'] = report['cases'][0]['status'] == 'finite'
        report['status'] = 'passed' if report['complete'] and report['conservative_fp16_finite'] and report['fp32_finite'] else 'failed'
    except BaseException:
        report['status'] = 'interrupted_or_failed'
        raise
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        report['checkpoint_file_unchanged'] = file_hash(checkpoint) == source_hash
        if not report['checkpoint_file_unchanged']:
            report['status'] = 'failed'
        save()
    if not report['checkpoint_file_unchanged']:
        raise AssertionError('Source checkpoint changed during diagnosis')
    return report
