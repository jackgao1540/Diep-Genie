"""Replay a stopped tokenizer batch on CPU without optimizer updates.

Native CPU FP16 backward bypasses oneDNN's unsupported backward path. This is a
numerical diagnostic, never a CUDA reproduction or a model-quality result.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch

from genie_diep.cohort import Cohort, MixedSampler, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel, image_loss


def recover_consumed_batch(data, state, seed, source_fraction):
    """Recover only the legacy, fully consumed failed-update sampling boundary."""
    contract = state['contract']
    if contract['stage'] != 'tokenizer':
        raise ValueError('Only tokenizer failure diagnosis is supported')
    count = contract['microbatch'] * contract['accumulation']
    before = 1 + state['step'] * count  # one codebook-initialization sample
    if state['sampler']['draws'] != before + count:
        raise ValueError('Not a legacy fully consumed failure boundary')
    sampler = MixedSampler(data, seed, source_fraction)
    for _ in range(before):
        sampler.draw()
    windows = [sampler.draw() for _ in range(count)]
    if (sampler.draws != state['sampler']['draws'] or
            sampler.fraction != state['sampler']['fraction'] or
            not torch.equal(sampler.rng.get_state(), state['sampler']['rng'].cpu())):
        raise ValueError('Sampler replay does not match checkpoint')
    return windows


def gradient_report(module, scale):
    records = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        finite = torch.isfinite(gradient)
        values = gradient[finite]
        records.append(dict(name=name, elements=gradient.numel(),
            nonfinite=int((~finite).sum()), nan=int(torch.isnan(gradient).sum()),
            infinity=int(torch.isinf(gradient).sum()),
            max_finite_unscaled=float(values.abs().max() / scale) if values.numel() else None))
    return records


def backward_case(model, batches, precision, scale, loss_fn, progress=None, fp16_backend='native'):
    if precision not in {'fp32', 'fp16'} or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid precision or loss scale')
    if next(model.parameters()).device.type != 'cpu':
        raise ValueError('This diagnostic is CPU-only')
    if fp16_backend not in {'native', 'mkldnn-forward'}:
        raise ValueError('Unsupported CPU FP16 backend')
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    result = dict(precision=precision, scale=scale,
                  backend=('native_cpu' if fp16_backend == 'native' else 'mkldnn_forward_native_backward') if precision == 'fp16' else 'default_cpu',
                  status='running', microbatches=[])
    started = time.monotonic()
    previous_backend = torch.backends.mkldnn.enabled
    try:
        for index, x in enumerate(batches):
            if precision == 'fp16':
                torch.backends.mkldnn.enabled = fp16_backend == 'mkldnn-forward'
            with torch.autocast('cpu', dtype=torch.float16, enabled=precision == 'fp16'):
                loss = loss_fn(model, x)
            item = dict(index=index, loss=float(loss.detach()) if torch.isfinite(loss) else None,
                        finite_loss=bool(torch.isfinite(loss)))
            if item['finite_loss']:
                if precision == 'fp16':
                    torch.backends.mkldnn.enabled = False
                (loss / len(batches) * scale).backward()
                item['gradients'] = gradient_report(model, scale)
                item['nonfinite_gradient_parameters'] = [r['name'] for r in item['gradients'] if r['nonfinite']]
            result['microbatches'].append(item)
            if progress:
                progress(result)
            if not item['finite_loss'] or item['nonfinite_gradient_parameters']:
                result['status'] = 'nonfinite'
                break
        else:
            result['status'] = 'finite'
            result['unscaled_gradient_norm'] = float(torch.linalg.vector_norm(torch.stack([
                p.grad.double().norm() / scale for p in model.parameters() if p.grad is not None])))
    except RuntimeError as error:
        result['status'] = 'runtime_error'
        result['error'] = str(error)
    finally:
        torch.backends.mkldnn.enabled = previous_backend
        result['seconds'] = time.monotonic() - started
        result['model_state_unchanged'] = all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
        if not result['model_state_unchanged']:
            raise AssertionError('Diagnostic changed model state')
    return result


def run(args):
    torch.set_num_threads(args.threads)
    root = Path(args.root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Use a new diagnostic directory; previous evidence is immutable')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    spec = read(root / args.spec)
    if file_hash(root / args.spec) != state['contract']['spec_sha256']:
        raise ValueError('Use the original checkpoint specification')
    data = Cohort(root, spec['data_binding'], 'train', state['config']['context'])
    if data.binding_sha256 != state['contract']['binding_sha256']:
        raise ValueError('Checkpoint dataset binding changed')
    windows = recover_consumed_batch(data, state, spec['optimization']['seed'],
                                    spec['data']['source_balanced_sampling_fraction'])
    microbatch = state['contract']['microbatch']
    batches = [torch.stack([data.window(*w) for w in windows[start:start + microbatch]])
               for start in range(0, len(windows), microbatch)]
    out.mkdir(parents=True)
    report = dict(schema_version=1, checkpoint_sha256=file_hash(checkpoint),
        specification_sha256=file_hash(root / args.spec), binding_sha256=data.binding_sha256,
        diagnostic_source_sha256=file_hash(__file__),
        model_source_sha256=file_hash(Path(__file__).resolve().parents[1] / 'genie_diep/model.py'),
        cohort_source_sha256=file_hash(Path(__file__).resolve().parents[1] / 'genie_diep/cohort.py'),
        checkpoint_implementation_sha256=state['contract']['implementation_sha256'],
        failed_update=state['step'] + 1,
        failed_samples_recovered_exactly=True, samples=[data.descriptor(*w) for w in windows],
        device='cpu', torch_version=str(torch.__version__), optimizer_updates=0,
        limitation='Native CPU FP16 is not CUDA; differences in kernels and reduction order prevent exact GPU conclusions.',
        cases=[])
    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    stage = next(s for s in spec['stages'] if s['name'] == 'tokenizer')
    def loss_fn(model, x):
        prediction, _, vq = model.tokenizer(x)
        return 10 * image_loss(prediction, x, color_balance=stage['color_balance']) + vq
    for precision, scale in [('fp32', 1)] + [('fp16', scale) for scale in args.scales]:
        model = WorldModel(Config(**state['config']))
        model.load_state_dict(state['model'])
        model.requires_grad_(False)
        model.tokenizer.requires_grad_(True)
        model.eval()
        model.tokenizer.train()
        report['cases'].append({})
        def progress(case):
            report['cases'][-1] = case
            save()
            item = case['microbatches'][-1]
            print(json.dumps(dict(precision=precision, scale=scale, index=item['index'],
                loss=item['loss'], nonfinite_gradient_parameters=item.get('nonfinite_gradient_parameters', []))), flush=True)
        report['cases'][-1] = backward_case(model, batches, precision, scale, loss_fn, progress, args.fp16_backend)
        save()
        print(json.dumps({k: v for k, v in report['cases'][-1].items() if k != 'microbatches'}), flush=True)
        del model
        gc.collect()
    report['checkpoint_file_unchanged'] = file_hash(checkpoint) == report['checkpoint_sha256']
    if not report['checkpoint_file_unchanged']:
        raise AssertionError('Source checkpoint changed during diagnosis')
    report['complete'] = True
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--spec', default='configs/training-baseline-v1.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--scales', type=float, nargs='+', default=[65536, 1024, 128])
    parser.add_argument('--fp16-backend', choices=['native', 'mkldnn-forward'], default='mkldnn-forward',
                        help='Native backward is required on CPUs without oneDNN FP16 backward support')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
