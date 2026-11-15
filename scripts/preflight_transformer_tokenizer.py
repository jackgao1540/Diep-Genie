"""One CPU update on a frozen training window; execution evidence, never quality."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from genie_diep.baseline import implementation_hash
from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel, image_loss


def peak_memory_bytes():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def run(args):
    started = perf_counter(); torch.set_num_threads(args.threads)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if not out.is_relative_to(root/'runs'):
        raise ValueError('Keep private preflight artifacts under project runs')
    out.mkdir(parents=True, exist_ok=False)
    spec_path = root/'configs/tokenizer-transformer-v1.json'; spec = read(spec_path)
    if any(spec[k] for k in ('training_approved', 'execution_ready', 'local_execution_ready', 'gpu_authorized')):
        raise ValueError('This check requires an unapproved local-only configuration')
    seed = spec['optimization']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = WorldModel(Config(**spec['model'])).cpu()
    model.requires_grad_(False); tok = model.tokenizer; tok.requires_grad_(True).train()
    selection_path = root/spec['evaluation_selection']; selection = read(selection_path)
    dataset = Cohort(root, spec['data_binding'], 'train', model.config.context)
    if dataset.binding_sha256 != selection['binding_sha256']:
        raise ValueError('Frozen training binding mismatch')
    descriptor = selection['training_calibration'][0]
    clip = dataset.window(*dataset.locate(descriptor))[None]
    inactive = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('tokenizer.')}
    with torch.no_grad():
        # Preserve the existing initialization rule for this architecture-only check.
        tok.vq.initialize(tok.features(clip))
    optimizer = torch.optim.AdamW(tok.parameters(), lr=spec['stages'][0]['learning_rate'],
                                  weight_decay=spec['optimization']['weight_decay'])
    forward_start = perf_counter()
    prediction, ids, auxiliary = tok(clip)
    loss = 10*image_loss(prediction, clip, color_balance=True)+auxiliary
    forward_seconds = perf_counter()-forward_start
    backward_start = perf_counter(); loss.backward()
    gradients = [p.grad for p in tok.parameters() if p.grad is not None]
    if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Nonfinite loss/gradients')
    norm = torch.nn.utils.clip_grad_norm_(tok.parameters(), spec['optimization']['gradient_clip_norm'])
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite gradient norm')
    optimizer.step(); backward_seconds = perf_counter()-backward_start
    if not all(torch.isfinite(p).all() for p in tok.parameters()):
        raise ValueError('Nonfinite updated parameters')
    for name, tensor in inactive.items():
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    counts = {name: sum(p.numel() for p in getattr(tok, name).parameters()) for name in ('encoder', 'decoder', 'vq')}
    if counts != spec['parameter_counts']:
        raise ValueError('Declared parameter counts differ')
    torch.save(dict(config=model.config_dict(), model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=1, stage='tokenizer', preflight=True, quality='unmeasured'), out/'local-only.pt')
    report = dict(purpose='CPU execution check only; not trained quality or GPU readiness',
        device='cpu', torch_version=str(torch.__version__), cpu_threads=args.threads, shape=list(clip.shape),
        parameter_counts=counts, grid_shape=list(ids.shape), source=descriptor,
        spec_sha256=file_hash(spec_path), binding_sha256=dataset.binding_sha256,
        selection_sha256=file_hash(selection_path), implementation_sha256=implementation_hash(),
        loss=float(loss.detach()), gradient_norm_before_clip=float(norm), gradients_finite=True,
        inactive_action_dynamics_unchanged=True, forward_seconds=round(forward_seconds, 3),
        backward_update_seconds=round(backward_seconds, 3), seconds=round(perf_counter()-started, 3),
        process_peak_working_set_bytes=peak_memory_bytes(), updates=1, gpu_allocated=False,
        quality='unmeasured', initialization='Existing first flattened feature rows; not a representative data pool')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())
