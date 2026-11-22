"""Local real-train-window CPU checks of a small-codebook staged tokenizer.

One update per invocation is execution evidence only, never quality evidence.
Stage two must consume the stage-one handoff and freezes encoder and codebook.
"""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, VideoTokenizer, image_loss
from scripts.preflight_transformer_tokenizer import peak_memory_bytes


def run(args):
    started = perf_counter()
    torch.set_num_threads(args.threads)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if not out.is_relative_to(root / 'runs'):
        raise ValueError('Private outputs must be under project runs')
    spec_path = root / args.spec
    spec = read(spec_path)
    if any(spec[k] for k in ('training_approved', 'execution_ready', 'local_execution_ready', 'gpu_authorized')):
        raise ValueError('Expected local-only unapproved production configuration')
    if spec['model']['vocab'] != 16 or spec['model']['token_dim'] != 64:
        raise ValueError('This preflight explicitly requires 16 codes of dimension 64')
    phase = spec['preflight_phase']
    if phase not in ('joint', 'decoder_only') or (phase == 'decoder_only') != bool(args.handoff):
        raise ValueError('Frozen decoder phase requires a joint-phase handoff')
    out.mkdir(parents=True, exist_ok=False)
    seed = spec['optimization']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = VideoTokenizer(Config(**spec['model'])).cpu().train()
    counts = {k: sum(p.numel() for p in getattr(tok, k).parameters()) for k in ('encoder', 'decoder', 'vq')}
    if counts != spec['parameter_counts']:
        raise ValueError('Declared parameter counts differ')
    selection_path = root / spec['evaluation_selection']
    selection = read(selection_path)
    data = Cohort(root, spec['data_binding'], 'train', spec['model']['context'])
    if data.binding_sha256 != selection['binding_sha256']:
        raise ValueError('Frozen training binding mismatch')
    descriptor = selection['training_calibration'][0]
    clip = data.window(*data.locate(descriptor))[None]
    if phase == 'joint':
        with torch.no_grad():
            tok.vq.initialize(tok.features(clip))
    else:
        handoff_path = Path(args.handoff).resolve()
        if not handoff_path.is_relative_to(root / 'runs'):
            raise ValueError('Handoff must be a private local artifact')
        handoff = torch.load(handoff_path, map_location='cpu', weights_only=True)
        if (handoff['phase'] != 'joint' or handoff['updates'] != 1 or
                handoff['binding_sha256'] != data.binding_sha256 or handoff['source'] != descriptor):
            raise ValueError('Unexpected handoff provenance')
        first = Config(**handoff['model_config'])
        second = Config(**spec['model'])
        first.tokenizer_transformer = {k: v for k, v in first.tokenizer_transformer.items() if k.startswith('encoder_')}
        second.tokenizer_transformer = {k: v for k, v in second.tokenizer_transformer.items() if k.startswith('encoder_')}
        if first != second:
            raise ValueError('Encoder and data dimensions must match handoff')
        for name in ('encoder', 'vq'):
            module = getattr(tok, name)
            module.load_state_dict(handoff[name], strict=True)
            module.requires_grad_(False).eval()
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, handoff[name][key], rtol=0, atol=0)
    before = {k: v.clone() for k, v in tok.state_dict().items()}
    params = [p for p in tok.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=spec['optimization']['learning_rate'],
                                  weight_decay=spec['optimization']['weight_decay'])
    forward_start = perf_counter()
    if phase == 'joint':
        q, ids, auxiliary = tok.encode(clip)
    else:
        with torch.no_grad():
            q, ids, auxiliary = tok.encode(clip)
    with torch.no_grad():
        torch.testing.assert_close(q.detach(), tok.vq.lookup(ids), rtol=0, atol=1e-7)
        if phase == 'decoder_only':
            torch.testing.assert_close(ids, handoff['reference_ids'], rtol=0, atol=0)
            torch.testing.assert_close(tok.vq.lookup(ids), handoff['reference_embeddings'], rtol=0, atol=0)
    prediction = tok.decode(q) if phase == 'joint' else tok.decode_ids(ids)
    if prediction.shape != clip.shape or not torch.isfinite(prediction).all():
        raise ValueError('Invalid reconstructed output')
    if ids.min() < 0 or ids.max() >= 16 or ids.shape != (1, 16, 32, 32):
        raise ValueError('Invalid code grid')
    loss = 10 * image_loss(prediction, clip, color_balance=True)
    if phase == 'joint':
        loss = loss + auxiliary
    forward_seconds = perf_counter() - forward_start
    backward_start = perf_counter()
    loss.backward()
    gradients = [p.grad for p in params]
    if (not torch.isfinite(loss) or any(g is None for g in gradients) or
            not all(torch.isfinite(g).all() for g in gradients)):
        raise ValueError('Missing or nonfinite gradient')
    norm = torch.nn.utils.clip_grad_norm_(params, spec['optimization']['gradient_clip_norm'])
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite gradient norm')
    optimizer.step()
    backward_seconds = perf_counter() - backward_start
    if not all(torch.isfinite(p).all() for p in tok.parameters()):
        raise ValueError('Nonfinite updated parameter')
    changes = {name: sum(not torch.equal(value, before[key]) for key, value in tok.state_dict().items()
                         if key.startswith(name + '.')) for name in ('encoder', 'vq', 'decoder')}
    if phase == 'joint':
        if not all(changes.values()):
            raise ValueError('Expected each joint component to update')
        tok.eval()
        with torch.no_grad():
            _, reference_ids, _ = tok.encode(clip)
            reference_embeddings = tok.vq.lookup(reference_ids)
        torch.save(dict(phase='joint', updates=1, model_config=spec['model'],
                        encoder=tok.encoder.state_dict(), vq=tok.vq.state_dict(),
                        reference_ids=reference_ids, reference_embeddings=reference_embeddings,
                        binding_sha256=data.binding_sha256, source=descriptor,
                        quality='unmeasured'), out / 'handoff.pt')
    else:
        if changes['encoder'] or changes['vq'] or not changes['decoder']:
            raise ValueError('Freeze/update invariant failed')
        if any(p.grad is not None for module in (tok.encoder, tok.vq) for p in module.parameters()):
            raise ValueError('Frozen component received gradients')
        with torch.no_grad():
            after_q, after_ids, _ = tok.encode(clip)
            torch.testing.assert_close(ids, after_ids, rtol=0, atol=0)
            torch.testing.assert_close(q, after_q, rtol=0, atol=0)
            torch.testing.assert_close(tok.vq.lookup(after_ids), handoff['reference_embeddings'], rtol=0, atol=0)
    report = dict(purpose='CPU execution check only; not trained quality or GPU readiness',
                  phase=phase, device='cpu', dtype='float32', torch_version=str(torch.__version__),
                  cpu_threads=args.threads, shape=list(clip.shape), grid_shape=list(ids.shape),
                  parameter_counts=counts, trainable_parameters=sum(p.numel() for p in params),
                  source=descriptor, split='train', binding_sha256=data.binding_sha256,
                  spec_sha256=file_hash(spec_path), selection_sha256=file_hash(selection_path),
                  script_sha256=file_hash(__file__), model_sha256=file_hash(root / 'genie_diep/model.py'),
                  transformer_sha256=file_hash(root / 'genie_diep/transformer_tokenizer.py'),
                  loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                  gradients_finite=True, output_finite=True, code_lookup_matches=True,
                  changed_state_tensors=changes, frozen_encoder_codebook_exact=phase == 'decoder_only',
                  frozen_ids_embeddings_exact=phase == 'decoder_only',
                  handoff_reference_ids_embeddings_exact=phase == 'decoder_only',
                  handoff_sha256=file_hash(args.handoff if args.handoff else out / 'handoff.pt'),
                  forward_seconds=round(forward_seconds, 3), backward_update_seconds=round(backward_seconds, 3),
                  seconds=round(perf_counter()-started, 3), process_peak_working_set_bytes=peak_memory_bytes(),
                  updates=1, gpu_allocated=False, quality='unmeasured',
                  initialization='Stage one first flattened feature rows; stage two exact one-update encoder/codebook handoff')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--handoff')
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
