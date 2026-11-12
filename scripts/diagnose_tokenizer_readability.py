"""CPU training-only ridge readouts of natural pre/post-VQ features.

This measures accessibility to a fixed linear probe, not semantic G1 retention or
irreversible information loss. Target pixels enter fitting only on training sources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.diagnose_tokenizer_regression import GEOMETRY, geometry_grid, geometry_profile
from scripts.evaluate_tokenizer_quality import validate_state


ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2)
LABELS = ('baseline_pre', 'baseline_post', 'detail_pre', 'detail_post', 'raw')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def source_groups(items):
    sources = [r['source_id'] for r in items]
    if len(sources) != 84 or len(set(sources)) != 84:
        raise ValueError('Expected one frozen calibration window per training source')
    order = sorted(sources, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    return set(order[:64]), set(order[64:])


def patches(image):
    return F.pixel_unshuffle(image[None], 4)[0].permute(1, 2, 0).reshape(-1, 48)


def unpatch(rows):
    return F.pixel_shuffle(rows.reshape(32, 32, 48).permute(2, 0, 1)[None], 4)[0]


def neighborhoods(features):
    """Identical zero-padded 3x3 token neighborhoods for every representation."""
    return F.unfold(features.permute(2, 0, 1)[None], 3, padding=1)[0].T.contiguous()


def fit_ridge(x, y, alpha):
    x, y = x.double(), y.double()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    covariance = xc.T@xc/len(x)
    cross = xc.T@yc/len(x)
    weight = torch.linalg.solve(covariance+alpha*torch.eye(x.shape[1], dtype=x.dtype), cross)
    return weight, xm, ym


def predict(x, fit, clamp=True):
    w, xm, ym = fit
    image = unpatch(((x.double()-xm)@w+ym).float())
    return image.clamp(0, 1) if clamp else image


def metrics(target, prediction, coordinates=None):
    residual = prediction-target
    luma = target.new_tensor([.2126, .7152, .0722])[:, None, None]
    lr = (residual*luma).sum(0)
    edge = sum(float(lr.diff(dim=axis).square().mean()) for axis in (-2, -1))/2
    gray = target.amax(0)-target.amin(0) <= .04
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
    center = (xx-64).square()+(yy-64).square() <= 24**2
    out = dict(rgb_mse=float(residual.square().mean()), luma_mse=float(lr.square().mean()),
               luma_gradient_mse=edge, score=float(residual.square().mean())+edge,
               luma_bias=float(lr.mean()), neutral_luma_bias=float(lr[gray].mean()) if gray.any() else None,
               center_rgb_mse=float(residual[:, center].square().mean()))
    if coordinates:
        ring, sector = geometry_grid(target.shape, coordinates)
        out['annular_rgb_mse'] = float(residual[:, ring].square().mean())
        out['annular_luma_mse'] = float(lr[ring].square().mean())
        out['annular_profile'] = geometry_profile(prediction, ring, sector)
    return out


def aggregate(rows):
    keys = ('rgb_mse', 'luma_mse', 'luma_gradient_mse', 'score', 'luma_bias',
            'neutral_luma_bias', 'center_rgb_mse', 'annular_rgb_mse', 'annular_luma_mse')
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in keys if any(r.get(k) is not None for r in rows)}


def sheet(path, images, crop=False):
    panels = []
    for label, frame in images.items():
        pixels = frame.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(pixels)
        if crop:
            image = image.crop((32, 32, 96, 96))
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        panel = Image.new('RGB', (256, 280), 'white')
        panel.paste(image, (0, 24)); ImageDraw.Draw(panel).text((4, 5), label, fill='black')
        panels.append(panel)
    canvas = Image.new('RGB', (256*len(panels), 280), 'white')
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i*256, 0))
    canvas.save(path)


def invariants(model, clip):
    tok = model.tokenizer
    with torch.no_grad():
        z = F.normalize(tok.features(clip[None]), dim=-1)
        q, ids, _ = tok.encode(clip[None])
        lookup = tok.vq.lookup(ids)
        changed = clip.clone(); changed[-1] = 1-changed[-1]
        earlier = tok.features(changed[None])[:, :-1]
        original = tok.features(clip[None])[:, :-1]
        torch.testing.assert_close(earlier, original, atol=0, rtol=0)
        torch.testing.assert_close(q, lookup, atol=2e-7, rtol=0)
        native = tok.decode(q)
        decoded = tok.decode_ids(ids)
        torch.testing.assert_close(native, decoded, atol=2e-6, rtol=0)
    x = z.detach().clone().requires_grad_(True)
    output, _, _ = tok.vq(x)
    enc_grad, code_grad = torch.autograd.grad(output.square().mean(),
        (x, tok.vq.embedding.weight), allow_unused=True)
    if enc_grad is None or not bool(enc_grad.norm()>0) or code_grad is not None:
        raise ValueError('Unexpected reconstruction gradient routing')
    _, _, auxiliary = tok.vq(x)
    aux = torch.autograd.grad(auxiliary, (x, tok.vq.embedding.weight))
    if not all(torch.isfinite(g).all() and bool(g.norm()>0) for g in aux):
        raise ValueError('Invalid VQ auxiliary gradients')
    return dict(causal_prefix_max=float((earlier-original).abs().max()),
                native_lookup_max=float((q-lookup).abs().max()),
                native_decode_ids_max=float((native-decoded).abs().max()),
                reconstruction_encoder_gradient_norm=float(enc_grad.norm()),
                reconstruction_codebook_gradient='absent as intended',
                auxiliary_gradient_norms=[float(g.norm()) for g in aux],
                feature_norm_max_error=float((z.norm(dim=-1)-1).abs().max()))


def run(args):
    started = perf_counter(); torch.set_num_threads(args.threads)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if not out.is_relative_to(root/'runs'):
        raise ValueError('Diagnostic artifacts must remain in canonical runs')
    out.mkdir(parents=True, exist_ok=False)
    spec = read(root/'configs/training-baseline-v1.json')
    selection_path = root/spec['evaluation_selection']; selection = read(selection_path)
    binding = file_hash(root/spec['data_binding']); selection_hash = file_hash(selection_path)
    if selection['binding_sha256'] != binding:
        raise ValueError('Frozen selection mismatch')
    fit_sources, tune_sources = source_groups(selection['training_calibration'])
    val_sources = {r['source_id'] for r in selection['validation']}
    if (fit_sources|tune_sources)&val_sources:
        raise ValueError('Source leakage')
    plan = dict(schema_version=1, frozen_at=datetime.now(timezone.utc).isoformat(),
                purpose='Geometry-sensitive linear readability; not semantic retention or promotion',
                script_sha256=file_hash(__file__), binding_sha256=binding, selection_sha256=selection_hash,
                fit_sources=sorted(fit_sources), tune_sources=sorted(tune_sources),
                target='last frame RGB, 4x4 patches, all 1024 sites; no masks/augmentation',
                contexts='16 observed frames encoder; final 3x3 spatial tokens, 64 dims; no future frame',
                alpha_candidates=ALPHAS, selection='Shared alpha minimizing mean tune-source RGB+luma-gradient MSE across four probes',
                no_feature_standardization=True, solve='FP64 centered normal equations with intercept',
                positive_control='Packed raw 4x4 RGB padded to64 dims: trivial target-containing sensitivity upper bound',
                validation_geometry_indices=sorted(GEOMETRY), training_geometry_indices=[23],
                tokenizer_updates=0, cpu_threads=args.threads)
    models = {}; receipts = {}
    for arm in ('baseline', 'detail'):
        path = root/f'runs/tokenizer-continuation-v1-2026-11-10/continuation/{arm}/step-005000.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_state(state, spec, binding, selection_hash)
        if state['step'] != 5000 or state['contract']['preflight']:
            raise ValueError('Expected frozen production step5000')
        model = WorldModel(Config(**state['config'])); model.load_state_dict(state['model']); model.eval()
        models[arm] = model
        receipts[arm] = dict(path=str(path.relative_to(root)), sha256=file_hash(path))
    plan['checkpoints'] = receipts; write(out/'frozen-plan.json', plan)
    datasets = {s: Cohort(root, spec['data_binding'], s, 16) for s in ('train', 'val')}
    first = datasets['train'].window(*datasets['train'].locate(selection['training_calibration'][0]))
    checks = {a: invariants(m, first) for a, m in models.items()}
    write(out/'invariants.json', checks)

    def extract(split, items):
        cache = []
        for index, item in enumerate(items):
            clip = datasets[split].window(*datasets[split].locate(item))
            row = dict(index=index, source_id=item['source_id'], target=clip[-1], representations={}, native={})
            with torch.no_grad():
                for arm, model in models.items():
                    tok = model.tokenizer; z = F.normalize(tok.features(clip[None]), dim=-1)
                    ids = (z@F.normalize(tok.vq.embedding.weight, dim=-1).T).argmax(-1)
                    post = tok.vq.lookup(ids)
                    row['representations'][arm+'_pre'] = z[0, -1].clone()
                    row['representations'][arm+'_post'] = post[0, -1].clone()
                    row['native'][arm] = tok.decode(post[:, -1:])[0, 0]
            raw = F.pixel_unshuffle(clip[-1: ], 4)[0]
            row['representations']['raw'] = F.pad(raw, (0, 0, 0, 0, 0, 16)).permute(1, 2, 0)
            cache.append(row)
            if (index+1)%8 == 0:
                print(json.dumps(dict(split=split, extracted=index+1, seconds=round(perf_counter()-started))), flush=True)
        return cache

    train = extract('train', selection['training_calibration'])
    fit_rows = [r for r in train if r['source_id'] in fit_sources]
    tune_rows = [r for r in train if r['source_id'] in tune_sources]
    y = torch.cat([patches(r['target']) for r in fit_rows])
    fits = {}; tuning = {}
    for label in LABELS:
        x = torch.cat([neighborhoods(r['representations'][label]) for r in fit_rows])
        fits[label] = {}
        tuning[label] = {}
        for alpha in ALPHAS:
            fitted = fit_ridge(x, y, alpha); fits[label][alpha] = fitted
            tuning[label][str(alpha)] = aggregate([metrics(r['target'], predict(neighborhoods(r['representations'][label]), fitted)) for r in tune_rows])
        print(json.dumps(dict(fitted=label, seconds=round(perf_counter()-started))), flush=True)
    scores = {a: np.mean([tuning[l][str(a)]['score'] for l in LABELS[:4]]) for a in ALPHAS}
    alpha = min(ALPHAS, key=lambda a: scores[a])
    decision = dict(frozen_at=datetime.now(timezone.utc).isoformat(), alpha=alpha,
                    shared_scores={str(a): float(v) for a, v in scores.items()}, tuning=tuning)
    write(out/'training-only-selection.json', decision)
    torch.save({label: fits[label][alpha] for label in LABELS}, out/'ridge-readouts.pt')
    layout = torch.stack([r['target'] for r in fit_rows]).mean(0)
    constant = layout.mean((-2, -1), keepdim=True).expand_as(layout)
    validation = extract('val', selection['validation'])
    records = {}
    for split, rows in (('fit', fit_rows), ('tune', tune_rows), ('validation', validation)):
        records[split] = []
        sensitivity = {r['index'] for r in sorted(rows, key=lambda r: r['source_id'])[:4]}
        for r in rows:
            images = dict(target=r['target'], native_baseline=r['native']['baseline'], native_detail=r['native']['detail'])
            images.update({label: predict(neighborhoods(r['representations'][label]), fits[label][alpha]) for label in LABELS})
            images.update(layout_mean=layout, rgb_mean=constant)
            coords = GEOMETRY.get(r['index']) if split == 'validation' else ((65, 64, 8, 16) if r['index']==23 else None)
            row = dict(index=r['index'], source_id=r['source_id'],
                       metrics={l: metrics(r['target'], p, coords) for l, p in images.items() if l!='target'})
            records[split].append(row)
            if coords or r['index'] in sensitivity:
                sheet(out/f'{split}-{r["index"]:02d}-full.png', images)
                sheet(out/f'{split}-{r["index"]:02d}-crop.png', images, crop=True)
        write(out/f'{split}-records.json', records[split])
    summary = {split: {l: aggregate([r['metrics'][l] for r in rows]) for l in rows[0]['metrics']} for split, rows in records.items()}
    rng = np.random.default_rng(1729); uncertainty = {}
    for arm in ('baseline', 'detail'):
        groups = {}
        for source in sorted(val_sources):
            rows = [r for r in records['validation'] if r['source_id']==source]
            groups[source] = {k: float(np.mean([r['metrics'][arm+'_post'][k]-r['metrics'][arm+'_pre'][k] for r in rows]))
                              for k in ('rgb_mse', 'luma_mse', 'luma_gradient_mse', 'center_rgb_mse', 'luma_bias')}
        intervals = {}
        for key in next(iter(groups.values())):
            values = np.array([v[key] for v in groups.values()])
            boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
            intervals[key] = dict(post_minus_pre=float(values.mean()), clustered_percentile95=np.quantile(boot, [.025, .975]).tolist())
        uncertainty[arm] = dict(per_source=groups, equal_source=intervals, sources=len(groups))
    for receipt in receipts.values():
        if file_hash(root/receipt['path'])!=receipt['sha256']:
            raise ValueError('Checkpoint changed')
    write(out/'report.json', dict(plan=plan, selection=decision, invariants=checks, summary=summary,
          paired_validation=uncertainty, seconds=round(perf_counter()-started, 2),
          limits='Linear accessibility only. Gray/color edges are not semantic labels. Known multi-barrel failures share one validation source. No G1 or controllability conclusion.'))
    print(json.dumps(dict(completed=True, alpha=alpha, seconds=round(perf_counter()-started))), flush=True)


def range_supplement(args):
    """Replay saved probes without fitting, selection or tokenizer updates."""
    torch.set_num_threads(args.threads)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    plan = read(out/'frozen-plan.json')
    if file_hash(root/'configs/baseline-evaluation-v1.json') != plan['selection_sha256']:
        raise ValueError('Frozen selection changed')
    fits = torch.load(out/'ridge-readouts.pt', map_location='cpu', weights_only=True)
    selection = read(root/'configs/baseline-evaluation-v1.json')
    models = {}
    for arm, receipt in plan['checkpoints'].items():
        if file_hash(root/receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen checkpoint changed')
        state = torch.load(root/receipt['path'], map_location='cpu', weights_only=True)
        m = WorldModel(Config(**state['config'])); m.load_state_dict(state['model']); m.eval()
        models[arm] = m
    records = {'fit': [], 'tune': [], 'validation': []}
    with torch.no_grad():
        for split, key in (('train', 'training_calibration'), ('val', 'validation')):
            dataset = Cohort(root, 'configs/baseline-data-v9.json', split, 16)
            if dataset.binding_sha256 != plan['binding_sha256']:
                raise ValueError('Frozen data binding changed')
            for index, item in enumerate(selection[key]):
                clip = dataset.window(*dataset.locate(item)); target = clip[-1]
                representations = {}
                for arm, m in models.items():
                    z = F.normalize(m.tokenizer.features(clip[None]), dim=-1)
                    ids = (z@F.normalize(m.tokenizer.vq.embedding.weight, dim=-1).T).argmax(-1)
                    representations[arm+'_pre'] = z[0, -1]
                    representations[arm+'_post'] = m.tokenizer.vq.lookup(ids)[0, -1]
                raw = F.pixel_unshuffle(target[None], 4)[0]
                representations['raw'] = F.pad(raw, (0, 0, 0, 0, 0, 16)).permute(1, 2, 0)
                group = 'validation' if split=='val' else ('fit' if item['source_id'] in plan['fit_sources'] else 'tune')
                coords = GEOMETRY.get(index) if split=='val' else ((65, 64, 8, 16) if index==23 else None)
                row = dict(index=index, source_id=item['source_id'], readouts={})
                for label, representation in representations.items():
                    unbounded = predict(neighborhoods(representation), fits[label], clamp=False)
                    row['readouts'][label] = dict(minimum=float(unbounded.min()), maximum=float(unbounded.max()),
                        outside_unit_fraction=float(((unbounded<0)|(unbounded>1)).float().mean()),
                        unclipped=metrics(target, unbounded, coords), clipped=metrics(target, unbounded.clamp(0, 1), coords))
                records[group].append(row)
            print(json.dumps(dict(range_supplement_split=split, windows=len(selection[key]))), flush=True)
    summaries = {}
    for split, rows in records.items():
        summaries[split] = {}
        for label in LABELS:
            entries = [r['readouts'][label] for r in rows]
            summaries[split][label] = dict(minimum=min(r['minimum'] for r in entries),
                maximum=max(r['maximum'] for r in entries),
                outside_unit_fraction=float(np.mean([r['outside_unit_fraction'] for r in entries])),
                unclipped=aggregate([r['unclipped'] for r in entries]), clipped=aggregate([r['clipped'] for r in entries]))
    path = out/'range-supplement.json'
    if path.exists():
        raise FileExistsError('Supplement already exists')
    write(path, dict(created_at=datetime.now(timezone.utc).isoformat(), script_sha256=file_hash(__file__),
        frozen_plan_sha256=file_hash(out/'frozen-plan.json'), original_report_sha256=file_hash(out/'report.json'),
        readout_sha256=file_hash(out/'ridge-readouts.pt'), refits=0, new_selection=False,
        summary=summaries, records=records,
        limits='Uniform RGB fitting can underweight thin gray structures. One frame/source does not characterize exposure. Linear accessibility gaps do not prove irreversible loss or select a vocabulary size. Visuals use clipped outputs; both metrics reported here.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--range-supplement', action='store_true')
    args = p.parse_args()
    range_supplement(args) if args.range_supplement else run(args)
