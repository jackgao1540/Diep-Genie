"""CPU-only component and geometry probes for the frozen tokenizer comparison."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.detail_loss import detail_loss
from genie_diep.model import Config, WorldModel
from genie_diep.object_quality import compare
from scripts.evaluate_tokenizer_quality import native_frame, reconstruct, uint8_frames, validate_state


STEPS = {'b3': ('baseline', 3000), 'b4': ('baseline', 4000), 'b45': ('baseline', 4500),
         'b5': ('baseline', 5000), 'd3': ('detail', 3000), 'd4': ('detail', 4000),
         'd5': ('detail', 5000)}
GEOMETRY = {4: (65, 58, 6, 13), 12: (67, 62, 8, 16), 16: (66, 63, 7, 15),
            28: (58, 59, 8, 17), 36: (63, 60, 7, 15), 37: (63, 74, 8, 16),
            38: (66, 65, 8, 16), 39: (65, 65, 8, 16), 40: (64, 65, 8, 17),
            56: (64, 55, 8, 17)}
MARGIN_BINS = (.001, .002, .005, .01, .02, .05, .1, .2)


def variants():
    for encoder in ('b4', 'b5'):
        for codebook in ('b4', 'b5'):
            for decoder in ('b4', 'b5'):
                yield f'{encoder}/{codebook}/{decoder}', encoder, codebook, decoder, 0.
    for key in STEPS:
        if key not in ('b4', 'b5'):
            yield f'{key}/{key}/{key}', key, key, key, 0.
        if key in ('b4', 'b5', 'd5'):
            for alpha in (.25, 1.):
                yield f'{key}:alpha={alpha}', key, key, key, alpha


def quantize(z, codes, objective=False):
    scores = z @ codes.T
    ids = scores.argmax(-1)
    nearest = F.embedding(ids, codes)
    # Match the existing straight-through forward numerics.
    q = z + (nearest-z)
    if not objective:
        return q, ids, None, None
    top = scores.topk(2, dim=-1).values
    logits = (scores/.15).reshape(-1, len(codes))
    probability = logits.softmax(-1)
    aggregate = probability.mean(0)
    balance = (aggregate * (aggregate.clamp_min(1e-8).log()+math.log(len(codes)))).sum()
    entropy = -probability.mul(logits.log_softmax(-1)).sum(-1).mean()
    spread = F.relu(.1-z.reshape(-1, z.shape[-1]).std(0, unbiased=False)).mean()
    vq_loss = .125*F.mse_loss(nearest, z) + .1*balance + .01*entropy + spread
    return q, ids, top[..., 0]-top[..., 1], vq_loss


def decode(decoder, q, z, alpha):
    if alpha:
        q = F.normalize(q.lerp(z, alpha), dim=-1)
    return decoder.decode(q)[0]


def prepare_target(source):
    high = source.amax(1)
    chroma = high-source.amin(1)
    colored = (high >= 65/255) & (chroma/high.clamp_min(1e-8) >= 65/255)
    horizontal = source.diff(dim=-1)
    vertical = source.diff(dim=-2)
    seed_x = horizontal.abs().amax(1) >= .08
    seed_y = vertical.abs().amax(1) >= .08
    seed = (F.pad(seed_x, (0, 1)) | F.pad(seed_x, (1, 0)) |
            F.pad(seed_y, (0, 0, 0, 1)) | F.pad(seed_y, (0, 0, 1, 0)))
    edge = F.max_pool2d(seed[:, None].float(), 3, stride=1, padding=1)[:, 0].bool()
    regions = {
        'whole': torch.ones_like(colored),
        'colored_edge': colored & edge,
        'colored_flat': colored & ~edge,
        'noncolored_edge': ~colored & edge,
        'noncolored_flat': ~colored & ~edge,
        'neutral_edge': (chroma <= .12) & edge,
        'neutral_flat': (chroma <= .04) & ~edge,
    }
    weights = 1+10*(chroma > .12).float()
    weights /= weights.mean((-2, -1), keepdim=True)
    r, g, b = source.unbind(1)
    color_groups = torch.stack(((r-b > .12) & (g-b > .12), (r-g > .12) & (r-b > .08),
                                (b-r > .12) & (g-r > .12) & (b-g < .12),
                                (b-r > .12) & (b-g >= .12)), dim=1).float()
    return regions, weights, color_groups, horizontal, vertical


def reconstruction_metrics(source, prediction, target, native_vq=None):
    regions, weights, color_groups, horizontal, vertical = target
    residual = prediction-source
    errors = residual.square()
    moments = {}
    for name, mask in regions.items():
        weight = mask[:, None]
        moments[name] = [int(mask.sum()), (residual*weight).sum((0, 2, 3), dtype=torch.float64).tolist(),
                         (errors*weight).sum((0, 2, 3), dtype=torch.float64).tolist()]
    luminance = source.new_tensor([.2126, .7152, .0722]).reshape(1, 3, 1, 1)
    edge = {}
    gradient_mae = 0.
    for diff, axis in ((horizontal, -1), (vertical, -2)):
        rgb = residual.diff(dim=axis)
        gradient_mae += float(rgb.abs().mean())
        luma = (rgb*luminance).sum(1, keepdim=True)
        target_luma = (diff*luminance).sum(1, keepdim=True)
        target_chroma = diff-target_luma
        chroma = rgb-luma
        for region, mask in (('luma_edge', target_luma.abs() >= .08),
                             ('chroma_edge', target_chroma.abs().amax(1, keepdim=True) >= .08)):
            values = luma if region == 'luma_edge' else chroma
            weight = mask.expand_as(values)
            total, count = edge.get(region, (0., 0))
            edge[region] = [total+float((values.square()*weight).sum(dtype=torch.float64)),
                            count+int(weight.sum())]
    flat = regions['neutral_flat'][:, None]
    height, width = source.shape[-2:]
    def by_phase(values):
        return values.reshape(len(source), -1, height//4, 4, width//4, 4).sum((0, 2, 4), dtype=torch.float64)
    phase = dict(count=by_phase(flat)[0].tolist(), source_sum=by_phase(source*flat).tolist(),
                 residual_sum=by_phase(residual*flat).tolist(), residual_squared_sum=by_phase(errors*flat).tolist())
    result = dict(regions=moments, edge=edge, neutral_flat_phase=phase,
                  temporal=[float(residual.diff(dim=0).square().sum(dtype=torch.float64)),
                            residual[1:].numel()],
                  output_mean=prediction.mean((0, 2, 3)).tolist(),
                  output_min=prediction.amin((0, 2, 3)).tolist(),
                  output_max=prediction.amax((0, 2, 3)).tolist(),
                  output_saturation=((prediction < .01) | (prediction > .99)).float().mean((0, 2, 3)).tolist())
    if native_vq is not None:
        counts = color_groups.sum((-2, -1))
        valid = counts >= 4
        color_errors = (errors.mean(1, keepdim=True)*color_groups).sum((-2, -1))/counts.clamp_min(1)
        chroma_mse = float((errors*weights[:, None]).mean())
        mae = float(residual.abs().mean())
        color_balance = float((color_errors*valid).sum()/valid.sum().clamp_min(1))
        image_loss = chroma_mse + .1*mae + .2*gradient_mae + color_balance
        detail = float(detail_loss(prediction[None], source[None]))
        result['objective'] = dict(chroma_weighted_mse=chroma_mse, mae=mae, rgb_gradient_mae=gradient_mae,
                                   color_group_mse=color_balance, image_loss=image_loss, detail=detail,
                                   vq=float(native_vq), baseline_total=10*image_loss+float(native_vq))
    return result


def code_masks(target):
    regions = target[0]
    masks = {'whole': torch.ones_like(regions['whole'][:, ::4, ::4])}
    # A token's 4x4 output footprint is a spatial label, not its receptive field.
    for name in ('colored_edge', 'colored_flat', 'neutral_edge', 'neutral_flat'):
        masks[name] = F.avg_pool2d(regions[name][:, None].float(), 4, stride=4)[:, 0] >= .25
    _, height, width = masks['whole'].shape
    for row in range(2):
        for col in range(2):
            mask = torch.zeros_like(masks['whole'])
            mask[:, row*height//2:(row+1)*height//2, col*width//2:(col+1)*width//2] = True
            masks[f'quadrant_{row}{col}'] = mask
    return masks


def code_statistics(z, q, ids, margins, masks, vocab):
    z, q, ids, margins = z[0], q[0], ids[0], margins[0]
    distance = (q-z).square().sum(-1)
    bins = torch.bucketize(margins.contiguous(), margins.new_tensor(MARGIN_BINS))
    result = {}
    for name, mask in masks.items():
        result[name] = [int(mask.sum()), float(distance[mask].sum(dtype=torch.float64)),
                        float(margins[mask].sum(dtype=torch.float64)),
                        torch.bincount(ids[mask], minlength=vocab).tolist(),
                        torch.bincount(bins[mask], minlength=len(MARGIN_BINS)+1).tolist()]
    return result


def quant_geometry(z, q, ids, margin, ring):
    footprint = F.avg_pool2d(ring[None, None].float(), 4, stride=4)[0, 0] > 0
    z, q, ids, margin = z[0, -1], q[0, -1], ids[0, -1], margin[0, -1]
    return [dict(y=int(y), x=int(x), id=int(ids[y, x]),
                 squared_distance=float((z[y, x]-q[y, x]).square().sum()),
                 nearest_margin=float(margin[y, x])) for y, x in footprint.nonzero()]


def geometry_grid(shape, coordinates):
    cx, cy, inner, outer = coordinates
    y, x = torch.meshgrid(torch.arange(shape[-2])-cy, torch.arange(shape[-1])-cx, indexing='ij')
    radius = (x.square()+y.square()).sqrt()
    theta = (torch.atan2(y, x)+2*math.pi) % (2*math.pi)
    return (radius >= inner) & (radius <= outer), (theta*(36/(2*math.pi))).long().clamp_max(35)


def geometry_profile(frame, ring, sector):
    luma = (frame*frame.new_tensor([.2126, .7152, .0722])[:, None, None]).sum(0)
    chroma = frame.amax(0)-frame.amin(0)
    counts = torch.bincount(sector[ring], minlength=36)
    return dict(luma=(torch.bincount(sector[ring], weights=luma[ring], minlength=36)/counts).tolist(),
                chroma=(torch.bincount(sector[ring], weights=chroma[ring], minlength=36)/counts).tolist())


def means(rows):
    result = {}
    for region in rows[0]['regions']:
        moments = [r['regions'][region] for r in rows]
        count = sum(m[0] for m in moments)
        if not count:
            result[region] = None
            continue
        bias = np.sum([m[1] for m in moments], axis=0)/count
        mse = np.sum([m[2] for m in moments], axis=0)/count
        result[region] = dict(pixels=count, bias=bias.tolist(), mse=float(mse.mean()),
                              mse_per_channel=mse.tolist(), bias_square_mse=float(np.square(bias).mean()),
                              centered_mse=float((mse-np.square(bias)).mean()))
    for name, regions in (('colored', ('colored_edge', 'colored_flat')),
                          ('noncolored', ('noncolored_edge', 'noncolored_flat'))):
        parts = [result[region] for region in regions if result[region]]
        count = sum(p['pixels'] for p in parts)
        if count:
            bias = sum(np.array(p['bias'])*p['pixels'] for p in parts)/count
            mse = sum(np.array(p['mse_per_channel'])*p['pixels'] for p in parts)/count
            result[name] = dict(pixels=count, bias=bias.tolist(), mse=float(mse.mean()),
                                mse_per_channel=mse.tolist(), bias_square_mse=float(np.square(bias).mean()),
                                centered_mse=float((mse-np.square(bias)).mean()))
    for key in rows[0]['edge']:
        total, count = np.sum([r['edge'][key] for r in rows], axis=0)
        result[key] = dict(mse=float(total/count) if count else None, samples=int(count))
    total, count = np.sum([r['temporal'] for r in rows], axis=0)
    result['temporal_mse'] = float(total/count)
    for key in ('output_mean', 'output_saturation'):
        result[key] = np.mean([r[key] for r in rows], axis=0).tolist()
    result['output_min'] = np.min([r['output_min'] for r in rows], axis=0).tolist()
    result['output_max'] = np.max([r['output_max'] for r in rows], axis=0).tolist()
    phase = {key: np.sum([r['neutral_flat_phase'][key] for r in rows], axis=0)
             for key in rows[0]['neutral_flat_phase']}
    count = phase.pop('count')
    result['neutral_flat_phase'] = dict(count=count.astype(int).tolist())
    for key, values in phase.items():
        result['neutral_flat_phase'][key] = (
            np.divide(values, count, out=np.zeros_like(values), where=count > 0).tolist())
    if 'objective' in rows[0]:
        result['objective'] = {key: float(np.mean([r['objective'][key] for r in rows]))
                               for key in rows[0]['objective']}
    return result


def summarize_codes(rows):
    result = {}
    for name in rows[0]:
        entries = [row[name] for row in rows]
        count = sum(e[0] for e in entries)
        if not count:
            result[name] = None
            continue
        histogram = np.sum([e[3] for e in entries], axis=0)
        probability = histogram[histogram > 0]/count
        entropy = float(-(probability*np.log(probability)).sum())
        margins = np.sum([e[4] for e in entries], axis=0)
        result[name] = dict(tokens=count, distortion=sum(e[1] for e in entries)/count,
                            mean_margin=sum(e[2] for e in entries)/count,
                            used_codes=len(probability), entropy_nats=entropy, perplexity=math.exp(entropy),
                            margin_histogram=margins.tolist(), code_histogram=histogram.tolist())
    return result


def geometry_sheet(path, source, native, predictions, coordinates):
    cx, cy, _, outer = coordinates
    radius = outer+4
    names = ['native crop', '128px target', *predictions]
    arrays = [native, source, *predictions.values()]
    side, cols = 200, 5
    sheet = Image.new('RGB', (cols*side, math.ceil(len(names)/cols)*(side+24)), 'white')
    painter = ImageDraw.Draw(sheet)
    for i, (name, array) in enumerate(zip(names, arrays)):
        if array is None:
            continue
        x, y = i%cols*side, i//cols*(side+24)
        height, width = array.shape[:2]
        box = (round((cx-radius)*width/128), round((cy-radius)*height/128),
               round((cx+radius+1)*width/128), round((cy+radius+1)*height/128))
        sheet.paste(Image.fromarray(array).crop(box).resize((side, side), Image.Resampling.NEAREST), (x, y+24))
        painter.text((x+4, y+4), name, fill='black')
    sheet.save(path)


def load_models(root, spec, binding_hash, selection_hash):
    models, receipts = {}, {}
    expected = read(root/'configs/tokenizer-continuation-v1-results.json')
    audit = expected['private_artifact_index']['reports_and_final_checkpoints']['artifact_audit']
    audit_path = root/audit['path']
    if file_hash(audit_path) != audit['sha256']:
        raise ValueError('Continuation audit hash mismatch')
    checkpoints = {entry['path']: entry['sha256'] for entry in read(audit_path)['checkpoints']}
    for key, (arm, step) in STEPS.items():
        path = (root/'runs/tokenizer-continuation-v1-2026-11-10/continuation'/arm/f'step-{step:06d}.pt')
        digest = file_hash(path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_state(state, spec, binding_hash, selection_hash)
        if state['step'] != step:
            raise ValueError(f'Wrong checkpoint step for {key}')
        if digest != checkpoints[f'{arm}/{path.name}']:
            raise ValueError(f'Checkpoint hash missing from audited results for {key}')
        model = WorldModel(Config(**state['config'])).eval().requires_grad_(False)
        model.load_state_dict(state['model'])
        models[key] = model.tokenizer
        receipts[key] = dict(path=str(path.relative_to(root)), sha256=digest, step=step)
    return models, receipts


def check_native(model, codes, clip):
    reference, ids = reconstruct(model, clip)
    z = F.normalize(model.features(clip[None]), dim=-1)
    q, current_ids, _, vq = quantize(z, codes, objective=True)
    prediction = decode(model, q, z, 0.)
    reference_vq = model.vq(model.features(clip[None]))[2]
    if not torch.equal(current_ids[0], ids):
        raise ValueError('Whole-window quantization changed native token assignments')
    torch.testing.assert_close(prediction, reference, rtol=0, atol=3e-6)
    torch.testing.assert_close(vq, reference_vq, rtol=0, atol=3e-7)
    return float((prediction-reference).abs().max())


def component_counts(source, prediction):
    matches, extra = compare(source, prediction)
    return dict(unmatched_prediction=extra,
                by_area={f'{low}-{high}': dict(target=len(rows), matched=sum(r['matched'] for r in rows),
                                               iou_sum=sum(r['iou'] for r in rows),
                                               boundary_f1_sum=sum(r['boundary_f1'] for r in rows))
                         for low, high in ((4, 7), (8, 15), (16, 31), (32, 63), (64, 16384))
                         for rows in [[r for r in matches if low <= r['source_area'] <= high]]})


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    started = perf_counter()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    spec = read(root/'configs/training-baseline-v1.json')
    binding_hash = file_hash(root/spec['data_binding'])
    selection_path = root/spec['evaluation_selection']
    selection_hash = file_hash(selection_path)
    selection = read(selection_path)
    if selection['binding_sha256'] != binding_hash:
        raise ValueError('Evaluation cohort binding mismatch')
    models, receipts = load_models(root, spec, binding_hash, selection_hash)
    codes = {name: F.normalize(model.vq.embedding.weight, dim=-1) for name, model in models.items()}
    all_variants = list(variants())
    similarities = codes['b4']@codes['b5'].T
    sources = {}
    for recipe in read(root/'configs/longform-cohort-v6.json')['source_recipes']:
        sources.update({source['id']: source for source in read(root/recipe)['sources']})
    report = dict(schema_version=1, cpu_threads=args.threads, torch_version=str(torch.__version__),
                  updates=0, selection_sha256=selection_hash, binding_sha256=binding_hash,
                  script_sha256=file_hash(__file__), checkpoints=receipts, margin_bins=MARGIN_BINS,
                  codebook_cosine=dict(same_index=similarities.diag().tolist(),
                                       closest_from_b4=similarities.amax(1).tolist(),
                                       closest_from_b5=similarities.amax(0).tolist()),
                  variants=all_variants, splits={}, geometry={})
    out.mkdir(parents=True, exist_ok=False)
    caps, verified = {}, set()
    try:
        for split, selection_name, expected_count in (('val', 'validation', 64),
                                                       ('train', 'training_calibration', 84)):
            dataset = Cohort(root, spec['data_binding'], split, spec['model']['context'])
            items = selection[selection_name]
            if len(items) != expected_count or {i['source_id'] for i in items} != set(dataset.sources):
                raise ValueError(f'Unexpected frozen {split} selection')
            locations = [dataset.locate(item) for item in items]
            records = []
            overlap = torch.zeros((spec['model']['vocab'], spec['model']['vocab']), dtype=torch.int64)
            for index, (item, location) in enumerate(zip(items, locations)):
                clip = dataset.window(*location)
                if split == 'val' and index in (4, 36):
                    report.setdefault('native_identity', {})[index] = {
                        name: check_native(model, codes[name], clip) for name, model in models.items()}
                target = prepare_target(clip)
                masks = code_masks(target)
                features = {name: F.normalize(model.features(clip[None]), dim=-1)
                            for name, model in models.items()}
                pairs = {(enc, code): quantize(features[enc], codes[code], objective=enc == code)
                         for enc, code in dict.fromkeys((enc, code) for _, enc, code, _, _ in all_variants)}
                ids_b4, ids_b5 = pairs['b4', 'b4'][1], pairs['b5', 'b5'][1]
                vocab = len(codes['b4'])
                overlap += torch.bincount((ids_b4*vocab+ids_b5).flatten(), minlength=vocab*vocab).reshape(vocab, vocab)
                row = dict(index=index, source_id=item['source_id'], reconstructions={}, codes={}, components={})
                for name in models:
                    q, ids, margins, _ = pairs[name, name]
                    row['codes'][name] = code_statistics(features[name], q, ids, margins, masks, vocab)
                anchor = split == 'val' and index in GEOMETRY
                images = {}
                source = uint8_frames(clip[-1:])[0]
                if anchor:
                    coords = GEOMETRY[index]
                    ring, sector = geometry_grid(clip.shape, coords)
                    geom = dict(center_radius=coords, target=geometry_profile(clip[-1], ring, sector),
                                reconstructions={}, latents={})
                    for name in models:
                        q, ids, margins, _ = pairs[name, name]
                        geom['latents'][name] = quant_geometry(features[name], q, ids, margins, ring)
                for label, enc, code, decoder, alpha in all_variants:
                    q, _, _, vq = pairs[enc, code]
                    prediction = decode(models[decoder], q, features[enc], alpha)
                    if not torch.isfinite(prediction).all():
                        raise ValueError(f'Nonfinite reconstruction in {label}')
                    native = enc == code == decoder and not alpha
                    row['reconstructions'][label] = reconstruction_metrics(
                        clip, prediction, target, native_vq=vq if native else None)
                    if native or anchor:
                        pixels = uint8_frames(prediction[-1:])[0]
                        row['components'][label] = component_counts(source, pixels)
                    if anchor:
                        images[label] = pixels
                        geom['reconstructions'][label] = geometry_profile(prediction[-1], ring, sector)
                if anchor:
                    native = native_frame(dataset, item, root, sources, caps, verified)
                    if native is None:
                        raise ValueError(f'Native fidelity unavailable for anchor {index}')
                    geometry_sheet(out/f'anchor-{index:02d}.png', source, native, images, coords)
                    report['geometry'][index] = geom
                records.append(row)
                if (index+1) % 8 == 0 or index+1 == len(items):
                    print(json.dumps(dict(split=split, completed=index+1, total=len(items),
                                          seconds=round(perf_counter()-started))), flush=True)
            by_source = defaultdict(list)
            for row in records:
                by_source[row['source_id']].append(row)
            report['splits'][split] = dict(
                windows=len(records), sources=len(by_source),
                reconstructions={label: dict(
                    pooled=means([row['reconstructions'][label] for row in records]),
                    per_source={source: means([r['reconstructions'][label] for r in rows])
                                for source, rows in by_source.items()})
                    for label, *_ in all_variants},
                codes={name: summarize_codes([row['codes'][name] for row in records]) for name in models},
                code_agreement=dict(tokens=int(overlap.sum()), same_index=int(overlap.diag().sum()),
                                    sum_row_modal_count=int(overlap.amax(1).sum()),
                                    sum_column_modal_count=int(overlap.amax(0).sum())))
            (out/f'{split}-summary.json').write_text(
                json.dumps(report['splits'][split], allow_nan=False)+'\n', encoding='utf-8')
            with (out/f'{split}-windows.jsonl').open('w', encoding='utf-8') as stream:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
    finally:
        for capture in caps.values():
            capture.release()
    for checkpoint in receipts.values():
        if file_hash(root/checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Checkpoint changed during diagnosis')
    report['seconds'] = round(perf_counter()-started, 2)
    (out/'diagnosis.json').write_text(json.dumps(report, allow_nan=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
