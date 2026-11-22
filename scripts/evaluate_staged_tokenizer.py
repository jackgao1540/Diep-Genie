"""Read-only CPU FP32 assessment of the frozen-encoder staged16 endpoints.

Manifest binds {plan:{path,sha256}, execution_report:{path,sha256},
historical_metrics:{path,sha256}}. The last field is optional context only.
Production requires both full-mode step-5000 endpoints and all 64 frozen windows.
Explicit nonproduction checks cannot establish G1 or reconstruction quality.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from genie_diep.baseline import implementation_hash
from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.diagnose_tokenizer_regression import GEOMETRY, prepare_target, code_masks
from scripts.evaluate_small_codebook_architecture import (
    FROZEN_SELECTION_SHA256, aggregate, paired_differences, private_output, resolve,
    save_atlas, totals,
)
from scripts.evaluate_tokenizer_quality import validate_state, uint8_frames
from scripts.run_staged_tokenizer import validate_plan, PHASES
from scripts.run_tokenizer_comparison import compare_exact


def bound_file(root, reference):
    path = resolve(root, reference['path'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Artifact SHA256 mismatch: ' + str(path))
    return path


def validate_staged_state(state, spec, binding, selection, spec_sha, plan_sha,
                          phase, nonproduction=False):
    validate_state(state, spec, binding, selection)
    contract = state['contract']
    if contract.get('spec_sha256') != spec_sha or contract.get('plan_sha256') != plan_sha:
        raise ValueError('Staged specification/plan identity mismatch')
    if contract.get('phase') != phase or contract.get('execution_gate') != 'staged-plan-v1':
        raise ValueError('Staged phase/execution identity mismatch')
    if contract.get('implementation_sha256') != implementation_hash():
        raise ValueError('Model implementation differs from the trained endpoint')
    if spec['model']['vocab'] != 16 or spec['model']['token_dim'] != 64:
        raise ValueError('Expected staged sixteen-by-sixty-four codebook')
    if spec['model']['size'] != 128 or spec['model']['context'] != 16:
        raise ValueError('Expected frozen 128px / sixteen-frame setup')
    if contract.get('microbatch') not in (1, 2, 4) or contract['microbatch'] * contract.get('accumulation', 0) != 16:
        raise ValueError('Staged effective update batch differs')
    if not nonproduction and (state['step'] != 5000 or contract.get('staged_mode') != 'full' or
            contract.get('preflight') is not False or contract.get('device_type') != 'cuda' or
            contract.get('precision') != 'fp16'):
        raise ValueError('Expected a full production CUDA FP16 step-5000 endpoint')


def validate_frozen_pair(joint, decoder, joint_sha):
    if decoder['contract'].get('handoff_sha256') != joint_sha:
        raise ValueError('Decoder is not bound to this actual joint endpoint')
    keys = [k for k in joint['model'] if k.startswith(('tokenizer.encoder.', 'tokenizer.vq.'))]
    other = [k for k in decoder['model'] if k.startswith(('tokenizer.encoder.', 'tokenizer.vq.'))]
    if set(keys) != set(other): raise ValueError('Frozen encoder/codebook state keys differ')
    for key in keys:
        if not torch.equal(joint['model'][key], decoder['model'][key]):
            raise ValueError('Frozen encoder/codebook tensor differs: ' + key)
    return len(keys)


def validate_initial(initial, endpoint, spec, binding, selection, spec_sha):
    validate_state(initial, spec, binding, selection)
    if (initial['contract'] != endpoint['contract'] or initial['step'] != 0 or
            initial['optimizer']['state'] or initial['contract']['spec_sha256'] != spec_sha):
        raise ValueError('Initial checkpoint is not the exact fresh stage-zero state')
    inactive = [k for k in endpoint['model'] if not k.startswith('tokenizer.')]
    for key in inactive:
        if not torch.equal(initial['model'][key], endpoint['model'][key]):
            raise ValueError('Inactive action/dynamics state changed: ' + key)
    return len(inactive)


def prepare(root, manifest_path, nonproduction=False):
    manifest = read(manifest_path)
    plan_path = bound_file(root, manifest['plan'])
    plan = validate_plan(root, plan_path)
    report_path = bound_file(root, manifest['execution_report'])
    execution = read(report_path)
    if execution['identity']['plan_sha256'] != manifest['plan']['sha256']:
        raise ValueError('Execution report plan identity mismatch')
    if nonproduction:
        if execution['status'] not in ('local_check_complete', 'pilot_complete', 'full_complete'):
            raise ValueError('Only completed execution artifacts can be checked')
    elif execution['status'] != 'full_complete' or execution['identity']['mode'] != 'full' or execution['identity']['local_check']:
        raise ValueError('Production evaluation requires completed full training')
    states, entries, models, artifacts = {}, {}, {}, []
    common = None
    for phase in PHASES:
        stage = plan['stages'][phase]
        spec_path = resolve(root, stage['spec']); spec = read(spec_path)
        binding, selection = (file_hash(resolve(root, spec[key]))
                              for key in ('data_binding', 'evaluation_selection'))
        if selection != FROZEN_SELECTION_SHA256 or common not in (None, (binding, selection)):
            raise ValueError('Staged evaluation must use the original shared frozen validation identity')
        common = binding, selection
        row = execution['phases'][phase]
        if not row.get('complete'): raise ValueError('Execution phase is incomplete')
        path = (report_path.parent / row['checkpoint']['path']).resolve()
        if not path.is_relative_to(report_path.parent): raise ValueError('Checkpoint escapes the execution directory')
        if file_hash(path) != row['checkpoint']['sha256']: raise ValueError('Endpoint SHA256 mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_staged_state(state, spec, binding, selection, stage['spec_sha256'],
                              manifest['plan']['sha256'], phase, nonproduction)
        if state['step'] != row['latest_step']: raise ValueError('Execution report endpoint step differs')
        initial_path = report_path.parent / phase / 'initial.pt'
        if file_hash(initial_path) != row['initial_checkpoint_sha256']: raise ValueError('Initial checkpoint SHA256 mismatch')
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        inactive = validate_initial(initial, state, spec, binding, selection, stage['spec_sha256'])
        if phase == 'decoder_only':
            validate_frozen_pair(states['joint'], initial, entries['joint']['checkpoint_sha256'])
        model = WorldModel(Config(**state['config'])).cpu().float().eval()
        model.load_state_dict(state['model'], strict=True); model.requires_grad_(False)
        counts = {n: sum(p.numel() for p in getattr(model.tokenizer, n).parameters()) for n in ('encoder','decoder','vq')}
        if counts != spec['parameter_counts']: raise ValueError('Staged parameter counts differ')
        states[phase], models[phase] = state, model
        entries[phase] = dict(checkpoint=str(path), checkpoint_sha256=row['checkpoint']['sha256'],
            step=state['step'], training_contract=state['contract'], spec=str(spec_path),
            spec_sha256=stage['spec_sha256'], initial_checkpoint=str(initial_path),
            initial_sha256=row['initial_checkpoint_sha256'], inactive_tensors_exact=inactive,
            parameter_counts=counts)
        artifacts.extend([(path, row['checkpoint']['sha256']), (initial_path, row['initial_checkpoint_sha256'])])
    frozen_count = validate_frozen_pair(states['joint'], states['decoder_only'], entries['joint']['checkpoint_sha256'])
    spec = read(entries['joint']['spec'])
    selection_path = resolve(root, spec['evaluation_selection']); selection = read(selection_path)
    if selection['binding_sha256'] != common[0]: raise ValueError('Frozen selection binding mismatch')
    dataset = Cohort(root, spec['data_binding'], 'val', 16)
    items = selection['validation']
    if (len(items) != 64 or len({x['source_id'] for x in items}) != 16 or
            set(dataset.sources) != {x['source_id'] for x in items}):
        raise ValueError('Require all original 64 windows / sixteen held-out sources')
    for item in items: dataset.locate(item)
    anchor_path = root / 'configs/baseline-fidelity-review-v1.json'; anchors = read(anchor_path)
    anchor_map = {a['index']: a for a in anchors['manual_anchors']}
    if (anchors['evaluation_sha256'] != common[1] or set(anchor_map) != set(range(64)) or
            any(anchor_map[i]['selection'] != items[i] for i in range(64))):
        raise ValueError('Frozen semantic assessment anchor identity differs')
    artifacts.extend([(plan_path, manifest['plan']['sha256']), (report_path, manifest['execution_report']['sha256'])])
    return dict(manifest=manifest, models=models, entries=entries, dataset=dataset, items=items,
        common=common, anchors=anchor_map, anchor_path=anchor_path, frozen_tensors=frozen_count,
        artifacts=artifacts)


@torch.inference_mode()
def hard_predictions(models, clip):
    ids, embeddings, predictions = {}, {}, {}
    for phase in PHASES:
        tokenizer = models[phase].tokenizer
        _, ids[phase], _ = tokenizer.encode(clip[None].float())
        embeddings[phase] = tokenizer.vq.lookup(ids[phase])
        predictions[phase] = tokenizer.decode_ids(ids[phase])[0]
        if not torch.isfinite(predictions[phase]).all(): raise ValueError('Nonfinite FP32 reconstruction')
    if not torch.equal(ids['joint'], ids['decoder_only']): raise ValueError('Frozen encoder produced different discrete IDs')
    if not torch.equal(embeddings['joint'], embeddings['decoder_only']): raise ValueError('Frozen codebook produced different embeddings')
    return predictions, ids['joint'][0]


def usage_masks(target):
    masks = code_masks(target)
    regions = target[0]
    for name, pixels in dict(foreground=regions['colored_edge'] | regions['colored_flat'],
                              edge=regions['colored_edge'] | regions['noncolored_edge']).items():
        masks[name] = F.avg_pool2d(pixels[:, None].float(), 4, stride=4)[:, 0] >= .25
    return {name: mask for name, mask in masks.items() if not name.startswith('quadrant_')}


def histogram_summary(histogram):
    total = int(histogram.sum())
    probability = histogram.double() / max(1, total)
    return dict(counts=histogram.tolist(), assignments=total, used_codes=int((histogram > 0).sum()),
        entropy_bits=float(-(probability * probability.clamp_min(1e-30).log2()).sum()))


def comparison_sheet(out, index, item, arrays, crop):
    # Each original 128x128 panel and its crop use integer nearest-neighbor
    # magnification. Source/A/B always share identical pixel coordinates.
    sheet = Image.new('RGB', (3 * 520, 300), 'white'); draw = ImageDraw.Draw(sheet)
    for column, (name, array) in enumerate(zip(('input', *PHASES), arrays)):
        left = column * 520
        draw.text((left + 2, 1), f'{index:02d} {name} | frame15 | {item["source_id"]}', fill='black')
        image = Image.fromarray(array)
        sheet.paste(image.resize((256, 256), Image.Resampling.NEAREST), (left, 25))
        if crop:
            sheet.paste(image.crop(crop).resize((256, 256), Image.Resampling.NEAREST), (left + 260, 25))
        else: draw.text((left + 265, 140), 'Player crop unassessable', fill='black')
    path = out / f'comparison-{index:02d}.png'; sheet.save(path)
    return dict(path=path.name, sha256=file_hash(path), frame=15, crop=crop,
                crop_policy='Historical geometry center where available, image center otherwise; viewing aid only')


def historical_context(root, reference, common, items):
    path = bound_file(root, reference); report = read(path)
    if (report['status'] != 'completed' or report['windows'] != 64 or report['sources'] != 16 or
            (report['binding_sha256'], report['evaluation_sha256']) != common or
            [r['selection'] for r in report['records']] != items):
        raise ValueError('Historical comparator does not use these exact complete validation windows')
    arms = {a['name']: a for a in report['arms']}
    if not {'cnn32', 'st32'} <= set(arms): raise ValueError('Historical tiny32 comparator identities missing')
    return dict(path=str(path), sha256=reference['sha256'],
        provenance={name: arms[name] for name in ('cnn32', 'st32')},
        summary={name: report['summary'][name] for name in ('cnn32', 'st32')},
        note='Previously measured endpoints, not newly scored or a matched capacity/vocabulary intervention. '
             'Historical native straight-through reconstruction has tiny rounding differences from direct hard-ID lookup here.')


def run(args):
    if args.threads < 1: raise ValueError('Threads must be positive')
    torch.set_num_threads(args.threads)
    root = Path(args.root).resolve(); out = private_output(root, args.out)
    manifest_path = resolve(root, args.manifest)
    prepared = prepare(root, manifest_path, args.allow_nonproduction)
    indices = list(range(64))
    if args.check_indices is not None:
        if not args.allow_nonproduction: raise ValueError('Production evaluation cannot select a validation subset')
        indices = args.check_indices
        if not indices or len(set(indices)) != len(indices) or any(i not in range(64) for i in indices):
            raise ValueError('Invalid nonproduction check indices')
    historical = None
    if 'historical_metrics' in prepared['manifest']:
        historical = historical_context(root, prepared['manifest']['historical_metrics'], prepared['common'], prepared['items'])
    out.mkdir(parents=True)
    records, sheets = [], []
    histograms = defaultdict(lambda: torch.zeros(16, dtype=torch.int64))
    cell_counts = torch.zeros(32, 32, 16, dtype=torch.int64)
    for index in indices:
        item = prepared['items'][index]
        clip = prepared['dataset'].window(*prepared['dataset'].locate(item))
        target = prepare_target(clip)
        predictions, ids = hard_predictions(prepared['models'], clip)
        if ids.shape != (16, 32, 32): raise ValueError('Expected the actual 32x32 hard-code grid')
        for name, mask in usage_masks(target).items():
            histograms[name] += torch.bincount(ids[mask], minlength=16)
        cell_counts += F.one_hot(ids, 16).sum(0)
        source = uint8_frames(clip)
        arrays = [source[-1]]
        row = dict(index=index, selection=item, arms={}, atlases={'input': save_atlas(out, 'input', index, source)},
                   encoder_ids_exact=True, lookup_embeddings_exact=True)
        for phase in PHASES:
            row['arms'][phase] = totals(clip, predictions[phase], target[0]['neutral_flat'])
            frames = uint8_frames(predictions[phase]); arrays.append(frames[-1])
            row['atlases'][phase] = save_atlas(out, phase, index, frames)
        center = GEOMETRY.get(index, (64, 64))[:2]
        crop = (center[0]-32, center[1]-32, center[0]+32, center[1]+32) if prepared['anchors'][index]['player_barrel_assessable'] else None
        sheets.append(comparison_sheet(out, index, item, arrays, crop))
        records.append(row)
        print(json.dumps(dict(completed_windows=len(records), selected_windows=len(indices), index=index)), flush=True)
    for path, sha in prepared['artifacts']:
        if file_hash(path) != sha: raise ValueError('Artifact changed during evaluation: ' + str(path))
    summary = {phase: aggregate(records, phase) for phase in PHASES}
    cell_path = out / 'shared-code-cell-counts.npz'
    np.savez_compressed(cell_path, counts=cell_counts.numpy())
    report = dict(schema_version=1, status='nonproduction_execution_check' if args.allow_nonproduction else 'completed',
        device='cpu', precision='fp32', torch_version=str(torch.__version__), updates=0,
        windows=len(records), eligible_frozen_windows=64, sources=len({r['selection']['source_id'] for r in records}),
        frames=len(records)*16, validation_indices=indices, binding_sha256=prepared['common'][0],
        evaluation_sha256=prepared['common'][1], manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
        endpoints=prepared['entries'], frozen_encoder_codebook_tensors_exact=prepared['frozen_tensors'],
        all_evaluated_ids_embeddings_exact=True, summary=summary,
        paired_equal_source_differences=paired_differences(summary),
        shared_encoder_code_usage={name:histogram_summary(hist) for name,hist in histograms.items()},
        per_cell_code_counts=dict(path=cell_path.name, sha256=file_hash(cell_path), shape=[32,32,16],
            axis='spatial y,x,code; pooled across selected held-out frames'),
        historical_tiny32_context=historical,
        metric_policy=dict(prediction='Both decoders consume direct normalized-codebook lookup from exact shared hard IDs, CPU FP32.',
            foreground='Source-only saturation>=65/255 and value>=65/255; colored pixels, not semantic tank/food/bullet labels.',
            neutral_flat='Target-only chroma<=.04, outside dilated RGB gradients>=.08; luminance .2126/.7152/.0722.',
            code_regions='At least25% of a nominal4x4 output footprint in the target region; overlapping source-only masks, not receptive fields or semantic labels.',
            uncertainty='Paired source bootstrap2000, seed20261115; descriptive only for a one-source local check; not training-seed uncertainty.',
            display='Lossless uint8 PNGs: clip[0,1], round RGB*255; nearest-neighbor integer magnification only.'),
        semantic_G1=dict(status='unmeasured', reason='Objects, barrel separation and angles require complete independent semantic scoring; image metrics and code counts cannot pass G1.'),
        quality_claim='None: nonproduction execution check' if args.allow_nonproduction else 'Diagnostics only; crispness and frozen G1 scoring remain required.',
        source_sha256={p:file_hash(root/p) for p in ('scripts/evaluate_staged_tokenizer.py',
            'scripts/evaluate_small_codebook_architecture.py','scripts/evaluate_tokenizer_quality.py',
            'scripts/diagnose_tokenizer_regression.py','genie_diep/model.py','genie_diep/transformer_tokenizer.py')},
        anchors_sha256=file_hash(prepared['anchor_path']), records=records, assessment_sheets=sheets)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    baseline = read(root / 'configs/training-baseline-v1.json')
    template = dict(status='unscored', evaluation_sha256=prepared['common'][1],
        rubric=baseline['gates']['G1'], rubric_sha256=file_hash(root / 'configs/training-baseline-v1.json'),
        instructions='Use the existing semantic rubric on frame15; preserve uncertain/crowded examples. '
        'Positive anchor tags are not absence labels. Do not infer food/bullet categories or angles from color masks. '
        'Record rare large pentagons separately. Missing scores remain unmeasured.',
        observations={phase:[dict(index=i, frame=15, source_id=prepared['items'][i]['source_id'],
            checkpoint_sha256=prepared['entries'][phase]['checkpoint_sha256'], anchor=prepared['anchors'][i],
            player_retained=None, barrel_retained=None, food_tp=None, food_fp=None, food_fn=None,
            bullet_tp=None, bullet_fp=None, bullet_fn=None, angle_error_degrees=None,
            barrel_separation_observation=None, rare_large_pentagon_observation=None, note='') for i in indices]
            for phase in PHASES})
    (out / 'scoring-template.json').write_text(json.dumps(template, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--allow-nonproduction', action='store_true')
    p.add_argument('--check-indices', type=int, nargs='+')
    run(p.parse_args())


if __name__ == '__main__': main()
