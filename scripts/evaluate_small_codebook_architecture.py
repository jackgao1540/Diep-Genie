"""CPU-only frozen-validation comparison; image metrics never establish G1.

Manifest: {"arms": [{"name": "cnn512", "checkpoint": "runs/...pt",
"spec": "configs/...json", "expected_sha256": "..."}, ...]}.
Supply exactly CNN512, CNN32 and matched ST32 endpoints at update 5000.
All media are private, pixel-derived viewing aids under a fresh runs directory.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import torch

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.diagnose_tokenizer_regression import GEOMETRY, prepare_target
from scripts.evaluate_tokenizer_quality import metric_totals, reconstruct, uint8_frames, validate_state


FROZEN_SELECTION_SHA256 = 'a7bf97bed304c7effc0095b5d81465b9c271129ba78317721d197a855ce1945d'
METRICS = ('mse', 'foreground_mse', 'temporal_difference_mse', 'neutral_flat_luminance_bias')


def resolve(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root/path).resolve()


def private_output(root, value):
    out = resolve(root, value)
    if not out.is_relative_to((root/'runs').resolve()) or out == (root/'runs').resolve():
        raise ValueError('Output must be a private subdirectory of project runs')
    if out.exists():
        raise FileExistsError('Choose a fresh evaluation output directory')
    return out


def validate_endpoint(state, spec, binding_hash, selection_hash, spec_hash):
    validate_state(state, spec, binding_hash, selection_hash)
    contract = state['contract']
    if state['step'] != 5000 or contract.get('preflight') is not False:
        raise ValueError('Expected a production update-5000 endpoint')
    if contract.get('spec_sha256') != spec_hash:
        raise ValueError('Checkpoint specification hash mismatch')
    if contract.get('precision') != 'fp16' or contract.get('device_type') != 'cuda':
        raise ValueError('Expected production CUDA FP16 training provenance')
    if contract.get('microbatch') != 4 or contract.get('accumulation') != 4:
        raise ValueError('Expected matching four-by-four training microbatches')
    if spec['model']['size'] != 128 or spec['model']['context'] != 16:
        raise ValueError('Expected frozen 128px, sixteen-frame setup')


def totals(source, prediction, neutral_flat):
    result = metric_totals(source, prediction)
    weights = source.new_tensor([.2126, .7152, .0722]).double()[None, :, None, None]
    residual = ((prediction.double()-source.double())*weights).sum(1)
    result['neutral_flat_luminance_bias_sum'] = float(residual[neutral_flat].sum())
    result['neutral_flat_luminance_bias_count'] = int(neutral_flat.sum())
    return result


def means(items):
    return {key: sum(row[key+'_sum'] for row in items)/sum(row[key+'_count'] for row in items)
            if sum(row[key+'_count'] for row in items) else None for key in METRICS}


def aggregate(records, name):
    groups = defaultdict(list)
    for row in records:
        groups[row['selection']['source_id']].append(row['arms'][name])
    sources = {key: means(rows) for key, rows in sorted(groups.items())}
    equal = {key: float(np.mean([row[key] for row in sources.values() if row[key] is not None]))
             if any(row[key] is not None for row in sources.values()) else None for key in METRICS}
    return dict(pooled=means([row['arms'][name] for row in records]), equal_source=equal, per_source=sources)


def paired_differences(summary, replicas=2000):
    result = {}
    for first, second in itertools.combinations(summary, 2):
        entry = {}
        for metric in METRICS:
            sources = sorted(set(summary[first]['per_source']) & set(summary[second]['per_source']))
            delta = np.array([summary[second]['per_source'][s][metric]-summary[first]['per_source'][s][metric]
                              for s in sources if summary[first]['per_source'][s][metric] is not None
                              and summary[second]['per_source'][s][metric] is not None])
            if not len(delta):
                entry[metric] = None
                continue
            rng = np.random.default_rng(20261115)
            bootstrap = delta[rng.integers(len(delta), size=(replicas, len(delta)))].mean(1)
            entry[metric] = dict(mean=float(delta.mean()), source_count=len(delta),
                                 source_bootstrap_95_percent=np.quantile(bootstrap, [.025, .975]).tolist())
        result[second+' minus '+first] = entry
    return result


def save_atlas(out, name, index, frames):
    atlas = np.concatenate([np.concatenate(list(frames[row*4:row*4+4]), axis=1) for row in range(4)], axis=0)
    path = out/f'{name}-{index:02d}.png'
    Image.fromarray(atlas).save(path)
    return dict(path=path.name, sha256=file_hash(path), layout='4x4 row-major', frames=16,
                frame_size=[128, 128], dtype='uint8', display='clamp [0,1], round RGB*255')


def contact_sheet(out, page, visuals, names):
    sheet = Image.new('RGB', (1536, 8*220), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (index, item, arrays, crop_box) in enumerate(visuals):
        top = row*220
        draw.text((4, top+1), f"{index:02d} {item['source_id']} | frame 15 | center crop is a viewing aid, not a player annotation", fill='black')
        for column, (label, array) in enumerate(zip(['target', *names], arrays)):
            left = column*384
            draw.text((left+3, top+17), label+' | full / fixed crop', fill='black')
            image = Image.fromarray(array)
            sheet.paste(image.resize((184, 184), Image.Resampling.NEAREST), (left, top+34))
            if crop_box is not None:
                sheet.paste(image.crop(crop_box).resize((184, 184), Image.Resampling.NEAREST), (left+192, top+34))
            else:
                draw.text((left+196, top+80), 'Player crop unassessable', fill='black')
    path = out/f'assessment-{page:02d}.png'
    sheet.save(path)
    return dict(path=path.name, sha256=file_hash(path), assessment_frame=15)


def run(args):
    torch.set_num_threads(args.threads)
    root = Path(args.root).resolve()
    out = private_output(root, args.out)
    manifest_path = resolve(root, args.manifest)
    manifest = read(manifest_path)
    arms = manifest['arms']
    if len(arms) != 3 or len({arm['name'] for arm in arms}) != 3:
        raise ValueError('Exactly three uniquely named comparison arms required')
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', arm['name']) or arm['name'] == 'target' for arm in arms):
        raise ValueError('Use simple safe arm names')
    entries = []
    common = None
    for arm in arms:
        spec_path = resolve(root, arm['spec']); spec = read(spec_path)
        binding_path = resolve(root, spec['data_binding'])
        selection_path = resolve(root, spec['evaluation_selection'])
        identity = (file_hash(binding_path), file_hash(selection_path))
        if identity[1] != FROZEN_SELECTION_SHA256 or (common is not None and identity != common):
            raise ValueError('Comparison must use the same original frozen validation binding/selection')
        common = identity
        path = resolve(root, arm['checkpoint'])
        sha = file_hash(path)
        if sha != arm['expected_sha256']:
            raise ValueError('Checkpoint SHA256 mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_endpoint(state, spec, *identity, file_hash(spec_path))
        model = WorldModel(Config(**state['config'])).cpu().float().eval()
        model.load_state_dict(state['model'], strict=True); model.requires_grad_(False)
        entries.append(dict(name=arm['name'], model=model, spec=spec, provenance=dict(
            checkpoint=str(path), checkpoint_sha256=sha, spec=str(spec_path), spec_sha256=file_hash(spec_path),
            training_contract=state['contract'], step=state['step'])))
        del state
    signatures = sorted((e['spec']['model'].get('tokenizer_arch', 'cnn'), e['spec']['model']['vocab']) for e in entries)
    if signatures != [('cnn', 32), ('cnn', 512), ('st_transformer', 32)]:
        raise ValueError('Expected CNN512, CNN32 and ST32 arms')
    entries.sort(key=lambda e: {'cnn': 0, 'st_transformer': 1}[e['spec']['model'].get('tokenizer_arch', 'cnn')]*1000-e['spec']['model']['vocab'])
    selection = read(selection_path)
    if selection['binding_sha256'] != common[0]:
        raise ValueError('Frozen selection cohort binding mismatch')
    dataset = Cohort(root, entries[0]['spec']['data_binding'], 'val', 16)
    items = selection['validation']
    if len(items) != 64 or len({item['source_id'] for item in items}) != 16 or set(dataset.sources) != {item['source_id'] for item in items}:
        raise ValueError('Expected all 64 frozen windows from sixteen validation sources')
    for item in items:
        dataset.locate(item)
    anchors_path = root/'configs/baseline-fidelity-review-v1.json'
    anchors = read(anchors_path)
    if anchors['evaluation_sha256'] != common[1]:
        raise ValueError('Fidelity assessment selection mismatch')
    anchor_map = {row['index']: row for row in anchors['manual_anchors']}
    if set(anchor_map) != set(range(64)) or any(anchor_map[i]['selection'] != items[i] for i in range(64)):
        raise ValueError('Fidelity assessment anchors mismatch')
    out.mkdir(parents=True)
    records = []; visuals = []; sheets = []
    counts = {entry['name']: torch.zeros(entry['spec']['model']['vocab'], dtype=torch.int64) for entry in entries}
    for index, item in enumerate(items):
        clip = dataset.window(*dataset.locate(item))
        neutral = prepare_target(clip)[0]['neutral_flat']
        source = uint8_frames(clip)
        center = GEOMETRY.get(index, (64, 64))[:2]
        crop = (center[0]-32, center[1]-32, center[0]+32, center[1]+32) if anchor_map[index]['player_barrel_assessable'] else None
        record = dict(index=index, selection=item, arms={},
                      player_barrel_assessable_anchor=anchor_map[index]['player_barrel_assessable'],
                      viewing_crop_box=crop, crop_note='Fixed historical geometry center where available, image center otherwise; viewing aid only',
                      atlases={'target': save_atlas(out, 'target', index, source)})
        arrays = [source[-1]]
        for entry in entries:
            prediction, ids = reconstruct(entry['model'].tokenizer, clip, args.frame_batch)
            name = entry['name']
            record['arms'][name] = totals(clip, prediction, neutral)
            frames = uint8_frames(prediction)
            record['atlases'][name] = save_atlas(out, name, index, frames)
            arrays.append(frames[-1]); counts[name] += torch.bincount(ids.flatten(), minlength=len(counts[name]))
        records.append(record); visuals.append((index, item, arrays, crop))
        if len(visuals) == 8:
            sheets.append(contact_sheet(out, index//8, visuals, [e['name'] for e in entries])); visuals = []
        print(json.dumps(dict(completed_windows=index+1, total_windows=64)), flush=True)
    summary = {e['name']: aggregate(records, e['name']) for e in entries}
    usage = {}
    for name, histogram in counts.items():
        probability = histogram.double()/histogram.sum()
        usage[name] = dict(counts=histogram.tolist(), used_codes=int((histogram > 0).sum()),
                           entropy_bits=float(-(probability*probability.clamp_min(1e-30).log2()).sum()))
    report = dict(schema_version=1, status='completed', device='cpu', precision='fp32',
                  windows=64, sources=16, frames_per_window=16, evaluation_sha256=common[1], binding_sha256=common[0],
                  fidelity_anchor_sha256=file_hash(anchors_path),
                  manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                  source_hashes={p: file_hash(root/p) for p in [
                      'scripts/evaluate_small_codebook_architecture.py', 'scripts/evaluate_tokenizer_quality.py',
                      'scripts/diagnose_tokenizer_regression.py', 'genie_diep/model.py', 'genie_diep/transformer_tokenizer.py']},
                  arms=[entry['provenance'] | {'name': entry['name']} for entry in entries], summary=summary,
                  paired_equal_source_differences=paired_differences(summary), code_usage=usage,
                  metric_output='unclipped FP32 predictions; atlases clamp solely for display',
                  uncertainty='Paired source bootstrap, 2000 replicas, seed20261115; single training seed, not seed uncertainty.',
                  semantic_G1='unmeasured; image errors and code usage do not count objects or barrel retention',
                  neutral_flat='Target-only chroma<=0.04 and no dilated RGB gradient>=0.08; luminance weights .2126/.7152/.0722',
                  records=records, assessment_sheets=sheets)
    (out/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--frame-batch', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
