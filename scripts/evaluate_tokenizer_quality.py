"""Read-only CPU comparison on the frozen validation set; no promotion claims.

Reconstruction metrics are diagnostics, not semantic food/bullet/angle labels.
Temporal chunking retains the encoder's six-frame causal receptive history.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from genie_diep import model as model_module


def normalized_config(config):
    return json.loads(json.dumps(config, sort_keys=True))


def validate_state(state, spec, binding_hash, selection_hash):
    contract = state['contract']
    if contract['stage'] != 'tokenizer':
        raise ValueError('Quality comparison requires tokenizer checkpoints')
    if contract['binding_sha256'] != binding_hash:
        raise ValueError('Checkpoint cohort binding mismatch')
    if contract['evaluation_sha256'] != selection_hash:
        raise ValueError('Checkpoint evaluation selection mismatch')
    if normalized_config(state['config']) != normalized_config(spec['model']):
        raise ValueError('Checkpoint architecture mismatch')
    if any(not torch.isfinite(t).all() for t in state['model'].values() if t.is_floating_point()):
        raise ValueError('Checkpoint contains nonfinite model state')


@torch.inference_mode()
def reconstruct(tokenizer, clip, frame_batch=4):
    """Preserve all causal context while bounding decoded frame batches."""
    if frame_batch < 1 or clip.ndim != 4:
        raise ValueError('Invalid reconstruction batch')
    if tokenizer.training:
        raise ValueError('Inference requires eval mode')
    # Three kernel-3 causal convolutions: six prior frames, irrespective of
    # spatial downsampling. Assert this architecture instead of silently
    # producing different predictions if the encoder changes in the future.
    convolutions = [m for m in tokenizer.encoder.modules() if isinstance(m, torch.nn.Conv3d)]
    if len(convolutions) != 3 or any(m.kernel_size[0] != 3 or m.stride[0] != 1 or m.dilation[0] != 1 for m in convolutions):
        raise ValueError('Unrecognized temporal receptive field')
    predictions = []; codes = []
    for start in range(0, len(clip), frame_batch):
        end = min(start + frame_batch, len(clip))
        left = max(0, start - 6)
        q, ids, _ = tokenizer.encode(clip[None, left:end])
        predictions.append(tokenizer.decode(q[:, start-left:])[0])
        codes.append(ids[0, start-left:])
    prediction = torch.cat(predictions)
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite reconstruction')
    return prediction, torch.cat(codes)


def metric_totals(source, prediction):
    """Fixed source-only color mask; does not identify semantic objects."""
    error = (prediction.double() - source.double()).square()
    # HSV-style saturation >=65/255 and value >=65/255, computed in FP32.
    high = source.amax(1, keepdim=True); low = source.amin(1, keepdim=True)
    foreground = (high >= 65/255) & ((high-low)/high.clamp_min(1e-8) >= 65/255)
    foreground = foreground.expand_as(error)
    temporal = ((prediction[1:].double()-prediction[:-1].double()) -
                (source[1:].double()-source[:-1].double())).square()
    return dict(mse_sum=float(error.sum()), mse_count=error.numel(),
                foreground_mse_sum=float(error[foreground].sum()), foreground_mse_count=int(foreground.sum()),
                temporal_difference_mse_sum=float(temporal.sum()), temporal_difference_mse_count=temporal.numel())


METRICS = ('mse', 'foreground_mse', 'temporal_difference_mse')


def metric_means(totals):
    return {name: totals[name+'_sum']/totals[name+'_count'] if totals[name+'_count'] else None for name in METRICS}


def aggregate(records, label):
    def pooled(items):
        return metric_means({key: sum(row[label][key] for row in items) for key in items[0][label]})
    groups = defaultdict(list)
    for row in records:
        groups[row['selection']['source_id']].append(row)
    per_source = {source: pooled(rows) for source, rows in groups.items()}
    equal = {key: float(np.mean([r[key] for r in per_source.values() if r[key] is not None]))
             if any(r[key] is not None for r in per_source.values()) else None for key in METRICS}
    return dict(pooled=pooled(records), equal_source=equal, per_source=per_source)


def uint8_frames(tensor):
    return tensor.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def native_frame(dataset, item, root, source_map, caps, verified):
    sequence, start = dataset.locate(item); offset = start + item['length'] - 1
    for key in dataset.sequences[sequence]['keys']:
        row = dataset.rows[key]
        if offset < row['frames']:
            break
        offset -= row['frames']
    source = source_map.get(row['source_id'])
    if source is None:
        return None
    path = Path(source['media_path'])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return None
    if row['source_id'] not in verified:
        if file_hash(path) != row['video_sha256']:
            raise ValueError('Native source hash mismatch')
        verified.add(row['source_id'])
        caps[row['source_id']] = cv2.VideoCapture(str(path))
    cap = caps[row['source_id']]
    cap.set(cv2.CAP_PROP_POS_FRAMES, row['source_frame_indices'][offset]); ok, frame = cap.read()
    if not ok:
        raise ValueError('Native frame decoding failed')
    x, y, w, h = row['crop']
    native = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    if not np.array_equal(cv2.resize(native, (128, 128), interpolation=cv2.INTER_AREA), dataset._frames(key)[offset]):
        raise ValueError('Native/proxy fidelity differs')
    return native


def write_sheets(out, visuals, labels):
    files = []
    for page in range((len(visuals)+7)//8):
        sheet = Image.new('RGB', (1536, 8*220), 'white'); draw = ImageDraw.Draw(sheet)
        for position, (index, item, images) in enumerate(visuals[page*8:(page+1)*8]):
            top = position*220
            draw.text((4, top+2), f"{index:02d}: {item['source_id']} | final frame {item['length']-1}", fill='black')
            for col, (label, array) in enumerate(zip(['Native crop', '128px input', *labels], images)):
                left = col*384
                draw.text((left+4, top+18), label, fill='black')
                if array is not None:
                    image = Image.fromarray(array)
                    sheet.paste(image.resize((184, 184), Image.Resampling.NEAREST), (left, top+34))
                    # A central crop is only a viewing aid, never a player label.
                    h, w = array.shape[:2]; crop = image.crop((w//4,h//4,3*w//4,3*h//4))
                    sheet.paste(crop.resize((184,184), Image.Resampling.NEAREST), (left+192,top+34))
                else:
                    draw.text((left+4,top+40), 'Unavailable', fill='black')
        path = out/f'comparison-{page:02}.png'; sheet.save(path)
        files.append(dict(path=path.name, sha256=file_hash(path)))
    return files


def run(args):
    torch.set_num_threads(2)
    root = Path(args.root).resolve(); out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Choose a new quality output directory')
    spec = read(root/args.spec); binding_hash = file_hash(root/spec['data_binding'])
    selection_path = root/spec['evaluation_selection']; selection_hash = file_hash(selection_path)
    selection = read(selection_path)
    if selection['binding_sha256'] != binding_hash:
        raise ValueError('Evaluation cohort binding mismatch')
    checkpoints = [Path(args.checkpoint_before).resolve(), Path(args.checkpoint_after).resolve()]
    checkpoint_hashes = [file_hash(path) for path in checkpoints]
    states = [torch.load(path, map_location='cpu', weights_only=True) for path in checkpoints]
    models = []
    for state in states:
        validate_state(state, spec, binding_hash, selection_hash)
        model = WorldModel(Config(**state['config'])).cpu().float().eval()
        model.load_state_dict(state['model']); model.requires_grad_(False); models.append(model)
    dataset = Cohort(root, spec['data_binding'], 'val', spec['model']['context'])
    items = selection['validation']
    if len(items) != 64 or set(row['source_id'] for row in items) != set(dataset.sources):
        raise ValueError('Expected complete frozen 64-window validation selection')
    for item in items:
        dataset.locate(item)
    anchors_path = root/'configs/baseline-fidelity-review-v1.json'
    anchors = read(anchors_path)
    if anchors['evaluation_sha256'] != selection_hash or any(row['selection'] != items[row['index']] for row in anchors['manual_anchors']):
        raise ValueError('Fidelity anchor binding mismatch')
    source_map = {}
    source_config = root/'configs/longform-cohort-v6.json'
    if source_config.is_file():
        for recipe in read(source_config)['source_recipes']:
            source_map.update({source['id']: source for source in read(root/recipe)['sources']})
    out.mkdir(parents=True)
    records = []; visuals = []; caps = {}; verified = set()
    histograms = [torch.zeros(spec['model']['vocab'], dtype=torch.int64) for _ in models]
    try:
        for index, item in enumerate(items):
            clip = dataset.window(*dataset.locate(item)); predictions = []
            row = dict(index=index, selection=item)
            for j, (label, model) in enumerate(zip(('before', 'after'), models)):
                prediction, ids = reconstruct(model.tokenizer, clip, args.frame_batch)
                row[label] = metric_totals(clip, prediction)
                histograms[j] += torch.bincount(ids.flatten(), minlength=len(histograms[j]))
                predictions.append(uint8_frames(prediction))
            source = uint8_frames(clip)
            path = out/f'reconstruction-{index:03}.npz'
            np.savez_compressed(path, source=source, before=predictions[0], after=predictions[1])
            row['artifact'] = dict(path=path.name, sha256=file_hash(path), dtype='uint8', layout='THWC')
            native = native_frame(dataset,item,root,source_map,caps,verified)
            row['native_pixel_exact'] = native is not None
            visuals.append((index,item,[native,source[-1],predictions[0][-1],predictions[1][-1]]))
            records.append(row)
            print(json.dumps(dict(completed=index+1, total=len(items))), flush=True)
    finally:
        for cap in caps.values():
            cap.release()
    if [file_hash(path) for path in checkpoints] != checkpoint_hashes:
        raise ValueError('Checkpoint file changed during evaluation')
    labels = [getattr(args, 'label_'+name, None) or f"Step {state['step']} reconstruction"
              for name, state in zip(('before', 'after'), states)]
    sheets = write_sheets(out,visuals,labels)
    report = dict(schema_version=1, device='cpu', precision='fp32', updates=0,
        torch_version=str(torch.__version__),
        model_implementation_sha256=hashlib.sha256(Path(model_module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        binding_sha256=binding_hash, selection_sha256=selection_hash,
        evaluator_sha256=file_hash(__file__), anchor_sha256=file_hash(anchors_path),
        checkpoints={label:dict(path=str(path),sha256=sha,step=state['step'],preflight=state['contract']['preflight'])
                     for label,path,sha,state in zip(('before','after'),checkpoints,checkpoint_hashes,states)},
        frames=len(items)*spec['model']['context'], windows=len(items), sources=len(dataset.sources),
        metric_policy=dict(foreground='Source-only HSV-style saturation>=65/255 and value>=65/255; colored pixels, not semantic objects',
            temporal='MSE between successive-frame differences in reconstruction and source; reconstruction consistency, not predictive rollout',
            aggregation='Pooled sums/counts and equal-weight source means; sources/windows are correlated',
            quantization='Metrics use float predictions; saved visual arrays are rounded uint8',
            sheet_frame='Final frame at offset 15 matches frozen source-fidelity anchors; central crops are viewing aids'),
        metrics={label:aggregate(records,label) for label in ('before','after')},
        code_occupancy={label:dict(used=int((hist>0).sum()),capacity=len(hist),counts=hist.tolist()) for label,hist in zip(('before','after'),histograms)},
        comparison_labels=dict(zip(('before', 'after'), labels)),
        G1=dict(status='unmeasured',reason='No complete semantic object TP/FP/FN and angle scoring' +
                ('; diagnostic preflight weights cannot be promoted' if any(state['contract']['preflight'] for state in states) else '')),
        G2=dict(status='unmeasured',reason='Tokenizer reconstruction does not measure independent control'),
        G3=dict(status='unmeasured',reason='Tokenizer reconstruction does not measure future-free dynamics'),
        native_sources_verified=len(verified),sheets=sheets,records=records)
    (out/'quality-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    template = dict(checkpoint_sha256=checkpoint_hashes[1],evaluation_sha256=selection_hash,
        status='unscored',instructions='Record independent visual observations; do not convert color probes into food/bullet labels. Bind each observation to index, frame offset15 and checkpoint. Preserve uncertain cases.',
        observations=[dict(index=a['index'],frame=15,source_id=a['selection']['source_id'],anchor=a,
            player_retained=None,barrel_retained=None,food_tp=None,food_fp=None,food_fn=None,
            bullet_tp=None,bullet_fp=None,bullet_fn=None,angle_error_degrees=None,note='') for a in anchors['manual_anchors']])
    (out/'scoring-template.json').write_text(json.dumps(template,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='diagnostics_complete',G1='unmeasured',metrics=report['metrics'])),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--spec', default='configs/training-baseline-v1.json')
    parser.add_argument('--checkpoint-before', required=True)
    parser.add_argument('--checkpoint-after', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--frame-batch', type=int, default=4)
    parser.add_argument('--label-before')
    parser.add_argument('--label-after')
    run(parser.parse_args())
