"""CPU metrics-only learning curves on exactly the original 64 validation windows.

Manifest: {"steps": [3000, 4000, 5000], "arms": [{"name": "cnn512",
"spec": "configs/...json", "checkpoints": [{"step": 3000,
"checkpoint": "runs/...pt", "expected_sha256": "..."}, ...]}, ...]}.
Exactly CNN512, CNN32 and ST32 are required, each at all three explicit steps.
This measures late changes at matched exposure; it cannot prove convergence.
"""
import argparse
import gc
import json
from pathlib import Path
import re

import torch

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.diagnose_tokenizer_regression import prepare_target
from scripts.evaluate_small_codebook_architecture import (
    FROZEN_SELECTION_SHA256, aggregate, paired_differences, private_output, resolve,
    totals, validate_endpoint,
)
from scripts.evaluate_tokenizer_quality import reconstruct


STEPS = (3000, 4000, 5000)


def validate_manifest(manifest):
    if manifest.get('steps') != list(STEPS):
        raise ValueError('Learning curve requires exactly explicit steps 3000, 4000 and 5000')
    arms = manifest['arms']
    if len(arms) != 3 or len({arm['name'] for arm in arms}) != 3:
        raise ValueError('Exactly three uniquely named comparison arms required')
    for arm in arms:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', arm['name']):
            raise ValueError('Use simple safe arm names')
        checkpoints = arm['checkpoints']
        if len(checkpoints) != 3 or sorted(item['step'] for item in checkpoints) != list(STEPS):
            raise ValueError('Each arm needs exactly one checkpoint at each required step')
        if any(not re.fullmatch(r'[a-fA-F0-9]{64}', item['expected_sha256']) for item in checkpoints):
            raise ValueError('Each checkpoint requires an explicit SHA256 fingerprint')
    return arms


def load_model(entry):
    if file_hash(entry['checkpoint']) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    state = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
    validate_endpoint(state, entry['spec'], entry['binding_sha256'], entry['evaluation_sha256'],
                      entry['spec_sha256'], expected_step=entry['step'])
    model = WorldModel(Config(**state['config'])).cpu().float().eval()
    model.load_state_dict(state['model'], strict=True); model.requires_grad_(False)
    return model, state['contract']


def prepare(root, manifest):
    entries = []; common = None; signatures = []
    for arm in validate_manifest(manifest):
        spec_path = resolve(root, arm['spec']); spec = read(spec_path)
        identity = (file_hash(resolve(root, spec['data_binding'])),
                    file_hash(resolve(root, spec['evaluation_selection'])))
        if identity[1] != FROZEN_SELECTION_SHA256 or (common is not None and common != identity):
            raise ValueError('Learning curves require the same original frozen cohort/selection')
        common = identity
        signatures.append((spec['model'].get('tokenizer_arch', 'cnn'), spec['model']['vocab']))
        for checkpoint in sorted(arm['checkpoints'], key=lambda item: item['step']):
            entry = dict(name=arm['name'], step=checkpoint['step'], spec=spec, spec_path=str(spec_path),
                         spec_sha256=file_hash(spec_path), checkpoint=str(resolve(root, checkpoint['checkpoint'])),
                         checkpoint_sha256=checkpoint['expected_sha256'].lower(), binding_sha256=identity[0],
                         evaluation_sha256=identity[1])
            # Validate every endpoint, including strict model keys/shapes, before
            # creating an output directory. Missing checkpoints never yield a
            # completed partial three-model comparison.
            model, contract = load_model(entry)
            entry['training_contract'] = contract
            entries.append(entry)
            del model
    if sorted(signatures) != [('cnn', 32), ('cnn', 512), ('st_transformer', 32)]:
        raise ValueError('Expected CNN512, CNN32 and ST32 arms')
    reference = entries[0]['spec']
    selection_path = resolve(root, reference['evaluation_selection'])
    selection = read(selection_path)
    if selection['binding_sha256'] != common[0]:
        raise ValueError('Frozen selection cohort binding mismatch')
    dataset = Cohort(root, reference['data_binding'], 'val', 16)
    items = selection['validation']
    if len(items) != 64 or len({item['source_id'] for item in items}) != 16 or set(dataset.sources) != {item['source_id'] for item in items}:
        raise ValueError('Expected all 64 frozen windows from sixteen validation sources')
    for item in items:
        dataset.locate(item)
    return entries, dataset, items, common


def run(args):
    torch.set_num_threads(args.threads)
    root = Path(args.root).resolve(); out = private_output(root, args.out)
    manifest_path = resolve(root, args.manifest)
    entries, dataset, items, common = prepare(root, read(manifest_path))
    out.mkdir(parents=True)
    curves = {}; artifacts = []; provenance = []
    for entry in entries:
        name = entry['name']; step = entry['step']
        model, _ = load_model(entry)
        records = []
        for index, item in enumerate(items):
            clip = dataset.window(*dataset.locate(item))
            neutral = prepare_target(clip)[0]['neutral_flat']
            prediction, _ = reconstruct(model.tokenizer, clip, args.frame_batch)
            records.append(dict(index=index, selection=item, arms={name: totals(clip, prediction, neutral)}))
            if (index+1) % 8 == 0:
                print(json.dumps(dict(arm=name, step=step, completed_windows=index+1, total_windows=64)), flush=True)
        summary = aggregate(records, name)
        curves.setdefault(name, {})[str(step)] = summary
        path = out/f'{name}-step-{step:06d}.json'
        path.write_text(json.dumps(dict(name=name, step=step, summary=summary, records=records),
                                   indent=2, allow_nan=False)+'\n', encoding='utf-8')
        artifacts.append(dict(name=name, step=step, path=path.name, sha256=file_hash(path)))
        provenance.append({key: value for key, value in entry.items() if key != 'spec'})
        del model, prediction, clip, records
        gc.collect()
    changes = {name: paired_differences({step: summaries[step] for step in map(str, STEPS)})
               for name, summaries in curves.items()}
    comparisons = {str(step): paired_differences({name: summaries[str(step)] for name, summaries in curves.items()})
                   for step in STEPS}
    source_paths = ['scripts/evaluate_small_codebook_learning_curve.py', 'scripts/evaluate_small_codebook_architecture.py',
                    'scripts/evaluate_tokenizer_quality.py', 'scripts/diagnose_tokenizer_regression.py',
                    'genie_diep/model.py', 'genie_diep/transformer_tokenizer.py', 'genie_diep/cohort.py']
    report = dict(schema_version=1, status='completed', device='cpu', precision='fp32', torch_version=str(torch.__version__),
                  steps=list(STEPS), windows=64, sources=16, frames_per_window=16,
                  binding_sha256=common[0], evaluation_sha256=common[1],
                  manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                  source_hashes={path: file_hash(root/path) for path in source_paths}, endpoints=provenance,
                  curves=curves, paired_late_changes=changes, paired_model_differences_at_step=comparisons,
                  checkpoint_metrics=artifacts,
                  metric_output='Unclipped CPU FP32 predictions, exactly sixteen frames per window; no media emitted.',
                  neutral_flat='Target-only chroma<=0.04 and no dilated RGB gradient>=0.08; luminance weights .2126/.7152/.0722',
                  uncertainty='Paired source bootstrap, 2000 replicas, seed20261115; single training seed, not seed uncertainty.',
                  interpretation='Matched 5000-update exposure is not proven convergence. Late metric changes do not count objects or demonstrate semantic G1.',
                  semantic_G1='unmeasured')
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
    if min(args.threads, args.frame_batch) < 1:
        parser.error('threads and frame-batch must be positive')
    run(args)


if __name__ == '__main__':
    main()
