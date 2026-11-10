"""Read-only objective scale/coverage check on frozen training-only windows."""
import argparse
import json
from pathlib import Path

import torch

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from genie_diep.detail_loss import DEFAULT_SPEC, detail_diagnostics, detail_loss
from genie_diep.model import Config, WorldModel, image_loss
from scripts.evaluate_tokenizer_quality import reconstruct, validate_state


def run(root, checkpoint, output):
    root, checkpoint, output = map(lambda p: Path(p).resolve(), (root, checkpoint, output))
    if output.exists():
        raise FileExistsError('Preserve completed probes')
    torch.set_num_threads(2)
    spec = read(root/'configs/training-baseline-v1.json')
    selection = read(root/spec['evaluation_selection'])
    before = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_state(state, spec, file_hash(root/spec['data_binding']), file_hash(root/spec['evaluation_selection']))
    model = WorldModel(Config(**state['config'])).cpu().float().eval()
    model.load_state_dict(state['model']); model.requires_grad_(False)
    dataset = Cohort(root, spec['data_binding'], 'train', 16)
    # Evenly spaced through the previously frozen training calibration list.
    calibration = selection['training_calibration']
    indices = [round(i*(len(calibration)-1)/7) for i in range(8)]
    records = []
    for index in indices:
        item = calibration[index]
        target = dataset.window(*dataset.locate(item))
        prediction, _ = reconstruct(model.tokenizer, target, 4)
        prediction = prediction.detach().clone()[None].requires_grad_(True)
        target = target[None]
        baseline = 10*image_loss(prediction, target, color_balance=True)
        detail = detail_loss(prediction, target, DEFAULT_SPEC)
        baseline_gradient = torch.autograd.grad(baseline, prediction, retain_graph=True)[0]
        detail_gradient = torch.autograd.grad(detail, prediction)[0]
        baseline_norm = baseline_gradient.norm()
        detail_norm = detail_gradient.norm()
        records.append(dict(selection=item, baseline_reconstruction_loss=float(baseline.detach()),
            detail_loss_at_weight_one=float(detail.detach()),
            output_gradient_norm_ratio=float(detail_norm/baseline_norm.clamp_min(1e-12)),
            output_gradient_cosine=float((baseline_gradient*detail_gradient).sum()/(baseline_norm*detail_norm).clamp_min(1e-12)),
            regions={key: float(value) for key, value in detail_diagnostics(target, DEFAULT_SPEC).items()
                     if value.numel() == 1}))
    assert file_hash(checkpoint) == before
    report = dict(schema_version=1, purpose='Training-only objective diagnostic; no model updates or quality claims',
        checkpoint_sha256=before, checkpoint_step=state['step'], checkpoint_unchanged=True,
        selection_sha256=file_hash(root/spec['evaluation_selection']), split='train', windows=len(records),
        sources=len({r['selection']['source_id'] for r in records}), optimizer_updates=0,
        coefficient=1.0, loss_spec=DEFAULT_SPEC, records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(dict(windows=len(records), sources=report['sources'],
        gradient_ratio=[r['output_gradient_norm_ratio'] for r in records],
        loss=[r['detail_loss_at_weight_one'] for r in records], checkpoint_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.root, args.checkpoint, args.out)
