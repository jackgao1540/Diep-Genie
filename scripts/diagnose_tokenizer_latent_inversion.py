"""Fit continuous inputs to frozen tokenizer decoders on two fixed validation anchors."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from scripts.diagnose_tokenizer_oracle import ARMS, crop, metrics, objective, pixel_roi, roi_counts, support, tensor_hash
from scripts.diagnose_tokenizer_regression import GEOMETRY, geometry_grid, geometry_profile, geometry_sheet, load_models
from scripts.evaluate_tokenizer_quality import native_frame, uint8_frames


ANCHORS = (36, 38)
STARTS = (('q', False), ('z', False), ('q', True), ('z', True))
STEPS = 150
LEARNING_RATE = .03
SEARCH_SECONDS = 120


def model_hashes(models):
    return {name: {key: tensor_hash(value) for key, value in model.state_dict().items()}
            for name, model in models.items()}


class LocalInput:
    def __init__(self, grid, roi):
        self.grid = grid
        height, width = roi.shape
        sites = [(y, x) for y in range(grid.shape[-2]) for x in range(grid.shape[-1])
                 if crop(roi, support(y, x, height, width)).any()]
        self.y, self.x = torch.tensor(sites).T
        self.top = max(0, int(self.y.min())-3)
        self.left = max(0, int(self.x.min())-3)
        bottom = min(grid.shape[-2], int(self.y.max())+4)
        right = min(grid.shape[-1], int(self.x.max())+4)
        self.pixels = (4*self.top, 4*bottom, 4*self.left, 4*right)
        self.patch = crop(grid, (self.top, bottom, self.left, right)).contiguous()
        if crop(roi, self.pixels).sum() != roi.sum():
            raise ValueError('Local decoder patch does not enclose the objective')

    def insert(self, values, local=False):
        if local:
            grid = self.patch.clone()
            grid[:, self.y-self.top, self.x-self.left] = values.T
        else:
            grid = self.grid.clone()
            grid[:, self.y, self.x] = values.T
        return grid


def nearest_codes(values, book):
    scores = F.normalize(values, dim=-1)@book.T
    similarity, ids = scores.max(-1)
    return book[ids], ids, similarity


class Inversion:
    @torch.no_grad()
    def __init__(self, model, clip, coordinates, project_nearest=False):
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError('Inversion requires a frozen model')
        self.decoder = model.decoder
        self.book = F.normalize(model.vq.embedding.weight, dim=-1) if project_nearest else None
        self.target = clip[-1]
        self.roi = pixel_roi(clip, coordinates)
        self.ring, self.sector = geometry_grid(clip.shape, coordinates)
        features = model.features(clip[None])
        z = F.normalize(features[0, -1], dim=-1)
        q = model.vq(features)[0][0, -1]
        self.local = LocalInput(q.permute(2, 0, 1).contiguous(), self.roi)
        self.starts = {name: value[self.local.y, self.local.x].clone() for name, value in [('q', q), ('z', z)]}
        self.target_patch = crop(self.target, self.local.pixels)
        self.roi_patch = crop(self.roi, self.local.pixels)
        self.initial = self.decoder(self.local.grid[None])[0]
        self.counts = roi_counts(self.roi)
        self.results, self.images, self.grids = {}, {}, {}
        self.projection_images, self.projection_ids = {}, {}

    def cost(self, values):
        pred = self.decoder(self.local.insert(values, local=True)[None])[0]
        return objective(pred, self.target_patch, self.roi_patch, self.counts)

    @torch.no_grad()
    def project(self, name, best):
        nearest, ids, similarity = nearest_codes(best, self.book)
        prediction = self.decoder(self.local.insert(nearest)[None])[0]
        self.projection_images[name] = uint8_frames(prediction[None])[0]
        self.projection_ids[name] = ids.numpy()
        self.results[name]['projection'] = dict(
            objective=float(objective(prediction, self.target, self.roi, self.counts)),
            disk=metrics(prediction, self.target, self.roi),
            annulus=metrics(prediction, self.target, self.ring),
            mean_nearest_cosine=float(similarity.mean()),
            profile=geometry_profile(prediction, self.ring, self.sector),
        )

    def optimize(self, start, unit, deadline):
        parameter = self.starts[start].clone().requires_grad_()
        optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE)
        best = best_loss = None
        updates = 0
        while True:
            values = F.normalize(parameter, dim=-1) if unit else parameter
            loss = self.cost(values)
            score = float(loss.detach())
            if not np.isfinite(score):
                raise ValueError('Nonfinite latent inversion objective')
            if best_loss is None:
                initial_loss = score
            if best_loss is None or score < best_loss:
                best, best_loss, best_step = values.detach().clone(), score, updates
            if updates >= STEPS or perf_counter() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1

        with torch.no_grad():
            name = f'{"unit" if unit else "free"}_{start}'
            grid = self.local.insert(best)
            prediction = self.decoder(grid[None])[0]
            patch_prediction = self.decoder(self.local.insert(best, local=True)[None])[0]
            full_roi_pixels = crop(prediction, self.local.pixels)[:, self.roi_patch]
            local_roi_pixels = patch_prediction[:, self.roi_patch]
            torch.testing.assert_close(full_roi_pixels, local_roi_pixels, atol=3e-6, rtol=0.)
            norms = best.norm(dim=-1)
            self.results[name] = dict(updates=updates, best_step=best_step, status='complete' if updates == STEPS else 'partial',
                                      start_objective=initial_loss, best_objective=best_loss,
                                      disk=metrics(prediction, self.target, self.roi),
                                      annulus=metrics(prediction, self.target, self.ring),
                                      norm_min=float(norms.min()), norm_median=float(norms.median()), norm_max=float(norms.max()),
                                      local_full_max_abs_error=float((full_roi_pixels-local_roi_pixels).abs().max()),
                                      profile=geometry_profile(prediction, self.ring, self.sector))
            self.images[name] = uint8_frames(prediction[None])[0]
            self.grids[name] = grid.numpy()
            if self.book is not None:
                self.project(name, best)


def run(args):
    torch.set_num_threads(args.threads)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    spec = read(root/'configs/training-baseline-v1.json')
    binding = file_hash(root/spec['data_binding'])
    selection_path = root/spec['evaluation_selection']
    selection = read(selection_path)
    if selection['binding_sha256'] != binding or len(selection['validation']) != 64:
        raise ValueError('Unexpected frozen validation selection')
    selection_hash = file_hash(selection_path)
    models, receipts = load_models(root, spec, binding, selection_hash)
    models = {arm: models[arm] for arm in ARMS}
    checkpoint_hashes = {arm: receipts[arm] for arm in ARMS}
    before = model_hashes(models)
    cohort = Cohort(root, spec['data_binding'], 'val', spec['model']['context'])
    sources = {}
    for recipe in read(root/'configs/longform-cohort-v6.json')['source_recipes']:
        sources.update({source['id']: source for source in read(root/recipe)['sources']})
    caps, verified, cases, native_frames = {}, set(), {}, {}
    try:
        for anchor in ANCHORS:
            item = selection['validation'][anchor]
            clip = cohort.window(*cohort.locate(item))
            native_frames[anchor] = native_frame(cohort, item, root, sources, caps, verified)
            if native_frames[anchor] is None:
                raise ValueError(f'Native frame unavailable at anchor {anchor}')
            for arm in ARMS:
                cases[anchor, arm] = Inversion(models[arm], clip, GEOMETRY[anchor], args.project_nearest)
    finally:
        for capture in caps.values():
            capture.release()

    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started+SEARCH_SECONDS
    for start, unit in STARTS:
        for (anchor, arm), case in cases.items():
            if perf_counter() >= deadline:
                break
            case.optimize(start, unit, deadline)
            print(json.dumps(dict(anchor=anchor, arm=arm, start=start, unit=unit,
                                  seconds=round(perf_counter()-started, 2))), flush=True)
    seconds = perf_counter()-started
    after = model_hashes(models)
    if after != before or any(file_hash(root/item['path']) != item['sha256'] for item in checkpoint_hashes.values()):
        raise ValueError('A frozen model parameter or checkpoint changed')

    complete = all(len(case.results) == len(STARTS) and all(r['status'] == 'complete' for r in case.results.values())
                   for case in cases.values())
    report = dict(schema_version=1, status='complete' if complete else 'search_cap_partial',
                  selection_sha256=selection_hash, binding_sha256=binding, script_sha256=file_hash(__file__),
                  checkpoint_hashes=checkpoint_hashes, model_state_hashes_before=before, model_state_hashes_after=after,
                  model_updates=0, steps_per_start=STEPS, learning_rate=LEARNING_RATE, cpu_threads=args.threads,
                  torch_version=str(torch.__version__), search_seconds=seconds, search_cap_seconds=SEARCH_SECONDS,
                  objective='Tank-disk mean RGB squared residual plus mean squared residual of signed luminance first differences within the disk',
                  policy='Fit only ROI-overlap sites. Continuous unit-norm and free inputs are both outside the discrete assignment contract.',
                  projection_policy=('At fitted sites only, replace continuous vectors by their nearest unit code by cosine similarity. '
                                     'Decode without further optimization.' if args.project_nearest else None),
                  requested_starts=[dict(start=start, unit=unit) for start, unit in STARTS], cases={})
    for (anchor, arm), case in cases.items():
        basename = f'anchor-{anchor:02d}-{arm}'
        sheet, latents = f'{basename}.png', f'{basename}-latents.npz'
        original = uint8_frames(case.initial[None])[0]
        geometry_sheet(out/sheet, uint8_frames(case.target[None])[0], native_frames[anchor],
                       {'original': original, **case.images}, GEOMETRY[anchor])
        np.savez_compressed(out/latents, initial_q=case.local.grid.numpy(), **case.grids)
        report['cases'][basename] = dict(sheet=sheet, sheet_sha256=file_hash(out/sheet), latents=latents,
                                         latents_sha256=file_hash(out/latents), fitted_sites=len(case.local.y),
                                         native=dict(disk=metrics(case.initial, case.target, case.roi),
                                                     annulus=metrics(case.initial, case.target, case.ring)),
                                         results=case.results)
        if case.projection_images:
            projection_sheet, projection_data = f'{basename}-projected.png', f'{basename}-projected.npz'
            geometry_sheet(out/projection_sheet, uint8_frames(case.target[None])[0], native_frames[anchor],
                           {'original': original, **case.projection_images}, GEOMETRY[anchor])
            np.savez_compressed(out/projection_data, y=case.local.y.numpy(), x=case.local.x.numpy(),
                                **{f'ids_{name}': value for name, value in case.projection_ids.items()},
                                **{f'rgb_{name}': value for name, value in case.projection_images.items()})
            report['cases'][basename].update(projection_sheet=projection_sheet,
                                             projection_sheet_sha256=file_hash(out/projection_sheet),
                                             projection_data=projection_data,
                                             projection_data_sha256=file_hash(out/projection_data))
    (out/'inversion.json').write_text(json.dumps(report, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], out=str(out))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--project-nearest', action='store_true')
    run(parser.parse_args())
