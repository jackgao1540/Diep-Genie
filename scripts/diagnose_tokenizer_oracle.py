"""Target-informed coordinate search using frozen tokenizer decoders and codebooks."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from scripts.diagnose_tokenizer_regression import GEOMETRY, geometry_grid, geometry_profile, geometry_sheet, load_models
from scripts.evaluate_tokenizer_quality import native_frame, uint8_frames


ANCHORS = (36, 37, 38, 39)
ARMS = ('b5', 'd5')
SWEEPS = 2
SEARCH_SECONDS = 300
LUMA = (.2126, .7152, .0722)


def support(y, x, height, width):
    return max(0, 4*y-4), min(height, 4*y+8), max(0, 4*x-4), min(width, 4*x+8)


def crop(value, bounds):
    top, bottom, left, right = bounds
    return value[..., top:bottom, left:right]


def local_delta(delta, top, left, base):
    """Place a convolution delta and remove positions clipped by true layer boundaries."""
    h, w = delta.shape[-2:]
    bounds = max(0, top), min(base.shape[-2], top+h), max(0, left), min(base.shape[-1], left+w)
    a, b, c, d = bounds
    return crop(delta, (a-top, b-top, c-left, d-left)), bounds


def local_candidates(decoder, grid, site, candidates):
    """Decode a halo around a changed site, propagating only its additive differences."""
    y, x = site
    start_y, start_x = max(0, y-3), max(0, x-3)
    patch = grid[:, start_y:y+4, start_x:x+4][None].contiguous()
    local_y, local_x = y-start_y, x-start_x
    conv1, activation1, conv2, activation2, conv3, output = decoder
    base1 = conv1(patch)
    active1 = activation1(base1)
    base2 = conv2(active1)
    active2 = activation2(base2)
    base3 = conv3(active2)

    delta = candidates-grid[:, y, x]
    delta1 = F.conv_transpose2d(delta[:, :, None, None], conv1.weight, stride=2)
    delta1, area1 = local_delta(delta1, 2*local_y-1, 2*local_x-1, base1)
    delta1 = activation1(crop(base1, area1)+delta1)-crop(active1, area1)
    delta2 = F.conv_transpose2d(delta1, conv2.weight, stride=2)
    delta2, area2 = local_delta(delta2, 2*area1[0]-1, 2*area1[2]-1, base2)
    delta2 = activation2(crop(base2, area2)+delta2)-crop(active2, area2)
    delta3 = F.conv2d(delta2, conv3.weight, padding=2)
    delta3, area3 = local_delta(delta3, area2[0]-1, area2[2]-1, base3)
    bounds = tuple(offset+coord for offset, coord in zip((4*start_y, 4*start_y, 4*start_x, 4*start_x), area3))
    return output(crop(base3, area3)+delta3), bounds


def luma(value):
    return (value*value.new_tensor(LUMA).reshape(3, 1, 1)).sum(-3)


def roi_errors(prediction, target, roi):
    rgb = prediction-target
    dy = luma(rgb).diff(dim=-2)
    dx = luma(rgb).diff(dim=-1)
    mask_y = roi[..., 1:, :] & roi[..., :-1, :]
    mask_x = roi[..., :, 1:] & roi[..., :, :-1]
    rgb_sum = (rgb.square()*roi).sum(dim=(-3, -2, -1), dtype=torch.float64)
    edge_sum = ((dy.square()*mask_y).sum(dim=(-2, -1), dtype=torch.float64)
                + (dx.square()*mask_x).sum(dim=(-2, -1), dtype=torch.float64))
    return rgb_sum, edge_sum


def roi_counts(roi):
    edges = (roi[1:] & roi[:-1]).sum()+(roi[:, 1:] & roi[:, :-1]).sum()
    return 3*int(roi.sum()), int(edges)


def objective(prediction, target, roi, counts):
    rgb, edges = roi_errors(prediction, target, roi)
    return rgb/counts[0] + edges/counts[1]


def metrics(prediction, target, roi):
    rgb, edge = roi_errors(prediction, target, roi)
    rgb_count, edge_count = roi_counts(roi)
    chroma = (prediction.amax(0)-prediction.amin(0))-(target.amax(0)-target.amin(0))
    return dict(objective=float(rgb+edge), rgb_mse=float(rgb)/rgb_count,
                chroma_mse=float(chroma[roi].square().mean()), luma_edge_mse=float(edge)/edge_count,
                roi_pixels=int(roi.sum()), roi_edges=edge_count)


def pixel_roi(clip, coordinates):
    cx, cy, _, outer = coordinates
    y, x = torch.meshgrid(torch.arange(clip.shape[-2])-cy, torch.arange(clip.shape[-1])-cx, indexing='ij')
    return x.square()+y.square() <= outer**2


def verify_local(decoder, grid, codes, sites):
    height, width = grid.shape[-2]*4, grid.shape[-1]*4
    reference = decoder(grid[None])[0]
    candidates = codes[[0, len(codes)//2, len(codes)-1]]
    maximum = 0.
    for y, x in sites:
        patches, bounds = local_candidates(decoder, grid, (y, x), candidates)
        if bounds != support(y, x, height, width):
            raise ValueError(f'Decoder support differs at {y}, {x}')
        full_inputs = grid[None].expand(len(candidates), -1, -1, -1).clone()
        full_inputs[:, :, y, x] = candidates
        full_outputs = decoder(full_inputs)
        assembled = reference[None].expand_as(full_outputs).clone()
        crop(assembled, bounds).copy_(patches)
        torch.testing.assert_close(assembled, full_outputs, atol=3e-6, rtol=0.)
        maximum = max(maximum, float((assembled-full_outputs).abs().max()))
    return maximum


class Search:
    def __init__(self, model, clip, coordinates, chunk, normalized=False):
        self.decoder = model.decoder
        self.codes = F.normalize(model.vq.embedding.weight, dim=-1)
        self.target = clip[-1]
        self.roi = pixel_roi(clip, coordinates)
        self.counts = roi_counts(self.roi) if normalized else (1, 1)
        self.ring, self.sector = geometry_grid(clip.shape, coordinates)
        self.chunk = chunk
        q, ids, _ = model.encode(clip[None])
        self.grid = q[0, -1].permute(2, 0, 1).contiguous()
        self.ids = ids[0, -1].clone()
        self.initial_ids = self.ids.clone()
        self.initial = self.decoder(self.grid[None])[0]
        self.current = self.initial.clone()
        height, width = self.target.shape[-2:]
        self.sites = [(y, x) for y in range(self.grid.shape[-2]) for x in range(self.grid.shape[-1])
                      if crop(self.roi, support(y, x, height, width)).any()]
        self.sites.sort(key=lambda site: ((4*site[0]-coordinates[1])**2+(4*site[1]-coordinates[0])**2, site))
        self.sweeps = []
        self.visited = self.accepted = 0

    def visit(self, site, deadline):
        y, x = site
        height, width = self.target.shape[-2:]
        top, bottom, left, right = support(y, x, height, width)
        halo = max(0, top-1), min(height, bottom+1), max(0, left-1), min(width, right+1)
        local = top-halo[0], bottom-halo[0], left-halo[2], right-halo[2]
        target, roi, incumbent = (crop(value, halo) for value in (self.target, self.roi, self.current))
        best = float(objective(incumbent, target, roi, self.counts))
        best_id = best_patch = None
        for start in range(0, len(self.codes), self.chunk):
            if perf_counter() >= deadline:
                return False
            candidates = self.codes[start:start+self.chunk]
            patches, bounds = local_candidates(self.decoder, self.grid, site, candidates)
            if bounds != (top, bottom, left, right):
                raise ValueError('Unexpected decoder support')
            canvases = incumbent[None].expand(len(candidates), -1, -1, -1).clone()
            crop(canvases, local).copy_(patches)
            costs = objective(canvases, target, roi, self.counts)
            minimum, index = costs.min(0)
            index = int(index)
            if float(minimum) < best-1e-7:
                best, best_id, best_patch = float(minimum), start+index, patches[index].clone()
        self.visited += 1
        if best_id is not None:
            self.grid[:, y, x] = self.codes[best_id]
            self.ids[y, x] = best_id
            crop(self.current, (top, bottom, left, right)).copy_(best_patch)
            self.accepted += 1
        return True

    def reconcile(self):
        full = self.decoder(self.grid[None])[0]
        torch.testing.assert_close(self.current, full, atol=3e-6, rtol=0.)
        self.current = full

    def summary(self):
        changes = (self.ids != self.initial_ids).nonzero()
        return dict(initial=metrics(self.initial, self.target, self.roi),
                    oracle=metrics(self.current, self.target, self.roi),
                    initial_search_objective=float(objective(self.initial, self.target, self.roi, self.counts)),
                    final_search_objective=float(objective(self.current, self.target, self.roi, self.counts)),
                    initial_ring=metrics(self.initial, self.target, self.ring),
                    oracle_ring=metrics(self.current, self.target, self.ring),
                    profiles={name: geometry_profile(image, self.ring, self.sector)
                              for name, image in [('target', self.target), ('initial', self.initial), ('oracle', self.current)]},
                    candidate_codes=len(self.codes), sites_per_sweep=len(self.sites), complete_sites=self.visited,
                    accepted_updates=self.accepted, changed_sites=len(changes), sweeps=self.sweeps,
                    code_changes=[dict(y=int(y), x=int(x), old=int(self.initial_ids[y, x]), new=int(self.ids[y, x]))
                                  for y, x in changes])


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
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
    cohort = Cohort(root, spec['data_binding'], 'val', spec['model']['context'])
    native_sources = {}
    for recipe in read(root/'configs/longform-cohort-v6.json')['source_recipes']:
        native_sources.update({source['id']: source for source in read(root/recipe)['sources']})
    caps, verified, cases, native_frames = {}, set(), {}, {}
    try:
        for anchor in args.anchors:
            item = selection['validation'][anchor]
            clip = cohort.window(*cohort.locate(item))
            native_frames[anchor] = native_frame(cohort, item, root, native_sources, caps, verified)
            if native_frames[anchor] is None:
                raise ValueError(f'Native frame unavailable at anchor {anchor}')
            for arm in ARMS:
                cases[anchor, arm] = Search(models[arm], clip, GEOMETRY[anchor], args.batch, args.normalized)
    finally:
        for capture in caps.values():
            capture.release()

    book_hashes = {arm: tensor_hash(models[arm].vq.embedding.weight) for arm in ARMS}
    identity = {}
    for arm in ARMS:
        case = cases[args.anchors[0], arm]
        identity[arm] = verify_local(case.decoder, case.grid, case.codes, (case.sites[0], (0, 0), (31, 31)))
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started+args.search_seconds
    for sweep in range(SWEEPS):
        starts = {key: (case.visited, case.accepted, float(objective(case.current, case.target, case.roi, case.counts)))
                  for key, case in cases.items()}
        for site_index in range(max(len(case.sites) for case in cases.values())):
            for key, case in cases.items():
                if perf_counter() >= deadline:
                    break
                if site_index < len(case.sites):
                    site = case.sites[site_index if not sweep else -1-site_index]
                    if not case.visit(site, deadline):
                        break
            if perf_counter() >= deadline:
                break
            if (site_index+1) % 10 == 0:
                print(json.dumps(dict(sweep=sweep+1, sites_per_case=site_index+1,
                                      seconds=round(perf_counter()-started))), flush=True)
        for key, case in cases.items():
            case.reconcile()
            visited, accepted, before = starts[key]
            after = float(objective(case.current, case.target, case.roi, case.counts))
            if after > before+1e-5:
                raise ValueError('A sweep increased the full-decoder objective')
            case.sweeps.append(dict(sweep=sweep+1, sites=case.visited-visited,
                                    accepted=case.accepted-accepted, objective_before=before,
                                    objective_after=after, complete=case.visited-visited == len(case.sites)))
        if perf_counter() >= deadline:
            break
    search_seconds = min(perf_counter(), deadline)-started

    report = dict(schema_version=1, status='complete' if all(case.visited == SWEEPS*len(case.sites) for case in cases.values())
                  else 'search_cap_partial', selection_sha256=selection_hash, binding_sha256=binding,
                  script_sha256=file_hash(__file__), torch_version=str(torch.__version__), cpu_threads=args.threads,
                  candidate_batch=args.batch, search_seconds=search_seconds, search_cap_seconds=args.search_seconds,
                  model_updates=0, checkpoint_hashes={arm: receipts[arm] for arm in ARMS},
                  embedding_hashes=book_hashes, local_full_max_abs_error=identity,
                  objective=('Tank-disk mean RGB squared residual plus mean squared residual of signed luminance first differences within the disk'
                             if args.normalized else 'Sum of RGB pixel squared residuals and luminance first-difference squared residuals inside the tank disk'),
                  normalized_objective=args.normalized,
                  requested_sweeps=SWEEPS, case_order='Round-robin by anchor and arm. Center-out then reverse.', cases={})
    for arm in ARMS:
        if file_hash(root/receipts[arm]['path']) != receipts[arm]['sha256'] or tensor_hash(models[arm].vq.embedding.weight) != book_hashes[arm]:
            raise ValueError(f'Frozen checkpoint or embedding changed for {arm}')
    for anchor in args.anchors:
        paths = {}
        predictions = {}
        for arm in ARMS:
            case = cases[anchor, arm]
            paths[arm] = case.summary()
            predictions[f'{arm} original'] = uint8_frames(case.initial[None])[0]
            predictions[f'{arm} oracle'] = uint8_frames(case.current[None])[0]
        image = f'anchor-{anchor:02d}.png'
        geometry_sheet(out/image, uint8_frames(case.target[None])[0], native_frames[anchor], predictions, GEOMETRY[anchor])
        report['cases'][anchor] = dict(sheet=image, arms=paths)
    (out/'oracle.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], out=str(out))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch', type=int, choices=(64, 128), default=64)
    parser.add_argument('--anchors', type=int, nargs='+', choices=ANCHORS, default=ANCHORS)
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--search-seconds', type=int, default=SEARCH_SECONDS)
    args = parser.parse_args()
    if len(set(args.anchors)) != len(args.anchors) or not 0 < args.search_seconds <= SEARCH_SECONDS:
        parser.error('Use distinct anchors and a search cap between 1 and 300 seconds')
    run(args)
