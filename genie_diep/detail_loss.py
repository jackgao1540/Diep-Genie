"""Target-derived, region-normalized RGB reconstruction loss for a loss-only arm.

This is a structural weighting hypothesis, not an object detector. Only the
current training target selects regions; predictions cannot select or change
their own weights. No mask is an input to the tokenizer or a generated rollout.

For each B,T frame let e[y,x] = mean_RGB((prediction-target)**2). Partition the
frame into fixed nonoverlapping 8x8 tiles. U is the mean of tile means (uniform
coverage; deliberately equivalent to global pixel MSE). Let g be the maximum
absolute RGB difference to any horizontal/vertical neighbor, assigning each
contrast to both incident pixels. A seed is g>=0.08; a 3x3 dilation includes its
immediate reconstruction neighborhood. The two structural channels are:

 A: dilated achromatic seeds, restricted to target chroma (max-min RGB)<=0.12.
 S: all dilated seeds, but only tiles with 2..32 selected pixels (<=half a tile).

For each channel, average e WITHIN its selected pixels in each eligible tile,
then average eligible tiles equally in that frame. A also requires >=2 pixels;
empty channels contribute zero. Finally average frames uniformly. The default
loss is D=0.25*U+0.50*A+0.25*S, bounded by one for unit-range RGB. Absent channels
do not redistribute their weight. Thus dense scenes, large connected regions,
and batches with more selected pixels cannot increase the total region weight.
Unlike averaging unmasked equal-size tiles, A/S do not reduce to global MSE.

Selectors/constants are frozen before held-out comparison and may be diagnosed
on training footage only. Grid lines, names and compression edges can activate
them. There is no added image-gradient/frequency penalty, temporal objective,
semantic label, predicted-region selector, random sampling, or architecture
change. The experiment applies its coefficient externally to this scalar.
"""
import math

import torch
from torch.nn import functional as F


DEFAULT_SPEC = {
    'patch_size': 8,
    'gradient_threshold': 0.08,
    'achromatic_chroma_max': 0.12,
    'dilation_radius': 1,
    'min_region_pixels': 2,
    'small_region_max_fraction': 0.5,
    'uniform_weight': 0.25,
    'achromatic_weight': 0.5,
    'small_weight': 0.25,
}


def _settings(spec):
    if spec is not None and not isinstance(spec, dict):
        raise ValueError('Detail loss specification must be a mapping')
    if spec and set(spec)-set(DEFAULT_SPEC):
        raise ValueError('Unknown detail loss setting')
    result = dict(DEFAULT_SPEC)
    if spec:
        result.update(spec)
    for key in ('patch_size', 'dilation_radius', 'min_region_pixels'):
        if isinstance(result[key], bool) or not isinstance(result[key], int):
            raise ValueError('Detail loss integer setting invalid')
    if result['patch_size'] < 2 or result['dilation_radius'] not in (0, 1, 2):
        raise ValueError('Invalid detail patch size or dilation')
    if not 1 <= result['min_region_pixels'] <= result['patch_size']**2:
        raise ValueError('Invalid minimum region size')
    for key in set(result)-{'patch_size', 'dilation_radius', 'min_region_pixels'}:
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError('Detail loss finite numeric setting required')
    if not 0 < result['gradient_threshold'] <= 1 or not 0 <= result['achromatic_chroma_max'] <= 1:
        raise ValueError('Invalid detail gradient/chroma threshold')
    if not 0 < result['small_region_max_fraction'] <= 1:
        raise ValueError('Invalid small-region fraction')
    if result['small_region_max_fraction']*result['patch_size']**2 < result['min_region_pixels']:
        raise ValueError('Small-region range is empty')
    weights = [result[key] for key in ('uniform_weight', 'achromatic_weight', 'small_weight')]
    if min(weights) < 0 or not math.isclose(sum(weights), 1., rel_tol=0, abs_tol=1e-9):
        raise ValueError('Detail channel weights must be nonnegative and sum to one')
    return result


def _validate_target(target, settings):
    if target.ndim != 5 or target.shape[2] != 3 or min(target.shape) < 1 or not target.is_floating_point():
        raise ValueError('Detail loss expects floating B,T,3,H,W tensors')
    patch = settings['patch_size']
    if target.shape[-2] % patch or target.shape[-1] % patch:
        raise ValueError('Image dimensions must be divisible by patch size')


def _tiles(value, patch):
    # N,1,H,W -> N,number_of_tiles,patch_area; fixed nonoverlapping coverage.
    return F.unfold(value, kernel_size=patch, stride=patch).transpose(1, 2)


def _selectors(target, settings):
    """Return detached per-tile masks; callers disable autocast for FP32 math."""
    image = target.detach().float().flatten(0, 1)
    horizontal = (image[..., 1:]-image[..., :-1]).abs().amax(1, keepdim=True)
    vertical = (image[..., 1:, :]-image[..., :-1, :]).abs().amax(1, keepdim=True)
    gradient = torch.maximum(
        torch.maximum(F.pad(horizontal, (0, 1, 0, 0)), F.pad(horizontal, (1, 0, 0, 0))),
        torch.maximum(F.pad(vertical, (0, 0, 0, 1)), F.pad(vertical, (0, 0, 1, 0))))
    seeds = gradient >= settings['gradient_threshold']
    achromatic = image.amax(1, keepdim=True)-image.amin(1, keepdim=True) <= settings['achromatic_chroma_max']
    radius = settings['dilation_radius']
    def dilate(mask):
        return F.max_pool2d(mask.float(), 2*radius+1, stride=1, padding=radius)
    achromatic_regions = dilate(seeds & achromatic)*achromatic
    structural_regions = dilate(seeds)
    return _tiles(achromatic_regions, settings['patch_size']), _tiles(structural_regions, settings['patch_size'])


def _region_mean(error, mask, minimum, maximum=None):
    counts = mask.sum(-1)
    valid = counts >= minimum
    if maximum is not None:
        valid = valid & (counts <= maximum)
    # Zero weights for empty tiles preserve finite differentiable zero, while
    # counts do not weight valid tiles relative to one another.
    per_tile = (error*mask).sum(-1)/counts.clamp_min(1)
    per_frame = (per_tile*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return per_frame.mean(), counts, valid


def detail_loss(pred, target, spec=None):
    """Return a scalar FP32 loss, with gradients only through prediction.

Inputs are unit-range RGB; the model's sigmoid supplies this contract. The
function does not clip predictions, which would hide out-of-range errors.
The caller retains the existing finite-loss/gradient guard. No CPU transfer,
random draw or optimizer mutation occurs here.
"""
    settings = _settings(spec)
    _validate_target(target, settings)
    if pred.shape != target.shape or pred.device != target.device or not pred.is_floating_point():
        raise ValueError('Prediction and target shapes/devices must agree')
    with torch.autocast(device_type=pred.device.type, enabled=False):
        error = (pred.float()-target.detach().float()).square().mean(2, keepdim=True).flatten(0, 1)
        error_tiles = _tiles(error, settings['patch_size'])
        achromatic, structural = _selectors(target, settings)
        achromatic_mean, _, _ = _region_mean(error_tiles, achromatic, settings['min_region_pixels'])
        small_mean, _, _ = _region_mean(error_tiles, structural, settings['min_region_pixels'],
            settings['small_region_max_fraction']*settings['patch_size']**2)
        return (settings['uniform_weight']*error_tiles.mean() +
                settings['achromatic_weight']*achromatic_mean + settings['small_weight']*small_mean)


@torch.no_grad()
def detail_diagnostics(target, spec=None):
    """Detached training-target selector accounting; tensors stay on device.

Per-tile counts and eligibility expose coverage rather than semantic labels.
Shapes are [B*T, tiles]; the aggregate fractions are scalar FP32 tensors.
"""
    settings = _settings(spec)
    _validate_target(target, settings)
    with torch.autocast(device_type=target.device.type, enabled=False):
        achromatic, structural = _selectors(target, settings)
        zeros = torch.zeros_like(achromatic)
        _, achromatic_counts, achromatic_valid = _region_mean(zeros, achromatic, settings['min_region_pixels'])
        _, structural_counts, small_valid = _region_mean(zeros, structural, settings['min_region_pixels'],
            settings['small_region_max_fraction']*settings['patch_size']**2)
        return dict(achromatic_pixels_per_tile=achromatic_counts,
            structural_pixels_per_tile=structural_counts,
            achromatic_eligible_tiles=achromatic_valid, small_eligible_tiles=small_valid,
            achromatic_eligible_tile_fraction=achromatic_valid.float().mean(),
            small_eligible_tile_fraction=small_valid.float().mean(),
            achromatic_pixel_fraction=achromatic.mean(), structural_pixel_fraction=structural.mean(),
            uniform_pixel_coverage=target.new_tensor(1., dtype=torch.float32))
