import pytest
import torch

from genie_diep.detail_loss import DEFAULT_SPEC, detail_diagnostics, detail_loss


def test_identical_predictions_have_exact_zero_loss_and_gradient():
    torch.manual_seed(12)
    target=torch.rand(2,3,3,16,16)
    pred=target.clone().requires_grad_()
    loss=detail_loss(pred,target)
    assert loss.dtype==torch.float32 and loss.item()==0
    loss.backward()
    assert torch.count_nonzero(pred.grad)==0


def test_constant_empty_masks_keep_uniform_coverage_and_finite_gradients():
    target=torch.full((1,2,3,16,16),.4)
    pred=torch.full_like(target,.6,requires_grad=True)
    loss=detail_loss(pred,target)
    assert loss.item()==pytest.approx(.25*.2**2)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and torch.count_nonzero(pred.grad)==pred.numel()
    stats=detail_diagnostics(target)
    assert stats['uniform_pixel_coverage']==1
    assert not stats['achromatic_eligible_tiles'].any()
    assert not stats['small_eligible_tiles'].any()


def test_thin_grey_geometry_is_stronger_than_equal_background_error():
    target=torch.full((1,1,3,32,32),.5)
    target[...,4]=.2
    missing=target.clone();missing[...,4]=.5
    background=target.clone();background[...,20]=.2
    assert (missing-target).square().mean()==(background-target).square().mean()
    assert detail_loss(missing,target)>4*detail_loss(background,target)


def test_regions_normalize_each_tile_instead_of_weighting_by_selected_area():
    # Direct accounting through constructed masks checks the key mathematical
    # distinction: one selected pixel and 16 selected pixels count equally.
    from genie_diep.detail_loss import _region_mean
    error=torch.zeros(1,2,64);mask=torch.zeros_like(error)
    error[0,0,0]=1;mask[0,0,0]=1
    error[0,1,:16]=.25;mask[0,1,:16]=1
    value,_,_=_region_mean(error,mask,minimum=1)
    assert value.item()==pytest.approx((1+.25)/2)
    assert value.item()!=pytest.approx((1+16*.25)/17)


def test_empty_frames_do_not_receive_reallocated_structural_weight():
    target=torch.full((1,1,3,16,16),.5);target[...,4]=.2
    pred=target.clone();pred[...,4]=.5
    empty=torch.full_like(target,.5)
    assert detail_loss(torch.cat([pred,empty],1),torch.cat([target,empty],1))==detail_loss(pred,target)/2


def test_target_detached_selectors_deterministic_and_no_input_mutation():
    torch.manual_seed(5)
    target=torch.rand(1,2,3,16,16,requires_grad=True)
    pred=torch.rand_like(target,requires_grad=True)
    original=target.detach().clone();state=torch.get_rng_state().clone()
    first=detail_loss(pred,target);second=detail_loss(pred,target)
    assert torch.equal(first,second) and torch.equal(torch.get_rng_state(),state)
    first.backward()
    assert target.grad is None and torch.equal(original,target)
    assert torch.isfinite(pred.grad).all()
    assert all(not item.requires_grad for item in detail_diagnostics(target).values())


def test_half_input_and_outer_autocast_accumulate_in_float32():
    target=torch.full((1,2,3,16,16),.5,dtype=torch.float16)
    target[...,4]=.2
    pred=target.clone();pred[...,4]=.5;pred.requires_grad_()
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        value=detail_loss(pred,target)
    assert value.dtype==torch.float32
    value.backward()
    assert pred.grad.dtype==torch.float16 and torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad)>0


def test_unit_range_loss_is_bounded_and_small_selector_uses_sparse_tiles():
    target=torch.zeros(1,1,3,16,16);target[:,:,0,3:5,3:5]=1
    pred=1-target
    value=detail_loss(pred,target)
    assert 0<=value<=1
    stats=detail_diagnostics(target)
    valid=stats['small_eligible_tiles'];counts=stats['structural_pixels_per_tile']
    assert valid.any()
    assert ((counts[valid]>=2)&(counts[valid]<=32)).all()


@pytest.mark.parametrize('spec',[
    {'unknown':1}, {'patch_size':True}, {'gradient_threshold':float('nan')},
    {'uniform_weight':2}, {'small_region_max_fraction':0},
    {'min_region_pixels':63,'small_region_max_fraction':.5}, {'dilation_radius':3}])
def test_invalid_spec_fails(spec):
    with pytest.raises(ValueError):
        detail_loss(torch.zeros(1,1,3,16,16),torch.zeros(1,1,3,16,16),spec)


def test_invalid_shapes_fail_and_default_spec_is_not_mutated():
    original=dict(DEFAULT_SPEC)
    with pytest.raises(ValueError,match='divisible'):
        detail_loss(torch.zeros(1,1,3,15,16),torch.zeros(1,1,3,15,16))
    with pytest.raises(ValueError,match='shapes'):
        detail_loss(torch.zeros(1,1,3,16,16),torch.zeros(1,2,3,16,16))
    detail_loss(torch.zeros(1,1,3,16,16),torch.zeros(1,1,3,16,16),{'gradient_threshold':.1})
    assert DEFAULT_SPEC==original
