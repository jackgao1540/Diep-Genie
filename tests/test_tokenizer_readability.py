import torch

from scripts.diagnose_tokenizer_readability import (
    fit_ridge, neighborhoods, patches, predict, source_groups, unpatch,
)


def test_patch_order_and_raw_control_can_recover_subpixel_rgb():
    image = torch.rand((3, 128, 128), generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(unpatch(patches(image)), image, rtol=0, atol=0)
    raw = torch.nn.functional.pixel_unshuffle(image[None], 4)[0]
    raw = torch.nn.functional.pad(raw, (0, 0, 0, 0, 0, 16)).permute(1, 2, 0)
    x = neighborhoods(raw)
    fitted = fit_ridge(x, patches(image), 1e-8)
    result = predict(x, fitted)
    assert float((result-image).square().mean()) < 1e-10


def test_source_groups_are_disjoint_and_order_independent():
    items = [{'source_id': f'source-{i}'} for i in range(84)]
    fit, tune = source_groups(items)
    assert len(fit)==64 and len(tune)==20 and not fit&tune
    assert source_groups(list(reversed(items))) == (fit, tune)


def test_ridge_generalizes_linear_signal_without_target_in_prediction():
    rng = torch.Generator().manual_seed(7)
    x = torch.randn((1000, 6), generator=rng)
    w = torch.randn((6, 48), generator=rng)*.01
    y = x@w+.5
    fit = fit_ridge(x, y, 1e-8)
    xt = torch.randn((1024, 6), generator=rng)
    result = predict(xt, fit)
    torch.testing.assert_close(result, unpatch(xt@w+.5), atol=2e-7, rtol=0)
