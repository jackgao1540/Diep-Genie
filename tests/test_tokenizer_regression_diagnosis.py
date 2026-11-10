import itertools

import pytest
import torch
from torch.nn import functional as F

from genie_diep.model import VQ, image_loss
from scripts.diagnose_tokenizer_regression import (
    code_masks, code_statistics, decode, means, prepare_target, quantize,
    reconstruction_metrics, summarize_codes, variants,
)
from scripts.evaluate_tokenizer_quality import metric_means, metric_totals


@torch.inference_mode()
def test_cosine_quantization_matches_native_vq():
    layer = VQ(5, 3)
    features = torch.randn((1, 2, 3, 4, 3), generator=torch.Generator().manual_seed(71))
    expected_q, expected_ids, expected_loss = layer(features)
    z = F.normalize(features, dim=-1)
    code = F.normalize(layer.embedding.weight, dim=-1)
    actual_q, actual_ids, margins, actual_loss = quantize(z, code, objective=True)
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_loss, expected_loss)
    assert margins.shape == expected_ids.shape and (margins >= 0).all()
    mask = torch.ones_like(margins[0], dtype=torch.bool)
    summary = summarize_codes([code_statistics(z, actual_q, actual_ids, margins, {'all': mask}, len(code))])['all']
    assert summary['tokens'] == expected_ids.numel()
    assert sum(summary['margin_histogram']) == expected_ids.numel()
    assert sum(summary['code_histogram']) == expected_ids.numel()
    assert summary['distortion'] == pytest.approx((actual_q-z).square().sum(-1).mean().item())


def test_factorial_routing_preserves_quantized_vectors_and_normalizes_bypass():
    choices = ('b4', 'b5')
    baseline = [entry for entry in variants() if all(name in choices for name in entry[1:4]) and not entry[4]]
    assert {entry[1:4] for entry in baseline} == set(itertools.product(choices, repeat=3))
    assert len(baseline) == 8

    class RecordingDecoder:
        def decode(self, vectors):
            return vectors

    z = torch.tensor([[1., 0.]])
    q = torch.tensor([[0., 1.]])
    torch.testing.assert_close(decode(RecordingDecoder(), q, z, 0.), q[0])
    torch.testing.assert_close(decode(RecordingDecoder(), q, z, 1.), z[0])
    torch.testing.assert_close(decode(RecordingDecoder(), q, z, .25),
                               F.normalize(torch.tensor([.25, .75]), dim=0))


def test_metric_decomposition_matches_existing_color_and_temporal_metrics():
    source = torch.full((2, 3, 8, 8), .5)
    source[:, 0, 2:4, 2:4] = .9
    prediction = source.clone()
    prediction[:, :, ::4, 1::4] += .125
    prediction[1, :, 0, 0] -= .125
    target = prepare_target(source)
    row = reconstruction_metrics(source, prediction, target, native_vq=.01)
    summary = means([row])
    existing = metric_means(metric_totals(source, prediction))
    assert summary['whole']['mse'] == pytest.approx(existing['mse'])
    assert summary['colored']['mse'] == pytest.approx(existing['foreground_mse'])
    assert summary['temporal_mse'] == pytest.approx(existing['temporal_difference_mse'])
    assert summary['whole']['mse'] == pytest.approx(
        summary['whole']['bias_square_mse']+summary['whole']['centered_mse'])
    assert summary['colored']['pixels']+summary['noncolored']['pixels'] == 2*8*8
    assert summary['objective']['image_loss'] == pytest.approx(
        float(image_loss(prediction[None], source[None], color_balance=True)))
    assert summary['neutral_flat_phase']['residual_sum'][0][0][1] == pytest.approx(.125)
    assert code_masks(target)['whole'].shape == (2, 2, 2)
