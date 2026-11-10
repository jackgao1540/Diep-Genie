import torch

from genie_diep.model import Config, VideoTokenizer
from scripts.diagnose_tokenizer_latent_inversion import LocalInput, nearest_codes
from scripts.diagnose_tokenizer_oracle import crop, objective, roi_counts


def test_local_inversion_objective_and_gradients_match_full_decode():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = VideoTokenizer(Config(width=8, token_dim=4, vocab=8, downsample=4)).eval().requires_grad_(False)
        grid = torch.randn(4, 19, 20)
        target = torch.rand(3, 76, 80)
    roi = torch.zeros(76, 80, dtype=torch.bool)
    roi[5:13, 36:47] = True
    local = LocalInput(grid, roi)
    values = (grid[:, local.y, local.x].T + .2).requires_grad_()
    counts = roi_counts(roi)
    full_grid = local.insert(values)
    edited = torch.zeros_like(grid[0], dtype=torch.bool)
    edited[local.y, local.x] = True
    assert torch.equal(full_grid[:, ~edited], grid[:, ~edited])
    full_loss = objective(model.decoder(full_grid[None])[0], target, roi, counts)
    local_loss = objective(model.decoder(local.insert(values, local=True)[None])[0],
                           crop(target, local.pixels), crop(roi, local.pixels), counts)
    torch.testing.assert_close(local_loss, full_loss, atol=1e-7, rtol=0.)
    full_grad, = torch.autograd.grad(full_loss, values)
    local_grad, = torch.autograd.grad(local_loss, values)
    torch.testing.assert_close(local_grad, full_grad, atol=1e-7, rtol=0.)
    assert torch.count_nonzero(local_grad)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nearest_code_projection_uses_cosine_and_existing_vectors():
    codebook = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    values = torch.tensor([[.2, .8], [-.7, .3]])
    projected, ids, cosine = nearest_codes(values, codebook)
    scaled, scaled_ids, scaled_cosine = nearest_codes(values*torch.tensor([[5.], [.1]]), codebook)
    assert ids.tolist() == [1, 2]
    assert torch.equal(projected, codebook[ids])
    assert torch.equal(scaled_ids, ids)
    assert torch.equal(scaled, projected)
    torch.testing.assert_close(scaled_cosine, cosine)
