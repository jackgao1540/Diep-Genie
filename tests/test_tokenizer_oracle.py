import torch

from genie_diep.model import Config, VideoTokenizer
from scripts.diagnose_tokenizer_oracle import crop, local_candidates, objective, roi_counts, support, verify_local


@torch.inference_mode()
def test_sparse_decoder_matches_full_decoder_at_interior_and_boundaries():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = VideoTokenizer(Config(width=8, token_dim=4, vocab=8, downsample=4)).eval()
        grid = torch.randn(4, 7, 8)
        codes = torch.randn(8, 4)
    assert verify_local(model.decoder, grid, codes, ((0, 0), (0, 4), (3, 4), (6, 7))) < 3e-6


@torch.inference_mode()
def test_candidate_halo_preserves_change_in_full_roi_objective():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = VideoTokenizer(Config(width=8, token_dim=4, vocab=8, downsample=4)).eval()
        grid = torch.randn(4, 7, 8)
        candidates = torch.randn(3, 4)
        target = torch.rand(3, 28, 32)
    roi = torch.zeros(28, 32, dtype=torch.bool)
    roi[8:27, 8:29] = True
    site = (3, 4)
    patches, bounds = local_candidates(model.decoder, grid, site, candidates)
    assert bounds == support(*site, *roi.shape)
    top, bottom, left, right = bounds
    halo = top-1, bottom+1, left-1, right+1
    incumbent = model.decoder(grid[None])[0]
    full = incumbent[None].expand(len(candidates), -1, -1, -1).clone()
    crop(full, bounds).copy_(patches)
    for counts in ((1, 1), roi_counts(roi)):
        full_change = objective(full, target, roi, counts)-objective(incumbent, target, roi, counts)
        local_change = (objective(crop(full, halo), crop(target, halo), crop(roi, halo), counts)
                        -objective(crop(incumbent, halo), crop(target, halo), crop(roi, halo), counts))
        torch.testing.assert_close(local_change, full_change, atol=1e-10, rtol=0.)
