import json
from dataclasses import asdict

import pytest
import torch

from genie_diep.model import Config, WorldModel, VideoTokenizer, image_loss
from genie_diep.baseline import Trainer
from genie_diep.baseline_evaluation import commanded_rollout
from scripts.evaluate_tokenizer_quality import reconstruct
from test_baseline_pipeline import project


torch.set_num_threads(2)


def config(**changes):
    values = dict(size=16, width=8, token_dim=4, vocab=8, context=4, heads=2,
                  layers=1, action_dim=2, action_codes=(3, 2), downsample=4,
                  tokenizer_arch='st_transformer',
                  tokenizer_transformer=dict(encoder_width=16, encoder_depth=2, encoder_heads=2,
                                             decoder_width=16, decoder_depth=2, decoder_heads=2))
    values.update(changes)
    return Config(**values)


def test_encoder_decoder_future_invariance_and_genuine_past_dependence():
    torch.manual_seed(11)
    tok = VideoTokenizer(config()).eval()
    x = torch.rand(1, 4, 3, 16, 16)
    changed = x.clone(); changed[:, 2:] = torch.rand_like(changed[:, 2:])
    z = tok.features(x)
    torch.testing.assert_close(z[:, :2], tok.features(changed)[:, :2], rtol=0, atol=0)
    earlier = x.clone(); earlier[:, :1] = 1-earlier[:, :1]
    assert (z[:, -1]-tok.features(earlier)[:, -1]).abs().max() > 1e-5
    ids = torch.randint(0, 8, (1, 4, 4, 4))
    decoded = tok.decode_ids(ids)
    future = ids.clone(); future[:, 2:] = (future[:, 2:]+1)%8
    torch.testing.assert_close(decoded[:, :2], tok.decode_ids(future)[:, :2], rtol=0, atol=0)
    past = ids.clone(); past[:, :1] = (past[:, :1]+1)%8
    assert (decoded[:, -1]-tok.decode_ids(past)[:, -1]).abs().max() > 1e-5
    # Full-frame spatial attention lets a distant same-frame patch affect a site.
    remote = x.clone(); remote[:, 0, :, -4:, -4:] = 1-remote[:, 0, :, -4:, -4:]
    assert (z[:, 0, 0, 0]-tok.features(remote)[:, 0, 0, 0]).abs().max() > 1e-5


def test_single_frame_shapes_ids_and_finite_optimizer_update():
    torch.manual_seed(12)
    tok = VideoTokenizer(config())
    x = torch.rand(1, 1, 3, 16, 16)
    output, ids, auxiliary = tok(x)
    assert output.shape == x.shape and ids.shape == (1, 1, 4, 4)
    torch.testing.assert_close(output, tok.decode_ids(ids), atol=2e-6, rtol=0)
    before = tok.encoder.patch.weight.detach().clone()
    optimizer = torch.optim.AdamW(tok.parameters(), lr=.001)
    loss = image_loss(output, x)+auxiliary; loss.backward()
    assert torch.isfinite(loss)
    for component in (tok.encoder, tok.vq, tok.decoder):
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum()>0 for g in gradients)
    optimizer.step()
    assert not torch.equal(before, tok.encoder.patch.weight)


def test_quality_adapter_preserves_full_decoder_history_and_batch_option():
    torch.manual_seed(13)
    tok = VideoTokenizer(config()).eval()
    x = torch.rand(4, 3, 16, 16)
    with torch.no_grad():
        native, ids, _ = tok(x[None])
        one, one_ids = reconstruct(tok, x, 1)
        four, four_ids = reconstruct(tok, x, 4)
    torch.testing.assert_close(one, native[0], rtol=0, atol=0)
    torch.testing.assert_close(one, four, rtol=0, atol=0)
    assert torch.equal(ids[0], one_ids) and torch.equal(one_ids, four_ids)


def test_transformer_quality_adapter_rejects_nonfinite_decoder_output(monkeypatch):
    tok = VideoTokenizer(config()).eval()
    monkeypatch.setattr(tok, 'decode', lambda q: torch.full((1, q.shape[1], 3, 16, 16), float('nan')))
    with pytest.raises(ValueError, match='Nonfinite reconstruction'):
        reconstruct(tok, torch.rand(4, 3, 16, 16))


def test_rollout_passes_past_ids_and_sliding_context_to_decoder(monkeypatch):
    model = WorldModel(config()).eval()
    seen = []
    original = model.tokenizer.decode_ids
    def record(ids):
        seen.append(ids.clone())
        return original(ids)
    monkeypatch.setattr(model.tokenizer, 'decode_ids', record)
    prefix = torch.rand(1, 3, 3, 16, 16)
    out = commanded_rollout(model, prefix, torch.zeros(1, 1, 2, dtype=torch.long), 3)
    assert out.shape == (1, 3, 3, 16, 16)
    assert [ids.shape[1] for ids in seen] == [4, 4, 4]
    torch.testing.assert_close(seen[0][:, 1:], seen[1][:, :-1], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['size', 'grid', 'time', 'empty', 'unknown', 'heads'])
def test_shape_and_architecture_errors(kind):
    c = config(); tok = VideoTokenizer(c)
    with pytest.raises(ValueError):
        if kind == 'size': tok.features(torch.rand(1, 2, 3, 12, 12))
        elif kind == 'grid': tok.decode(torch.rand(1, 2, 3, 3, 4))
        elif kind == 'time': tok.features(torch.rand(1, 5, 3, 16, 16))
        elif kind == 'empty': tok.features(torch.rand(1, 0, 3, 16, 16))
        elif kind == 'unknown': VideoTokenizer(config(tokenizer_arch='unknown'))
        else:
            c.tokenizer_transformer['encoder_heads'] = 3
            VideoTokenizer(c)


def test_legacy_cnn_config_serialization_and_state_keys_remain_unchanged():
    c = Config(size=16, width=8, token_dim=4, vocab=8, layers=1, heads=2, downsample=4)
    model = WorldModel(c)
    serialized = model.config_dict()
    expected = asdict(c); expected.pop('tokenizer_arch'); expected.pop('tokenizer_transformer')
    assert serialized == expected
    clone = WorldModel(Config(**serialized)); clone.load_state_dict(model.state_dict(), strict=True)
    x = torch.rand(1, 3, 3, 16, 16)
    torch.testing.assert_close(model.tokenizer(x)[0], clone.tokenizer(x)[0], rtol=0, atol=0)


def test_exact_transformer_trainer_resume_and_implementation_contract(project):
    path = project/'spec.json'; spec = json.loads(path.read_text())
    spec['model'].update(tokenizer_arch='st_transformer', downsample=4,
        tokenizer_transformer=dict(encoder_width=8, encoder_depth=1, encoder_heads=2,
                                  decoder_width=8, decoder_depth=1, decoder_heads=2))
    spec['stages'][0]['gradient_accumulation'] = 1
    path.write_text(json.dumps(spec))
    a = Trainer(project, 'spec.json', 'tokenizer', preflight=True)
    assert torch.isfinite(torch.tensor(a.update()))
    a.save(project/'transformer.pt')
    expected_loss = a.update()
    b = Trainer(project, 'spec.json', 'tokenizer', preflight=True, resume=project/'transformer.pt')
    assert b.update() == expected_loss
    for name, tensor in a.model.state_dict().items():
        torch.testing.assert_close(tensor, b.model.state_dict()[name], rtol=0, atol=0)
    saved = torch.load(project/'transformer.pt', weights_only=True)
    saved['contract']['implementation_sha256'] = 'old-implementation'
    torch.save(saved, project/'stale.pt')
    with pytest.raises(ValueError, match='contract changed'):
        Trainer(project, 'spec.json', 'tokenizer', preflight=True, resume=project/'stale.pt')
