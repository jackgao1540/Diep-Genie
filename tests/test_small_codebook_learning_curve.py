import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_small_codebook_architecture import validate_endpoint
from scripts import evaluate_small_codebook_learning_curve as curves
from scripts.evaluate_small_codebook_learning_curve import validate_manifest
from tests.test_small_codebook_evaluation import endpoint


def manifest():
    return dict(steps=[3000, 4000, 5000], arms=[dict(name=name, spec='configs/spec.json', checkpoints=[
        dict(step=step, checkpoint=f'runs/{name}-{step}.pt', expected_sha256='a'*64)
        for step in (3000, 4000, 5000)]) for name in ('cnn512', 'cnn32', 'st32')])


def test_intermediates_require_intentional_expected_step_and_final_default_is_strict():
    state, spec = endpoint()
    for step in (3000, 4000):
        state['step'] = step
        with pytest.raises(ValueError, match='5000'):
            validate_endpoint(state, spec, 'binding', 'selection', 'spec')
        validate_endpoint(state, spec, 'binding', 'selection', 'spec', expected_step=step)
    with pytest.raises(ValueError, match='explicitly'):
        validate_endpoint(state, spec, 'binding', 'selection', 'spec', expected_step=4500)


@pytest.mark.parametrize('mutation', ['missing-step', 'duplicate-step', 'wrong-step-list', 'missing-arm', 'bad-hash'])
def test_learning_curve_manifest_rejects_partial_or_ambiguous_exposure(mutation):
    data = manifest(); validate_manifest(data)
    if mutation == 'missing-step':
        data['arms'][0]['checkpoints'].pop()
    elif mutation == 'duplicate-step':
        data['arms'][0]['checkpoints'][0]['step'] = 4000
    elif mutation == 'wrong-step-list':
        data['steps'] = [1000, 4000, 5000]
    elif mutation == 'missing-arm':
        data['arms'].pop()
    else:
        data['arms'][0]['checkpoints'][0]['expected_sha256'] = 'unknown'
    with pytest.raises(ValueError):
        validate_manifest(data)


def test_manifest_allows_checkpoint_list_order_but_not_changed_steps():
    data = manifest()
    data['arms'][0]['checkpoints'].reverse()
    assert validate_manifest(data) == data['arms']
    original = copy.deepcopy(data)
    validate_manifest(data)
    assert data == original


def test_complete_curve_pipeline_preserves_all_windows_and_emits_only_metrics(tmp_path, monkeypatch):
    source_paths = ['scripts/evaluate_small_codebook_learning_curve.py', 'scripts/evaluate_small_codebook_architecture.py',
                    'scripts/evaluate_tokenizer_quality.py', 'scripts/diagnose_tokenizer_regression.py',
                    'genie_diep/model.py', 'genie_diep/transformer_tokenizer.py', 'genie_diep/cohort.py']
    for path in source_paths:
        target = tmp_path/path; target.parent.mkdir(exist_ok=True, parents=True); target.write_text('fixture')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest()))
    items = [dict(source_id=f'source-{i//4}', index=i, length=16) for i in range(64)]
    entries = [dict(name=name, step=step, training_contract={'fixture': True})
               for name in ('cnn512', 'cnn32', 'st32') for step in (3000, 4000, 5000)]

    class Dataset:
        def locate(self, item):
            return item['index'], 0

        def window(self, sequence, start):
            return torch.full((16, 3, 4, 4), .5)

    monkeypatch.setattr(curves, 'prepare', lambda root, data: (entries, Dataset(), items, ('binding', 'selection')))
    monkeypatch.setattr(curves, 'load_model', lambda entry: (SimpleNamespace(tokenizer=entry), {}))
    calls = []

    def prediction(tokenizer, clip, frame_batch):
        assert clip.shape[0] == 16
        calls.append((tokenizer['name'], tokenizer['step']))
        return clip+(5000-tokenizer['step'])/100000, None

    monkeypatch.setattr(curves, 'reconstruct', prediction)
    report = curves.run(SimpleNamespace(root=str(tmp_path), out='runs/curve', manifest='manifest.json', threads=1, frame_batch=1))
    assert report['status'] == 'completed'
    assert len(calls) == 9*64
    assert len(report['endpoints']) == 9
    assert len(list((tmp_path/'runs/curve').iterdir())) == 10
    assert all(path.suffix == '.json' for path in (tmp_path/'runs/curve').iterdir())
    assert report['curves']['cnn32']['5000']['pooled']['mse'] == 0
    assert report['paired_late_changes']['st32']['5000 minus 3000']['mse']['mean'] < 0
    assert report['paired_model_differences_at_step']['5000']['cnn32 minus cnn512']['mse']['mean'] == 0
