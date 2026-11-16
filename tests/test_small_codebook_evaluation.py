import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.evaluate_small_codebook_architecture import (
    aggregate, paired_differences, private_output, reconstruct, save_atlas, totals, validate_endpoint,
)
from scripts.diagnose_tokenizer_regression import prepare_target


def endpoint():
    spec = {'model': {'size': 128, 'context': 16, 'vocab': 32}}
    state = dict(step=5000, config=copy.deepcopy(spec['model']), model={'weight': torch.ones(2)}, contract=dict(
        stage='tokenizer', binding_sha256='binding', evaluation_sha256='selection', spec_sha256='spec',
        preflight=False, precision='fp16', device_type='cuda', microbatch=4, accumulation=4))
    return state, spec


@pytest.mark.parametrize('field,value,error', [
    ('binding_sha256', 'wrong', 'binding'), ('evaluation_sha256', 'wrong', 'selection'),
    ('spec_sha256', 'wrong', 'specification'), ('preflight', True, 'production'),
    ('stage', 'actions', 'tokenizer'), ('precision', 'fp32', 'provenance'),
    ('device_type', 'cpu', 'provenance'), ('microbatch', 1, 'microbatches'),
])
def test_rejects_unmatched_or_preflight_endpoint(field, value, error):
    state, spec = endpoint()
    validate_endpoint(state, spec, 'binding', 'selection', 'spec')
    state['contract'][field] = value
    with pytest.raises(ValueError, match=error):
        validate_endpoint(state, spec, 'binding', 'selection', 'spec')


def test_rejects_wrong_step_and_nonfinite_weights():
    state, spec = endpoint(); state['step'] = 4999
    with pytest.raises(ValueError, match='5000'):
        validate_endpoint(state, spec, 'binding', 'selection', 'spec')
    state['step'] = 5000; state['model']['weight'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_endpoint(state, spec, 'binding', 'selection', 'spec')


def test_output_guard_rejects_public_and_existing_paths(tmp_path):
    with pytest.raises(ValueError, match='private'):
        private_output(tmp_path, 'docs/comparison')
    with pytest.raises(ValueError, match='private'):
        private_output(tmp_path, 'runs/../docs/comparison')
    with pytest.raises(ValueError, match='private'):
        private_output(tmp_path, 'runs')
    (tmp_path/'runs'/'existing').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        private_output(tmp_path, 'runs/existing')
    assert private_output(tmp_path, 'runs/fresh') == tmp_path/'runs'/'fresh'


def test_neutral_brightness_source_mask_and_source_paired_uncertainty():
    source = torch.full((16, 3, 8, 8), .5)
    neutral = prepare_target(source)[0]['neutral_flat']
    records = []
    for source_id in ('one', 'two'):
        records.append(dict(selection=dict(source_id=source_id), arms=dict(
            original=totals(source, source, neutral), dark=totals(source, source-.1, neutral))))
    summary = {name: aggregate(records, name) for name in ('original', 'dark')}
    assert summary['dark']['pooled']['neutral_flat_luminance_bias'] == pytest.approx(-.1)
    assert summary['dark']['pooled']['mse'] == pytest.approx(.01)
    assert summary['dark']['pooled']['temporal_difference_mse'] == 0
    assert summary['dark']['pooled']['foreground_mse'] is None
    difference = paired_differences(summary)['dark minus original']
    assert difference['mse']['source_count'] == 2
    assert difference['mse']['source_bootstrap_95_percent'] == pytest.approx([.01, .01])
    assert difference['neutral_flat_luminance_bias']['mean'] == pytest.approx(-.1)


def test_transformer_reconstruction_passes_all_sixteen_frames_to_decoder():
    class TemporalTokenizer:
        training = False
        temporal_decoder = True

        def encode(self, clip):
            assert clip.shape[1] == 16
            return clip, torch.arange(16).reshape(1, 16, 1, 1), None

        def decode(self, q):
            assert q.shape[1] == 16
            return q.cumsum(1)

    clip = torch.arange(16).float().reshape(16, 1, 1, 1).expand(16, 3, 2, 2)
    prediction, codes = reconstruct(TemporalTokenizer(), clip, frame_batch=1)
    assert prediction[-1, 0, 0, 0] == 120
    assert codes.flatten().tolist() == list(range(16))


def test_atlas_is_exact_row_major_pixel_data(tmp_path):
    frames = np.stack([np.full((128, 128, 3), index, dtype=np.uint8) for index in range(16)])
    artifact = save_atlas(tmp_path, 'cnn32', 7, frames)
    atlas = np.array(Image.open(tmp_path/artifact['path']))
    assert atlas.shape == (512, 512, 3)
    for index, frame in enumerate(frames):
        row, col = divmod(index, 4)
        assert np.array_equal(atlas[row*128:(row+1)*128, col*128:(col+1)*128], frame)
    assert len(artifact['sha256']) == 64
