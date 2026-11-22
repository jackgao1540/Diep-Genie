import copy

import numpy as np
from PIL import Image
import pytest
import torch

from genie_diep.baseline import implementation_hash
from scripts.evaluate_staged_tokenizer import (
    validate_staged_state, validate_frozen_pair, validate_initial, hard_predictions,
    usage_masks, comparison_sheet,
)


def endpoint(phase='joint'):
    model = dict(size=128, context=16, vocab=16, token_dim=64)
    spec = dict(model=model)
    contract = dict(stage='tokenizer', binding_sha256='data', evaluation_sha256='selection',
        spec_sha256='spec', plan_sha256='plan', phase=phase, execution_gate='staged-plan-v1',
        implementation_sha256=implementation_hash(), microbatch=2, accumulation=8,
        staged_mode='full', preflight=False, device_type='cuda', precision='fp16',
        handoff_sha256='joint-endpoint' if phase == 'decoder_only' else None)
    state = dict(config=copy.deepcopy(model), step=5000, contract=contract,
        model={'tokenizer.encoder.weight':torch.ones(2), 'tokenizer.vq.embedding.weight':torch.ones(16,64),
               'tokenizer.decoder.weight':torch.ones(3), 'dynamics.weight':torch.ones(2)},
        optimizer=dict(state={}))
    return state, spec


def test_rejects_wrong_phase_plan_and_nonproduction_endpoint():
    state, spec = endpoint()
    validate_staged_state(state, spec, 'data', 'selection', 'spec', 'plan', 'joint')
    with pytest.raises(ValueError, match='plan identity'):
        validate_staged_state(state, spec, 'data', 'selection', 'spec', 'another-plan', 'joint')
    with pytest.raises(ValueError, match='phase/execution'):
        validate_staged_state(state, spec, 'data', 'selection', 'spec', 'plan', 'decoder_only')
    state['contract'].update(staged_mode='pilot', device_type='cpu', precision='fp32', preflight=True)
    state['step'] = 1
    with pytest.raises(ValueError, match='production'):
        validate_staged_state(state, spec, 'data', 'selection', 'spec', 'plan', 'joint')
    validate_staged_state(state, spec, 'data', 'selection', 'spec', 'plan', 'joint', True)


def test_rejects_wrong_handoff_even_when_encoder_weights_match():
    a, _ = endpoint(); b, _ = endpoint('decoder_only')
    assert validate_frozen_pair(a, b, 'joint-endpoint') == 2
    with pytest.raises(ValueError, match='actual joint endpoint'):
        validate_frozen_pair(a, b, 'a-different-checkpoint-with-identical-weights')
    b['model']['tokenizer.encoder.weight'][0] += 1
    with pytest.raises(ValueError, match='tensor differs'):
        validate_frozen_pair(a, b, 'joint-endpoint')


def test_initial_validation_detects_changed_inactive_dynamics():
    state, spec = endpoint()
    initial = copy.deepcopy(state); initial['step'] = 0
    assert validate_initial(initial, state, spec, 'data', 'selection', 'spec') == 1
    state['model']['dynamics.weight'][0] += 1
    with pytest.raises(ValueError, match='Inactive action/dynamics'):
        validate_initial(initial, state, spec, 'data', 'selection', 'spec')


def test_reconstruction_uses_real_ids_and_rejects_frozen_id_drift():
    class VQ:
        def lookup(self, ids): return ids[..., None].float()
    class Tokenizer:
        vq = VQ()
        def __init__(self, code): self.code = code
        def encode(self, clip): return torch.full_like(clip,99), torch.full((1,16,2,2),self.code), None
        def decode_ids(self, ids): return ids[:, :, None].expand(-1,-1,3,-1,-1).float()
    class Model:
        def __init__(self, code): self.tokenizer = Tokenizer(code)
    clip = torch.zeros(16,3,8,8)
    models = dict(joint=Model(1), decoder_only=Model(1))
    predictions, ids = hard_predictions(models, clip)
    assert predictions['joint'].max() == 1 and ids.max() == 1
    models['decoder_only'] = Model(2)
    with pytest.raises(ValueError, match='different discrete IDs'): hard_predictions(models, clip)


def test_region_masks_count_footprints_without_inventing_semantic_labels():
    from scripts.diagnose_tokenizer_regression import prepare_target
    source = torch.zeros(16,3,8,8)
    source[:,0,0,0:4] = 1
    masks = usage_masks(prepare_target(source))
    assert masks['foreground'][:,0,0].all()
    assert not masks['foreground'][:,1,1].any()
    assert masks['whole'].shape == (16,2,2)
    assert not {'tank','food','bullet'} & set(masks)


def test_comparison_sheet_preserves_exact_aligned_pixels(tmp_path):
    array = np.arange(128*128*3,dtype=np.uint8).reshape(128,128,3)
    artifact = comparison_sheet(tmp_path, 38, dict(source_id='source'), [array]*3, (32,32,96,96))
    sheet = np.asarray(Image.open(tmp_path / artifact['path']))
    for column in range(3):
        assert np.array_equal(sheet[25:281:2,column*520:column*520+256:2], array)
        assert np.array_equal(sheet[25:281:4,column*520+260:column*520+516:4], array[32:96,32:96])
