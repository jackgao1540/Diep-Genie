import pytest
import torch
from genie_diep.data import ClipDataset


def test_source_mass_is_equal_despite_clip_imbalance():
    dataset=ClipDataset.__new__(ClipDataset)
    dataset.rows=[dict(source_id='many',split='train') for _ in range(9)]+[dict(source_id='rare',split='train')]
    weights=dataset.source_balanced_weights()
    assert torch.allclose(weights[:9].sum(),weights[9])
    dataset.rows.append(dict(source_id='heldout',split='val'))
    with pytest.raises(ValueError,match='training-only'):dataset.source_balanced_weights()
