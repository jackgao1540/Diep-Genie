import pytest
from scripts.validate_baseline_checkpoint import validate_annotations


def test_dense_review_requires_all_unique_annotations():
    rows = [{'review_index': 0, 'split': 'train'}, {'review_index': 1, 'split': 'val'}]
    observations = [{'review_index': i, 'firing': 'none_visible', 'notes': 'Observed retained frames; inputs unknown.'} for i in range(2)]
    validate_annotations(rows, observations)
    for invalid in (observations[:1], observations + observations[:1]):
        with pytest.raises(ValueError):
            validate_annotations(rows, invalid)


def test_dense_review_rejects_reserved_test_and_invented_labels():
    observation = [{'review_index': 0, 'firing': 'none_visible', 'notes': 'No visible emission.'}]
    with pytest.raises(ValueError, match='Reserved test'):
        validate_annotations([{'review_index': 0, 'split': 'test'}], observation)
    observation[0]['firing'] = 'verified_fire_key_off'
    with pytest.raises(ValueError, match='Invalid visual'):
        validate_annotations([{'review_index': 0, 'split': 'train'}], observation)
