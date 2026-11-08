import copy

import pytest

from scripts.report_longform_coverage import select_sample, summarize_annotations


def block(source, number, split, frames=40):
    return dict(source_id=source, block_id=str(number), split=split, frames=frames, fps=10)


def test_sampling_is_order_independent_and_excludes_reserved_test():
    rows = [block('training', n, 'train') for n in range(8)]
    rows += [block('validation', n, 'val') for n in range(4)]
    rows += [block('reserved', n, 'test') for n in range(10)]
    counts = {'train':4, 'val':2}
    selected = select_sample(rows, 'fixed-seed', counts)
    assert selected == select_sample(list(reversed(rows)), 'fixed-seed', counts)
    assert {r['split'] for r in selected} == {'train', 'val'}
    assert len({(r['source_id'], r['block_id']) for r in selected}) == 6
    with pytest.raises(ValueError, match='Reserved test'):
        select_sample(rows, 'fixed-seed', {'test':1})


def test_sampling_refuses_duplicates_and_unequal_duration():
    r = block('training', 0, 'train')
    with pytest.raises(ValueError, match='Repeated source block'):
        select_sample([r, r], 'seed', {'train':1})
    with pytest.raises(ValueError, match='equal durations'):
        select_sample([r, block('training', 1, 'train', frames=20)], 'seed', {'train':1})


def sample_review():
    index = dict(rows=[dict(review_index=0, split='train', source_id='a', seconds=4),
                       dict(review_index=1, split='val', source_id='b', seconds=4)])
    review = dict(definitions={'player_firing_evidence':'observable emission'}, annotations=[
        dict(review_index=0, player_firing_evidence='uncertain'),
        dict(review_index=1, player_firing_evidence='not_seen')])
    return index, review


def test_uncertainty_is_not_counted_as_positive_or_negative():
    index, review = sample_review()
    result = summarize_annotations(index, review)
    assert result['train']['counts']['player_firing_evidence'] == {'uncertain':1}
    assert result['val']['counts']['player_firing_evidence'] == {'not_seen':1}
    assert result['train']['sources_with_observation']['player_firing_evidence'] == 0


def test_semantic_review_requires_complete_unique_valid_annotations():
    index, review = sample_review()
    bad = copy.deepcopy(review)
    bad['annotations'][1]['review_index'] = 0
    with pytest.raises(ValueError, match='Missing, repeated'):
        summarize_annotations(index, bad)
    bad = copy.deepcopy(review)
    bad['annotations'][0]['player_firing_evidence'] = 'probably'
    with pytest.raises(ValueError, match='Invalid semantic value'):
        summarize_annotations(index, bad)
    index['rows'][1]['split'] = 'test'
    with pytest.raises(ValueError, match='Reserved test'):
        summarize_annotations(index, review)
