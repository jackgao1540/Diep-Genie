"""Coverage schedule, split safety, inherited RNG and transactional restoration."""
import copy
from collections import defaultdict

import pytest
import torch

from genie_diep.cohort import MixedSampler
from genie_diep.coverage_sampler import CoverageSampler, sampler_state_sha256


class MetadataCohort:
    def __init__(self, sequences=((40,), (32, 40), (16,)), split='train'):
        self.split = split
        self.length = 16
        self.binding_sha256 = 'a' * 64
        self.rows = {}
        self.sequences = []
        self.by_source = defaultdict(list)
        self.decode_calls = []
        for i, counts in enumerate(sequences):
            source = f'source-{i % 2}'
            keys = [(source, f'{i}-{j}') for j in range(len(counts))]
            for key, count in zip(keys, counts):
                self.rows[key] = dict(split=split, frames=count)
            self.sequences.append(dict(source_id=source, keys=keys, windows=sum(counts) - 15))
            self.by_source[source].append(i)
        # The manifest includes other splits; their frames must never enter the plan.
        self.rows[('heldout', 'reserved')] = dict(split='test', frames=40)
        self.sources = sorted(self.by_source)
        self.cumulative = self._cumulative(range(len(self.sequences)))
        self.source_cumulative = {s: self._cumulative(ids) for s, ids in self.by_source.items()}

    def _cumulative(self, ids):
        total = 0
        result = []
        for i in ids:
            total += self.sequences[i]['windows']
            result.append(total)
        return result

    def window(self, sequence, start):
        assert 0 <= start < self.sequences[sequence]['windows']
        self.decode_calls.append((sequence, start))
        return torch.tensor([sequence, start])


def make(cohort=None, seed=20261110):
    cohort = cohort or MetadataCohort()
    parent = MixedSampler(cohort, seed=42)
    for _ in range(13):
        parent.draw()
    state = parent.state_dict()
    sampler = CoverageSampler(cohort, state, seed=seed,
                              expected_parent_sampler_sha256=sampler_state_sha256(state))
    return sampler, parent


def assert_state_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], dict):
            assert_state_equal(a[key], b[key])
        elif isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


def test_complete_pass_covers_every_train_frame_and_preserves_sequence_boundaries():
    sampler, parent = make()
    stats = sampler.plan_stats()
    assert stats['total_training_frames'] == 128
    assert stats['coverage_windows'] == 9  # 40 -> 0,16,24; 72 -> 0,16,32,48,56; 16 -> 0.
    assert sampler.cohort.decode_calls == []
    visited = [set() for _ in sampler.cohort.sequences]
    starts = [set() for _ in visited]
    for _ in range(stats['coverage_windows']):
        sequence, start = sampler.draw()
        starts[sequence].add(start)
        visited[sequence].update(range(start, start + 16))
        assert sampler.plan_stats()['coverage_unique_frames'] == sum(map(len, visited))
        assert_state_equal(sampler.state_dict()['mixed'], parent.state_dict())
    assert starts == [{0, 16, 24}, {0, 16, 32, 48, 56}, {0}]
    assert visited == [set(range(40)), set(range(72)), set(range(16))]
    assert sampler.plan_stats()['full_coverage']
    assert sampler.plan_stats()['coverage_unique_frames'] == 128
    assert sampler.phase == 'mixed'
    assert sampler.draws == 13 + 9
    assert sampler.cohort.decode_calls == []


def test_fallback_resumes_exact_inherited_random_stream_without_global_rng_changes():
    global_rng = torch.get_rng_state().clone()
    sampler, parent = make()
    for _ in range(sampler.plan_stats()['coverage_windows']):
        sampler.draw()
    assert [sampler.draw() for _ in range(100)] == [parent.draw() for _ in range(100)]
    assert sampler.plan_stats()['fallback_draws'] == 100
    assert sampler.plan_stats()['coverage_unique_frames'] == 128
    assert_state_equal(sampler.state_dict()['mixed'], parent.state_dict())
    assert torch.equal(global_rng, torch.get_rng_state())


@pytest.mark.parametrize('prefix', [0, 3, 8, 9, 14])
def test_resume_and_rollback_reproduce_both_phases(prefix):
    sampler, _ = make()
    for _ in range(prefix):
        sampler.draw()
    boundary = copy.deepcopy(sampler.state_dict())
    expected = [sampler.draw() for _ in range(20)]
    after = sampler.state_dict()
    sampler.load_state_dict(boundary)
    assert [sampler.draw() for _ in range(20)] == expected
    assert_state_equal(sampler.state_dict(), after)
    restored, _ = make()
    restored.load_state_dict(boundary)
    assert [restored.draw() for _ in range(20)] == expected
    assert_state_equal(restored.state_dict(), after)


def test_batch_can_cross_phase_boundary_and_state_has_no_rng_aliases():
    sampler, parent = make()
    for _ in range(8):
        sampler.draw()
    boundary = sampler.state_dict()
    expected = [sampler.draw(), parent.draw(), parent.draw()]
    sampler.load_state_dict(boundary)
    assert sampler.batch(3).tolist() == [list(pair) for pair in expected]
    assert sampler.cohort.decode_calls == expected
    assert sampler.draws == 24
    saved = sampler.state_dict()
    saved['mixed']['rng'].zero_()
    assert not torch.equal(saved['mixed']['rng'], sampler.state_dict()['mixed']['rng'])
    with pytest.raises(ValueError, match='batch size'):
        sampler.batch(0)


def test_shuffle_is_deterministic_and_plan_binds_metadata_seed_and_parent():
    first, _ = make()
    same, _ = make()
    other, _ = make(seed=123)
    assert first.plan_sha256 == same.plan_sha256
    assert first.plan_sha256 != other.plan_sha256
    assert [first.draw() for _ in range(9)] == [same.draw() for _ in range(9)]
    changed, _ = make(MetadataCohort(((41,), (32, 40), (16,))))
    assert changed.plan_sha256 != first.plan_sha256
    with pytest.raises(ValueError, match='identity changed'):
        other.load_state_dict(first.state_dict())
    state = first.state_dict()['mixed']
    with pytest.raises(ValueError, match='Parent sampler identity'):
        CoverageSampler(first.cohort, state, expected_parent_sampler_sha256='0' * 64)


@pytest.mark.parametrize('field,value', [
    ('phase', 'mixed'), ('coverage_index', 99), ('coverage_index', -1),
    ('coverage_index', True), ('draws', 999), ('coverage_unique_frames', 0),
    ('seed', 7), ('parent_draws', 0), ('plan_sha256', '0' * 64),
    ('parent_sampler_sha256', '0' * 64), ('binding_sha256', '0' * 64),
])
def test_corrupt_state_rejected_without_partial_mutation(field, value):
    sampler, _ = make()
    for _ in range(3):
        sampler.draw()
    before = sampler.state_dict()
    corrupt = copy.deepcopy(before)
    corrupt[field] = value
    with pytest.raises(ValueError):
        sampler.load_state_dict(corrupt)
    assert_state_equal(sampler.state_dict(), before)


def test_inherited_rng_cannot_advance_during_coverage_even_with_consistent_checksum():
    sampler, parent = make()
    before = sampler.state_dict()
    parent.draw()
    corrupt = copy.deepcopy(before)
    corrupt['mixed'] = parent.state_dict()
    corrupt['mixed_state_sha256'] = sampler_state_sha256(corrupt['mixed'])
    corrupt['draws'] += 1
    with pytest.raises(ValueError, match='advanced before'):
        sampler.load_state_dict(corrupt)
    assert_state_equal(sampler.state_dict(), before)


def test_rng_checksum_corruption_is_rejected_after_coverage():
    sampler, parent = make()
    for _ in range(12):
        sampler.draw()
    before = sampler.state_dict()
    corrupt = copy.deepcopy(before)
    corrupt['mixed']['rng'] = parent.state_dict()['rng']
    with pytest.raises(ValueError, match='Saved mixed sampler changed'):
        sampler.load_state_dict(corrupt)
    assert_state_equal(sampler.state_dict(), before)


@pytest.mark.parametrize('split', ['val', 'test'])
def test_heldout_cohort_denied(split):
    cohort = MetadataCohort(split=split)
    parent = make()[1].state_dict()
    with pytest.raises(ValueError, match='TRAIN'):
        CoverageSampler(cohort, parent, expected_parent_sampler_sha256=sampler_state_sha256(parent))


def test_short_or_omitted_training_sequences_fail_closed():
    state = make()[1].state_dict()
    with pytest.raises(ValueError, match='cannot be completely covered'):
        CoverageSampler(MetadataCohort(((15,),)), state,
                        expected_parent_sampler_sha256=sampler_state_sha256(state))
    cohort = MetadataCohort()
    cohort.rows[('short-train', 'missing-sequence')] = dict(split='train', frames=4)
    with pytest.raises(ValueError, match='no eligible coverage window'):
        make(cohort)


@pytest.mark.parametrize('mutation', ['heldout', 'duplicate', 'source', 'length'])
def test_invalid_plan_membership_is_rejected(mutation):
    cohort = MetadataCohort()
    if mutation == 'heldout':
        cohort.sequences[0]['keys'] = [('heldout', 'reserved')]
    elif mutation == 'duplicate':
        cohort.sequences[1]['keys'] = cohort.sequences[0]['keys']
    elif mutation == 'source':
        cohort.sequences[0]['source_id'] = 'wrong-source'
    else:
        cohort.length = 8
    with pytest.raises(ValueError):
        make(cohort)


def test_parent_fingerprint_is_stable_and_binds_rng_draws_fraction():
    state = make()[1].state_dict()
    original = sampler_state_sha256(state)
    assert original == sampler_state_sha256(copy.deepcopy(state))
    for key, value in [('draws', state['draws'] + 1), ('fraction', .25),
                       ('rng', torch.Generator().manual_seed(99).get_state())]:
        changed = copy.deepcopy(state)
        changed[key] = value
        assert sampler_state_sha256(changed) != original
    state['draws'] = -1
    with pytest.raises(ValueError):
        sampler_state_sha256(state)
