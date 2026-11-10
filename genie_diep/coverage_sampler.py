"""One deterministic complete TRAIN-frame pass, then inherited mixed sampling.

This is an explicit data-schedule branch, not an exact continuation of the old
sampling schedule. Within every already checked Cohort sequence, 16-frame
windows start at 0, 16, ...; a final window ending at the last frame is appended
when necessary. A private CPU generator shuffles these windows once. No window
crosses a checked-sequence boundary. Short or omitted training sequences fail
closed rather than silently weakening the full-frame-coverage claim.

The parent's MixedSampler RNG is untouched during the coverage pass. Afterwards
its exact saved stream resumes. ``draws`` counts inherited plus new windows;
``coverage_unique_frames`` counts only frames visited by this new coverage pass,
not the parent's previous observations. State restoration verifies the immutable
plan/parent identities, phase, counters, prefix coverage and saved RNG checksum.
The enclosing experiment must additionally bind its parent checkpoint hashes.
No image data is read while constructing or inspecting this sampler.
"""
import hashlib
import json
import math

import torch

from .cohort import MixedSampler


_SCHEMA = 'train-frame-coverage-v1'


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'Invalid {name}')
    return value


def _mixed_state(state):
    if not isinstance(state, dict) or set(state) != {'rng', 'draws', 'fraction'}:
        raise ValueError('Invalid mixed sampler state fields')
    draws = _integer(state['draws'], 'mixed draw count')
    fraction = state['fraction']
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 <= fraction <= 1):
        raise ValueError('Invalid mixed sampler fraction')
    rng = state['rng']
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError('Invalid mixed sampler RNG')
    rng = rng.detach().cpu().clone()
    try:
        torch.Generator().set_state(rng)
    except (RuntimeError, TypeError) as error:
        raise ValueError('Invalid mixed sampler RNG') from error
    return dict(rng=rng, draws=draws, fraction=float(fraction))


def sampler_state_sha256(state):
    """Stable fingerprint of an existing MixedSampler state, without mutation."""
    state = _mixed_state(state)
    digest = hashlib.sha256()
    digest.update(json.dumps({'draws': state['draws'], 'fraction': state['fraction']},
                             sort_keys=True, separators=(',', ':')).encode('ascii'))
    digest.update(b'\0')
    digest.update(state['rng'].numpy().tobytes())
    return digest.hexdigest()


class CoverageSampler:
    def __init__(self, cohort, parent_mixed_state, *, seed=20261110,
                 expected_parent_sampler_sha256):
        if cohort.split != 'train' or cohort.length != 16:
            raise ValueError('Coverage sampler requires 16-frame TRAIN windows')
        _integer(seed, 'coverage seed')
        if seed >= 2**63:
            raise ValueError('Coverage seed exceeds supported range')
        parent = _mixed_state(parent_mixed_state)
        parent_sha = sampler_state_sha256(parent)
        if parent_sha != expected_parent_sampler_sha256:
            raise ValueError('Parent sampler identity changed')
        self.cohort = cohort
        self.seed = seed
        self.fraction = parent['fraction']
        self.parent_draws = parent['draws']
        self.parent_sampler_sha256 = parent_sha
        self._parent_state = parent
        self._mixed = MixedSampler(cohort, source_fraction=self.fraction)
        self._mixed.load_state_dict(parent)

        expected_keys = {key for key, row in cohort.rows.items() if row['split'] == 'train'}
        seen = set()
        metadata = []
        ordered = []
        self._frame_counts = []
        for index, sequence in enumerate(cohort.sequences):
            keys = sequence['keys']
            if not keys or any(key in seen or key not in expected_keys for key in keys):
                raise ValueError('Invalid or non-training coverage sequence membership')
            if len(set(keys)) != len(keys):
                raise ValueError('Repeated block in coverage sequence')
            seen.update(keys)
            counts = [_integer(cohort.rows[key]['frames'], 'block frame count', 1) for key in keys]
            count = sum(counts)
            if count < cohort.length or sequence['windows'] != count - cohort.length + 1:
                raise ValueError('Training sequence cannot be completely covered')
            if any(key[0] != sequence['source_id'] for key in keys):
                raise ValueError('Coverage sequence source mismatch')
            starts = list(range(0, count - cohort.length + 1, cohort.length))
            if starts[-1] != count - cohort.length:
                starts.append(count - cohort.length)
            ordered.extend((index, start) for start in starts)
            self._frame_counts.append(count)
            metadata.append(dict(source_id=sequence['source_id'], keys=keys, frames=counts))
        if not ordered or seen != expected_keys:
            raise ValueError('Some training frames have no eligible coverage window')
        permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(seed)).tolist()
        self._plan = tuple(ordered[index] for index in permutation)
        self._total_frames = sum(self._frame_counts)
        self._source_count = len({s['source_id'] for s in cohort.sequences})
        self.plan_sha256 = _sha(dict(schema=_SCHEMA, binding_sha256=cohort.binding_sha256,
                                     length=cohort.length, stride=cohort.length, seed=seed,
                                     sequences=metadata, windows=self._plan))
        self._identity = dict(schema=_SCHEMA, binding_sha256=cohort.binding_sha256,
                              plan_sha256=self.plan_sha256, seed=seed,
                              parent_sampler_sha256=parent_sha, parent_draws=self.parent_draws)
        self.coverage_index = 0
        self.draws = self.parent_draws
        self._coverage_cache = (0, 0)

    @property
    def phase(self):
        return 'coverage' if self.coverage_index < len(self._plan) else 'mixed'

    def draw(self):
        if self.phase == 'coverage':
            result = self._plan[self.coverage_index]
            self.coverage_index += 1
        else:
            result = self._mixed.draw()
        self.draws += 1
        return result

    def batch(self, count):
        _integer(count, 'batch size', 1)
        return torch.stack([self.cohort.window(*self.draw()) for _ in range(count)])

    def _unique_frames(self, index):
        if index == len(self._plan):
            return self._total_frames
        if self._coverage_cache[0] == index:
            return self._coverage_cache[1]
        # Metadata-only interval union; never decode a proxy to audit coverage.
        intervals = [[] for _ in self._frame_counts]
        for sequence, start in self._plan[:index]:
            intervals[sequence].append((start, start + self.cohort.length))
        total = 0
        for values in intervals:
            end = 0
            for start, stop in sorted(values):
                total += max(0, stop - max(start, end))
                end = max(end, stop)
        return total

    def plan_stats(self):
        """Metadata-only plan and cumulative coverage since the parent boundary."""
        unique = self._unique_frames(self.coverage_index)
        self._coverage_cache = (self.coverage_index, unique)
        return dict(**self._identity, total_training_frames=self._total_frames,
                    training_sources=self._source_count, training_sequences=len(self._frame_counts),
                    coverage_windows=len(self._plan), coverage_frame_presentations=len(self._plan) * 16,
                    phase=self.phase, coverage_index=self.coverage_index,
                    coverage_unique_frames=unique, full_coverage=self.coverage_index == len(self._plan),
                    draws=self.draws, post_parent_draws=self.draws - self.parent_draws,
                    post_parent_frame_presentations=(self.draws - self.parent_draws) * 16,
                    fallback_draws=self._mixed.draws - self.parent_draws)

    def state_dict(self):
        stats = self.plan_stats()
        mixed = self._mixed.state_dict()
        return dict(**self._identity, phase=self.phase, coverage_index=self.coverage_index,
                    coverage_unique_frames=stats['coverage_unique_frames'], draws=self.draws,
                    mixed=mixed, mixed_state_sha256=sampler_state_sha256(mixed))

    def load_state_dict(self, state):
        """Validate everything before mutation, including coverage-to-mixed transition."""
        required = set(self._identity) | {'phase', 'coverage_index', 'coverage_unique_frames',
                                           'draws', 'mixed', 'mixed_state_sha256'}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError('Invalid coverage sampler state fields')
        if any(state[key] != value for key, value in self._identity.items()):
            raise ValueError('Coverage sampler identity changed')
        index = _integer(state['coverage_index'], 'coverage index')
        draws = _integer(state['draws'], 'total draw count')
        unique = _integer(state['coverage_unique_frames'], 'unique frame count')
        if index > len(self._plan):
            raise ValueError('Coverage index exceeds plan')
        phase = 'coverage' if index < len(self._plan) else 'mixed'
        if state['phase'] != phase:
            raise ValueError('Premature coverage phase transition')
        mixed = _mixed_state(state['mixed'])
        mixed_sha = sampler_state_sha256(mixed)
        if mixed_sha != state['mixed_state_sha256'] or mixed['fraction'] != self.fraction:
            raise ValueError('Saved mixed sampler changed')
        fallback = mixed['draws'] - self.parent_draws
        if fallback < 0 or draws != self.parent_draws + index + fallback:
            raise ValueError('Inconsistent sampler draw counters')
        if phase == 'coverage' and mixed_sha != self.parent_sampler_sha256:
            raise ValueError('Inherited RNG advanced before complete coverage')
        if unique != self._unique_frames(index):
            raise ValueError('Saved coverage evidence disagrees with plan prefix')
        self._mixed.load_state_dict(mixed)
        self.coverage_index = index
        self.draws = draws
        self._coverage_cache = (index, unique)
