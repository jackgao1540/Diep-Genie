"""Private continuation export provenance; image encoding has separate tests."""
import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from genie_diep.data import file_hash
from genie_diep.cohort import read
import scripts.export_tokenizer_continuation_viewer as exporter
import scripts.export_tokenizer_comparison_viewer as original_exporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class MetadataOnlyCohort:
    """A small checked sequence; decoding during replay must fail the test."""
    def __init__(self, root, binding, split, length):
        assert split == 'train' and length == 16
        self.split, self.length = split, length
        self.binding_sha256 = file_hash(root/binding)
        self.rows = {('training', '0'): dict(split='train', frames=64),
                     ('heldout', '0'): dict(split='test', frames=64)}
        self.sequences = [dict(source_id='training', keys=[('training', '0')], windows=49)]
        self.sources = ['training']
        self.by_source = {'training': [0]}
        self.cumulative = [49]
        self.source_cumulative = {'training': [49]}

    def window(self, *_):
        raise AssertionError('Exporter must not decode training or test frames')


def stub_bundle(quality_dir, expected):
    assert all(set(value) == {'sha256', 'step', 'preflight'} for value in expected.values())
    return dict(schema_version=1, kind='diep-genie-reconstructions', mode='local',
        checkpoints=copy.deepcopy(expected), quality_report_sha256=file_hash(Path(quality_dir)/'quality-report.json'), clips=[])


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path
    binding = root/'configs/binding.json'
    write(binding, {'fixture': 'metadata only'})
    selections = [dict(source_id=f'source-{i % 16:02}', first_block='0', start_frame=i, length=16) for i in range(64)]
    selection_path = root/'configs/selection.json'
    write(selection_path, dict(binding_sha256=file_hash(binding), validation=selections))
    spec = dict(model={'size': 128, 'context': 16}, data_binding='configs/binding.json', evaluation_selection='configs/selection.json')
    prior = dict(source_hashes={'implementation_sha256': 'a'*64, 'model_sha256': 'b'*64},
                 limits={'unique_updates_per_arm': 999}, arms={})
    for arm in exporter.ARMS.values():
        path = root/f'configs/{arm}.json'
        write(path, spec)
        prior['arms'][arm] = dict(spec=f'configs/{arm}.json', spec_sha256=file_hash(path), objective={'name': arm})
    prior_path = root/'configs/parent-experiment.json'
    write(prior_path, prior)
    plan = dict(parent_step=999, target_step=5000, coverage_seed=20261110,
                sampling_policy='full_train_frame_coverage_then_parent_mixed',
                parent_experiment=dict(path='configs/parent-experiment.json', sha256=file_hash(prior_path)),
                source_hashes={'runner_sha256': 'c'*64, 'coverage_sampler_sha256': 'd'*64}, parents={})
    parents = {}
    for arm, entry in prior['arms'].items():
        parent = dict(step=999, config=spec['model'], model={'weights': torch.zeros(1)},
            sampler=dict(draws=15985, fraction=.5, rng=torch.Generator().manual_seed(42).get_state()),
            scheduler={'last_epoch': 999}, scaler={'scale': 1024.},
            contract=dict(stage='tokenizer', binding_sha256=file_hash(binding), evaluation_sha256=file_hash(selection_path),
                preflight=False, microbatch=4, accumulation=4, precision='fp16', spec_sha256=entry['spec_sha256'],
                implementation_sha256='a'*64, comparison=dict(experiment_sha256=file_hash(prior_path), arm=arm,
                    objective=entry['objective'], source_hashes=prior['source_hashes'])))
        path = root/f'continuation-inputs/{arm}-step999.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(parent, path)
        plan['parents'][arm] = dict(path=path.relative_to(root).as_posix(), sha256=file_hash(path))
        parents[arm] = parent
    experiment_path = root/'configs/continuation.json'
    write(experiment_path, plan)
    # Frozen experiment source/config validation is covered by runner tests.
    # Keep its parent validator and the real independent sampling replay here.
    monkeypatch.setattr(exporter, 'validate_experiment', lambda root, path: read(path))
    monkeypatch.setattr(exporter, 'Cohort', MetadataOnlyCohort)
    monkeypatch.setattr(exporter, 'private_bundle', stub_bundle)
    cohort = MetadataOnlyCohort(root, spec['data_binding'], 'train', 16)
    samplers, _ = exporter.replay_sampling(cohort, parents['baseline']['sampler'], plan['coverage_seed'], {1001})
    checkpoints = {}
    for label, arm in exporter.ARMS.items():
        state = copy.deepcopy(parents[arm])
        state.update(step=1001, sampler=samplers[1001],
                     contract=exporter.expected_contract(parents[arm], plan, file_hash(experiment_path), arm))
        path = root/f'runs/continuation/{arm}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        checkpoints[label] = dict(path=str(path), sha256=file_hash(path), step=1001, preflight=False)
    quality = root/'runs/quality'
    quality.mkdir(parents=True)
    rows = []
    for index, selection in enumerate(selections):
        path = quality/f'reconstruction-{index:03}.npz'
        path.write_bytes(b'fixture artifact; established encoder validates arrays')
        rows.append(dict(index=index, selection=selection, artifact=dict(path=path.name, sha256=file_hash(path))))
    write(quality/'quality-report.json', dict(updates=0, frames=1024, windows=64, sources=16,
        model_implementation_sha256='b'*64, binding_sha256=file_hash(binding), selection_sha256=file_hash(selection_path),
        records=rows, checkpoints=checkpoints))
    return root, quality, experiment_path


def args(project):
    root, quality, experiment = project
    return Namespace(root=str(root), quality_dir=str(quality), experiment=str(experiment), out='exports/continuation.json')


def edit_report(project, fn):
    path = project[1]/'quality-report.json'
    report = read(path)
    fn(report)
    write(path, report)


def edit_checkpoint(project, label, fn):
    report_path = project[1]/'quality-report.json'
    report = read(report_path)
    path = Path(report['checkpoints'][label]['path'])
    state = torch.load(path, weights_only=True)
    fn(state)
    torch.save(state, path)
    report['checkpoints'][label]['sha256'] = file_hash(path)
    write(report_path, report)


def change_common_boundary(project, step):
    root, _, experiment = project
    plan = read(experiment)
    parent = torch.load(root/plan['parents']['baseline']['path'], weights_only=True)
    cohort = MetadataOnlyCohort(root, 'configs/binding.json', 'train', 16)
    states, _ = exporter.replay_sampling(cohort, parent['sampler'], plan['coverage_seed'], {step})
    for label in exporter.ARMS:
        edit_checkpoint(project, label, lambda state: state.update(step=step, sampler=states[step]))
    edit_report(project, lambda report: [metadata.update(step=step) for metadata in report['checkpoints'].values()])


@pytest.mark.parametrize('step', [999, 1001, 5000])
def test_exact_parent_or_partial_or_final_boundary_has_private_labels_and_provenance(project, step):
    change_common_boundary(project, step)
    root, quality, experiment = project
    public = root/'docs/reconstructions.json'
    write(public, {'preserved': 'original public comparison'})
    original = public.read_bytes()
    receipt = exporter.run(args(project))
    result = read(root/'exports/continuation.json')
    assert result['title'] == f'Matched tokenizer continuation · step {step}'
    assert result['checkpoints']['before']['label'] == 'Baseline'
    assert result['checkpoints']['after']['label'] == 'Detail loss'
    assert receipt['common_step'] == step and receipt['parent_step'] == 999
    assert receipt['coverage_replayed_from_metadata']
    assert receipt['coverage']['baseline']['post_parent_draws'] == (step-999)*16
    assert receipt['coverage']['baseline']['full_coverage'] is (step != 999)
    assert receipt['experiment_sha256'] == file_hash(experiment)
    assert receipt['quality_report_sha256'] == file_hash(quality/'quality-report.json')
    assert receipt['optimizer_updates'] == 0 and not receipt['inference_performed'] and not receipt['public_media_written']
    assert receipt['full_training_artifact_audit_performed'] is False
    assert public.read_bytes() == original
    with pytest.raises(FileExistsError):
        exporter.run(args(project))


@pytest.mark.parametrize('mutate,message', [
    (lambda s: s['contract']['comparison'].update(arm='baseline'), 'contract differs'),
    (lambda s: s['contract']['continuation'].update(experiment_sha256='0'*64), 'contract differs'),
    (lambda s: s['contract']['continuation'].update(parent_sha256='0'*64), 'contract differs'),
    (lambda s: s['contract']['continuation'].update(coverage_seed=42), 'contract differs'),
    (lambda s: s['sampler'].update(draws=123), 'Scalar state changed'),
    (lambda s: s.update(step=1000), 'contract differs'),
    (lambda s: s.update(last_failure={'reason': 'failed'}), 'failed update'),
])
def test_wrong_contract_sampler_or_boundary_fails_before_writes(project, mutate, message):
    edit_checkpoint(project, 'after', mutate)
    with pytest.raises(ValueError, match=message):
        exporter.run(args(project))
    assert not (project[0]/'exports').exists()


@pytest.mark.parametrize('step', [998, 5001, True])
def test_original_continuation_step_bounds_cannot_expand(project, step):
    edit_report(project, lambda r: [metadata.update(step=step) for metadata in r['checkpoints'].values()])
    with pytest.raises(ValueError, match='common step'):
        exporter.run(args(project))


def test_preflight_and_unmatched_steps_are_rejected(project):
    edit_report(project, lambda r: r['checkpoints']['after'].update(preflight=True))
    with pytest.raises(ValueError, match='production research'):
        exporter.run(args(project))
    edit_report(project, lambda r: r['checkpoints']['after'].update(preflight=False, step=1000))
    with pytest.raises(ValueError, match='common step'):
        exporter.run(args(project))


def test_parent_hash_and_checkpoint_hash_are_required(project):
    plan = read(project[2])
    (project[0]/plan['parents']['baseline']['path']).write_bytes(b'changed parent')
    with pytest.raises(ValueError, match='parent changed'):
        exporter.run(args(project))


def test_frozen_descriptors_and_npz_are_required(project):
    edit_report(project, lambda r: r['records'][0]['selection'].update(start_frame=999))
    with pytest.raises(ValueError, match='frozen validation'):
        exporter.run(args(project))
    edit_report(project, lambda r: r['records'][0]['selection'].update(start_frame=0))
    (project[1]/'reconstruction-000.npz').write_bytes(b'changed artifact')
    with pytest.raises(ValueError, match='reconstruction hash'):
        exporter.run(args(project))


def test_input_change_during_encoding_fails_before_output(project, monkeypatch):
    def mutate(quality, expected):
        bundle = stub_bundle(quality, expected)
        report = Path(quality)/'quality-report.json'
        report.write_text(report.read_text()+' ')
        return bundle
    monkeypatch.setattr(exporter, 'private_bundle', mutate)
    with pytest.raises(ValueError, match='changed during export'):
        exporter.run(args(project))
    assert not (project[0]/'exports').exists()


@pytest.mark.parametrize('destination', ['docs/private.json', '../private.json'])
def test_public_or_external_output_is_rejected(project, destination):
    arguments = args(project)
    arguments.out = destination
    with pytest.raises(ValueError, match='under project exports'):
        exporter.run(arguments)


def test_original_comparison_exporter_retains_999_cap(project):
    arguments = args(project)
    arguments.experiment = str(project[0]/'configs/parent-experiment.json')
    with pytest.raises(ValueError, match='Checkpoint step or preflight metadata differs'):
        original_exporter.run(arguments)
