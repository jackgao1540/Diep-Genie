import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from genie_diep.data import file_hash
import scripts.export_tokenizer_comparison_viewer as exporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def project(tmp_path):
    root = tmp_path
    binding = root/'configs/binding.json'
    write(binding, {'cohort': 'frozen'})
    selections = [dict(source_id=f'source-{index % 16:02}', first_block='0', start_frame=index, length=16) for index in range(64)]
    selection_path = root/'configs/selection.json'
    write(selection_path, dict(binding_sha256=file_hash(binding), validation=selections))
    spec = dict(model={'size': 128}, data_binding='configs/binding.json', evaluation_selection='configs/selection.json')
    experiment = dict(source_hashes={'implementation_sha256': 'a'*64, 'model_sha256': 'b'*64}, limits={'unique_updates_per_arm': 999}, arms={})
    for arm in ('baseline', 'detail'):
        path = root/f'configs/{arm}.json'
        write(path, spec)
        experiment['arms'][arm] = dict(spec=f'configs/{arm}.json', spec_sha256=file_hash(path), objective={'name': arm})
    experiment_path = root/'configs/experiment.json'
    write(experiment_path, experiment)
    checkpoints = {}
    for label, arm in exporter.ARMS.items():
        state = dict(step=999, config=spec['model'], model={'weights': torch.zeros(1)},
            sampler=dict(draws=15985, fraction=.5, rng=torch.tensor([1, 2], dtype=torch.uint8)),
            contract=dict(stage='tokenizer', binding_sha256=file_hash(binding), evaluation_sha256=file_hash(selection_path),
                preflight=False, spec_sha256=experiment['arms'][arm]['spec_sha256'], implementation_sha256='a'*64,
                comparison=dict(experiment_sha256=file_hash(experiment_path), arm=arm,
                    objective=experiment['arms'][arm]['objective'], source_hashes=experiment['source_hashes'])))
        path = root/f'runs/comparison/{arm}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        checkpoints[label] = dict(path=str(path), sha256=file_hash(path), step=999, preflight=False)
    quality = root/'runs/quality'
    quality.mkdir(parents=True)
    rows = []
    for index, selection in enumerate(selections):
        path = quality/f'reconstruction-{index:03}.npz'
        path.write_bytes(b'fixture artifact; established exporter handles array validation')
        rows.append(dict(index=index, selection=selection, artifact=dict(path=path.name, sha256=file_hash(path))))
    report = dict(updates=0, frames=1024, windows=64, sources=16, model_implementation_sha256='b'*64, binding_sha256=file_hash(binding),
        selection_sha256=file_hash(selection_path), records=rows, checkpoints=checkpoints)
    write(quality/'quality-report.json', report)
    return root, quality, experiment_path


def stub_bundle(quality_dir, expected):
    assert all(set(value) == {'sha256', 'step', 'preflight'} for value in expected.values())
    return dict(schema_version=1, kind='diep-genie-reconstructions', mode='local', checkpoints=copy.deepcopy(expected),
                quality_report_sha256=file_hash(Path(quality_dir)/'quality-report.json'), clips=[])


def args(project, name='paired.json'):
    root, quality, experiment = project
    return Namespace(root=str(root), quality_dir=str(quality), experiment=str(experiment), out=f'exports/{name}')


def edit_report(project, fn):
    path = project[1]/'quality-report.json'
    report = json.loads(path.read_text())
    fn(report)
    write(path, report)


def edit_checkpoint(project, label, fn):
    report_path = project[1]/'quality-report.json'
    report = json.loads(report_path.read_text())
    path = Path(report['checkpoints'][label]['path'])
    state = torch.load(path, weights_only=True)
    fn(state)
    torch.save(state, path)
    report['checkpoints'][label]['sha256'] = file_hash(path)
    write(report_path, report)


def test_private_paired_export_labels_after_validation_and_preserves_inputs(project, monkeypatch):
    monkeypatch.setattr(exporter, 'private_bundle', stub_bundle)
    root, quality, experiment = project
    public = root/'docs/reconstructions.json'
    write(public, {'preserved': 'original 219 / 518 public comparison'})
    original = public.read_bytes()
    receipt = exporter.run(args(project))
    result = json.loads((root/'exports/paired.json').read_text(encoding='utf-8'))
    assert result['title'] == 'Matched tokenizer comparison \u00b7 step 999'
    assert result['checkpoints']['before']['label'] == 'Baseline'
    assert result['checkpoints']['after']['label'] == 'Detail loss'
    assert receipt['quality_report_sha256'] == file_hash(quality/'quality-report.json')
    assert receipt['experiment_sha256'] == file_hash(experiment)
    assert receipt['optimizer_updates'] == 0 and not receipt['inference_performed'] and not receipt['public_media_written']
    assert public.read_bytes() == original
    with pytest.raises(FileExistsError):
        exporter.run(args(project))


def test_common_partial_boundary_is_supported(project, monkeypatch):
    monkeypatch.setattr(exporter, 'private_bundle', stub_bundle)
    for label in exporter.ARMS:
        edit_checkpoint(project, label, lambda state: state.update(step=250))
    edit_report(project, lambda report: [metadata.update(step=250) for metadata in report['checkpoints'].values()])
    assert exporter.run(args(project))['common_step'] == 250


@pytest.mark.parametrize('mutate,message', [
    (lambda state: state['contract']['comparison'].update(arm='baseline'), 'arm or experiment'),
    (lambda state: state['contract']['comparison'].update(experiment_sha256='0'*64), 'arm or experiment'),
    (lambda state: state['sampler'].update(draws=123), 'samplers'),
    (lambda state: state.update(step=750), 'step or preflight'),
])
def test_mismatched_checkpoint_contract_or_boundary_is_rejected(project, mutate, message):
    edit_checkpoint(project, 'after', mutate)
    with pytest.raises(ValueError, match=message):
        exporter.run(args(project))
    assert not (project[0]/'exports').exists()


def test_frozen_descriptors_and_artifact_hashes_are_required(project):
    edit_report(project, lambda report: report['records'][0]['selection'].update(start_frame=999))
    with pytest.raises(ValueError, match='frozen validation'):
        exporter.run(args(project))


def test_changed_npz_is_rejected(project):
    (project[1]/'reconstruction-000.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='reconstruction hash'):
        exporter.run(args(project))


def test_input_change_during_encoding_is_rejected_before_output(project, monkeypatch):
    def mutate(quality, expected):
        bundle = stub_bundle(quality, expected)
        report_path = Path(quality)/'quality-report.json'
        report_path.write_text(report_path.read_text() + ' ')
        return bundle
    monkeypatch.setattr(exporter, 'private_bundle', mutate)
    with pytest.raises(ValueError, match='changed during export'):
        exporter.run(args(project))
    assert not (project[0]/'exports').exists()


def test_private_output_cannot_use_public_or_external_directory(project):
    arguments = args(project)
    arguments.out = 'docs/paired.json'
    with pytest.raises(ValueError, match='under project exports'):
        exporter.run(arguments)
    arguments.out = '../paired.json'
    with pytest.raises(ValueError, match='under project exports'):
        exporter.run(arguments)
