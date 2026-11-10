"""Export a private, clearly labeled matched tokenizer comparison for local import.

Consumes previously saved quality evidence only. No inference, training, uploads
or writes to public media/manifests are performed.
"""
import argparse
import json
from pathlib import Path

import torch

from genie_diep.cohort import read
from genie_diep.data import file_hash
from scripts.evaluate_tokenizer_quality import validate_state
from scripts.export_reconstruction_viewer import private_bundle, sha_bytes


ARMS = {'before': 'baseline', 'after': 'detail'}
LABELS = {'before': 'Baseline', 'after': 'Detail loss'}


def private_destination(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != '.json' or not output.is_relative_to(root/'exports'):
        raise ValueError('Private comparison must be a JSON file under project exports')
    receipt = output.with_suffix('.receipt.json')
    if output.exists() or receipt.exists():
        raise FileExistsError('Preserve completed viewer exports; choose a new output')
    return output, receipt


def validate_inputs(root, quality_dir, experiment_path):
    root = Path(root).resolve()
    quality_dir = (root/quality_dir).resolve()
    experiment_path = (root/experiment_path).resolve()
    report_path = quality_dir/'quality-report.json'
    inputs = {report_path: file_hash(report_path), experiment_path: file_hash(experiment_path)}
    report = read(report_path)
    experiment = read(experiment_path)
    if report.get('updates') != 0 or report.get('windows') != 64 or report.get('sources') != 16 or report.get('frames') != 1024:
        raise ValueError('Expected a complete read-only 64-window quality comparison')
    if set(report.get('checkpoints', {})) != set(ARMS) or set(experiment.get('arms', {})) != set(ARMS.values()):
        raise ValueError('Expected exactly baseline and detail checkpoints')
    if experiment['limits']['unique_updates_per_arm'] != 999:
        raise ValueError('Unexpected comparison update cap')
    if report.get('model_implementation_sha256') != experiment['source_hashes']['model_sha256']:
        raise ValueError('Quality evaluation model source differs from the comparison')
    expected = {}
    checkpoint_paths = {}
    first_sampler = None
    for label, arm in ARMS.items():
        entry = experiment['arms'][arm]
        spec_path = (root/entry['spec']).resolve()
        inputs[spec_path] = entry['spec_sha256']
        if file_hash(spec_path) != inputs[spec_path]:
            raise ValueError('Comparison specification changed')
        spec = read(spec_path)
        binding_path = (root/spec['data_binding']).resolve()
        selection_path = (root/spec['evaluation_selection']).resolve()
        inputs[binding_path] = file_hash(binding_path)
        inputs[selection_path] = file_hash(selection_path)
        selection = read(selection_path)
        if (selection['binding_sha256'] != inputs[binding_path] or report['binding_sha256'] != inputs[binding_path]
                or report['selection_sha256'] != inputs[selection_path]
                or [row['selection'] for row in report['records']] != selection['validation']
                or len(selection['validation']) != 64
                or len({row['source_id'] for row in selection['validation']}) != 16):
            raise ValueError('Quality evidence differs from the frozen validation selection')
        metadata = report['checkpoints'][label]
        path = (root/metadata['path']).resolve()
        if not path.is_relative_to(root/'runs') or path.suffix != '.pt':
            raise ValueError('Comparison checkpoints must be preserved under project runs')
        if file_hash(path) != metadata['sha256']:
            raise ValueError('Quality checkpoint hash mismatch')
        inputs[path] = metadata['sha256']
        checkpoint_paths[label] = path
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_state(state, spec, inputs[binding_path], inputs[selection_path])
        contract = state['contract']
        expected_comparison = dict(experiment_sha256=inputs[experiment_path], arm=arm,
                                   objective=entry['objective'], source_hashes=experiment['source_hashes'])
        if contract.get('comparison') != expected_comparison:
            raise ValueError('Checkpoint comparison arm or experiment contract mismatch')
        if contract.get('spec_sha256') != entry['spec_sha256']:
            raise ValueError('Checkpoint specification contract mismatch')
        if contract.get('implementation_sha256') != experiment['source_hashes']['implementation_sha256']:
            raise ValueError('Checkpoint implementation contract mismatch')
        sampler = state['sampler']
        if first_sampler is None:
            first_sampler = sampler
        elif (sampler['draws'] != first_sampler['draws'] or sampler['fraction'] != first_sampler['fraction']
                or not torch.equal(sampler['rng'], first_sampler['rng'])):
            raise ValueError('Comparison checkpoint samplers are not at the same boundary')
        if (type(metadata.get('step')) is not int or not 1 <= metadata['step'] <= 999
                or type(metadata.get('preflight')) is not bool
                or state['step'] != metadata['step'] or contract['preflight'] != metadata['preflight']):
            raise ValueError('Checkpoint step or preflight metadata differs')
        expected[label] = {key: metadata[key] for key in ('sha256', 'step', 'preflight')}
    if expected['before']['step'] != expected['after']['step']:
        raise ValueError('Compare only a common saved checkpoint step')
    if expected['before']['preflight'] != expected['after']['preflight']:
        raise ValueError('Comparison checkpoints use different execution modes')
    for index, row in enumerate(report['records']):
        if row['index'] != index or row['artifact']['path'] != f'reconstruction-{index:03}.npz':
            raise ValueError('Unexpected quality artifact ordering or path')
        path = quality_dir/row['artifact']['path']
        inputs[path] = row['artifact']['sha256']
        if file_hash(path) != inputs[path]:
            raise ValueError('Held-out reconstruction hash mismatch')
    return quality_dir, report, expected, checkpoint_paths, inputs


def run(args):
    root = Path(args.root).resolve()
    output, receipt_path = private_destination(root, root/args.out)
    quality_dir, report, expected, paths, inputs = validate_inputs(root, args.quality_dir, args.experiment)
    # This established exporter validates NPZ array names, shape, dtype and
    # lossless atlas pixels. Labels are deliberately added only afterward.
    bundle = private_bundle(quality_dir, expected)
    if bundle['quality_report_sha256'] != inputs[quality_dir/'quality-report.json']:
        raise ValueError('Quality report changed during export')
    step = expected['before']['step']
    bundle['title'] = f'Matched tokenizer comparison · step {step}'
    bundle['experiment_sha256'] = inputs[(root/args.experiment).resolve()]
    bundle['interpretation'] = 'Baseline and detail-loss arms at the same saved step; labels do not imply a quality winner.'
    bundle['checkpoints'] = {label: {**metadata, 'label': LABELS[label]}
                             for label, metadata in bundle['checkpoints'].items()}
    encoded = (json.dumps(bundle, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(encoded) > 200*1024*1024:
        raise ValueError('Private comparison exceeds viewer import cap')
    for path, digest in inputs.items():
        if file_hash(path) != digest:
            raise ValueError('Bound comparison input changed during export: ' + path.name)
    receipt = dict(schema_version=1, private_bundle=dict(path=output.name, bytes=len(encoded), sha256=sha_bytes(encoded)),
        experiment_sha256=bundle['experiment_sha256'], quality_report_sha256=bundle['quality_report_sha256'],
        selection_sha256=report['selection_sha256'], binding_sha256=report['binding_sha256'],
        model_implementation_sha256=report['model_implementation_sha256'],
        common_step=step, maximum_common_step=999, checkpoints=expected,
        exporter_sha256=file_hash(__file__), parent_exporter_sha256=file_hash(Path(__file__).with_name('export_reconstruction_viewer.py')),
        optimizer_updates=0, inference_performed=False, checkpoint_files_unchanged=True,
        quality_inputs_unchanged=True, public_media_written=False, private_windows=64, private_sources=16,
        interpretation='Local visual comparison only; no declared quality winner or promotion.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded)
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--quality-dir', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
