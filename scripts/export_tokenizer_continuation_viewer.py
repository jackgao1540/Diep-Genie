"""Export a private matched continuation comparison without changing older gates.

Use the existing CPU quality evaluator first. This script only verifies saved
evidence and encodes local viewer images; it performs no inference or training.
Run the separate continuation artifact audit for optimizer, budget and frozen-
module evidence; this viewer receipt does not replace that audit.
"""
import argparse
import json
from pathlib import Path

import torch

from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from scripts.evaluate_tokenizer_quality import validate_state
from scripts.export_reconstruction_viewer import private_bundle, sha_bytes
from scripts.export_tokenizer_comparison_viewer import ARMS, LABELS, private_destination
from scripts.run_tokenizer_comparison import compare_exact
from scripts.run_tokenizer_continuation import validate_experiment, validate_parent
from scripts.validate_tokenizer_continuation_artifacts import expected_contract, replay_sampling


def checkpoint_path(root, metadata):
    path = (root/metadata['path']).resolve()
    if not path.is_relative_to(root/'runs') or path.suffix != '.pt':
        raise ValueError('Continuation checkpoints must be preserved under project runs')
    if file_hash(path) != metadata['sha256']:
        raise ValueError('Quality checkpoint hash mismatch')
    return path


def validate_inputs(root, quality_dir, experiment_path):
    root = Path(root).resolve()
    quality_dir = (root/quality_dir).resolve()
    experiment_path = (root/experiment_path).resolve()
    report_path = quality_dir/'quality-report.json'
    inputs = {report_path: file_hash(report_path), experiment_path: file_hash(experiment_path)}
    plan = validate_experiment(root, experiment_path)
    prior_path = root/plan['parent_experiment']['path']
    inputs[prior_path] = plan['parent_experiment']['sha256']
    prior = read(prior_path)
    report = read(report_path)
    if (report.get('updates') != 0 or report.get('windows') != 64 or report.get('sources') != 16
            or report.get('frames') != 1024 or set(report.get('checkpoints', {})) != set(ARMS)):
        raise ValueError('Expected a complete read-only 64-window continuation comparison')
    if report.get('model_implementation_sha256') != prior['source_hashes']['model_sha256']:
        raise ValueError('Quality evaluation model source differs from the parent experiment')
    steps = [report['checkpoints'][label]['step'] for label in ARMS]
    if any(type(step) is not int or not 999 <= step <= 5000 for step in steps) or len(set(steps)) != 1:
        raise ValueError('Continuation comparison requires one common step from 999 through 5000')
    step = steps[0]
    expected = {}
    coverage = {}
    first_sampler = None
    for label, arm in ARMS.items():
        entry = prior['arms'][arm]
        spec_path = root/entry['spec']
        inputs[spec_path] = entry['spec_sha256']
        if file_hash(spec_path) != inputs[spec_path]:
            raise ValueError('Parent comparison specification changed')
        spec = read(spec_path)
        binding_path = root/spec['data_binding']
        selection_path = root/spec['evaluation_selection']
        inputs[binding_path] = file_hash(binding_path)
        inputs[selection_path] = file_hash(selection_path)
        selection = read(selection_path)
        if (selection['binding_sha256'] != inputs[binding_path] or report['binding_sha256'] != inputs[binding_path]
                or report['selection_sha256'] != inputs[selection_path]
                or [row['selection'] for row in report['records']] != selection['validation']
                or len(selection['validation']) != 64
                or len({row['source_id'] for row in selection['validation']}) != 16):
            raise ValueError('Continuation quality differs from the frozen validation selection')
        metadata = report['checkpoints'][label]
        if metadata.get('preflight') is not False:
            raise ValueError('This exporter requires the production research continuation, not CPU preflight weights')
        path = checkpoint_path(root, metadata)
        inputs[path] = metadata['sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_state(state, spec, inputs[binding_path], inputs[selection_path])
        parent_path = root/plan['parents'][arm]['path']
        inputs[parent_path] = plan['parents'][arm]['sha256']
        if file_hash(parent_path) != inputs[parent_path]:
            raise ValueError('Preserved continuation parent changed')
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
        validate_parent(parent, plan, arm, prior, root)
        if state['step'] != step or state['contract'] != expected_contract(parent, plan, inputs[experiment_path], arm):
            raise ValueError('Continuation checkpoint step, arm, parent or experiment contract differs')
        if state.get('last_failure') is not None:
            raise ValueError('Continuation checkpoint retains a failed update')
        # Independent replay uses checked training metadata only. It verifies
        # the complete coverage prefix and inherited mixed stream at this step.
        data = Cohort(root, spec['data_binding'], 'train', spec['model']['context'])
        states, stats = replay_sampling(data, parent['sampler'], plan['coverage_seed'], {step})
        compare_exact(state['sampler'], states[step])
        if first_sampler is None:
            first_sampler = state['sampler']
        else:
            compare_exact(state['sampler'], first_sampler)
        coverage[arm] = stats[step]
        if step == 5000 and not coverage[arm]['full_coverage']:
            raise ValueError('Final checkpoint did not complete the bound training-frame coverage pass')
        expected[label] = {key: metadata[key] for key in ('sha256', 'step', 'preflight')}
    for index, row in enumerate(report['records']):
        if row['index'] != index or row['artifact']['path'] != f'reconstruction-{index:03}.npz':
            raise ValueError('Unexpected continuation quality artifact ordering or path')
        path = quality_dir/row['artifact']['path']
        inputs[path] = row['artifact']['sha256']
        if file_hash(path) != inputs[path]:
            raise ValueError('Held-out reconstruction hash mismatch')
    return quality_dir, report, expected, inputs, coverage, plan


def run(args):
    root = Path(args.root).resolve()
    output, receipt_path = private_destination(root, root/args.out)
    quality_dir, report, expected, inputs, coverage, plan = validate_inputs(root, args.quality_dir, args.experiment)
    bundle = private_bundle(quality_dir, expected)
    if bundle['quality_report_sha256'] != inputs[quality_dir/'quality-report.json']:
        raise ValueError('Quality report changed during export')
    step = expected['before']['step']
    bundle['title'] = f'Matched tokenizer continuation · step {step}'
    bundle['experiment_sha256'] = inputs[(root/args.experiment).resolve()]
    bundle['parent_experiment_sha256'] = plan['parent_experiment']['sha256']
    bundle['interpretation'] = 'Baseline and detail-loss arms under the same coverage schedule; labels and coverage do not imply a quality winner.'
    bundle['checkpoints'] = {label: {**metadata, 'label': LABELS[label]} for label, metadata in bundle['checkpoints'].items()}
    encoded = (json.dumps(bundle, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(encoded) > 200*1024*1024:
        raise ValueError('Private continuation exceeds viewer import cap')
    for path, digest in inputs.items():
        if file_hash(path) != digest:
            raise ValueError('Bound continuation input changed during export: ' + path.name)
    receipt = dict(schema_version=1, private_bundle=dict(path=output.name, bytes=len(encoded), sha256=sha_bytes(encoded)),
        experiment_sha256=bundle['experiment_sha256'], parent_experiment_sha256=bundle['parent_experiment_sha256'],
        quality_report_sha256=bundle['quality_report_sha256'], selection_sha256=report['selection_sha256'],
        binding_sha256=report['binding_sha256'], model_implementation_sha256=report['model_implementation_sha256'],
        parent_step=999, common_step=step, maximum_common_step=5000, checkpoints=expected,
        parent_checkpoint_sha256={arm: entry['sha256'] for arm, entry in plan['parents'].items()},
        coverage_replayed_from_metadata=True, coverage=coverage,
        exporter_sha256=file_hash(__file__), parent_exporter_sha256=file_hash(Path(__file__).with_name('export_reconstruction_viewer.py')),
        sampling_validator_sha256=file_hash(Path(__file__).with_name('validate_tokenizer_continuation_artifacts.py')),
        optimizer_updates=0, inference_performed=False, full_training_artifact_audit_performed=False,
        checkpoint_files_unchanged=True, quality_inputs_unchanged=True,
        public_media_written=False, private_windows=64, private_sources=16,
        interpretation='Private continuation comparison only; coverage is not a quality gate or promotion.')
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
