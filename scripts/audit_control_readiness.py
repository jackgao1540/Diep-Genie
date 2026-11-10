"""Read-only CPU audit of control readiness for tokenizer-stage checkpoints.

Comparing unchanged action/dynamics tensors is an eligibility check, not an
intervention result. No model, optimizer, commands, or rollouts are executed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def load_bound_checkpoint(path):
    path = Path(path)
    before = sha256(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if sha256(path) != before:
        raise ValueError('Checkpoint changed while being read')
    return state, before


def validate_checkpoint(state, config, binding_sha256, evaluation_sha256):
    if not isinstance(state, dict) or not isinstance(state.get('contract'), dict):
        raise ValueError('Checkpoint contract is missing')
    contract = state['contract']
    if contract.get('stage') != 'tokenizer':
        raise ValueError('This audit requires tokenizer-stage checkpoints')
    if type(contract.get('preflight')) is not bool:
        raise ValueError('Checkpoint preflight status is missing')
    if normalized(state.get('config')) != normalized(config):
        raise ValueError('Checkpoint architecture differs from the specification')
    if contract.get('binding_sha256') != binding_sha256:
        raise ValueError('Checkpoint dataset binding differs')
    if contract.get('evaluation_sha256') != evaluation_sha256:
        raise ValueError('Checkpoint evaluation selection differs')
    if type(state.get('step')) is not int or state['step'] < 0:
        raise ValueError('Checkpoint step is invalid')
    if not isinstance(state.get('model'), dict) or not state['model']:
        raise ValueError('Checkpoint model tensors are missing')


def compare_modules(current, reference):
    """Require identical tensor schemas; report exact equality and differences."""
    if set(current) != set(reference):
        raise ValueError('Checkpoint model tensor names differ')
    prefixes = ('tokenizer.', 'lam.', 'dynamics.')
    if any(not isinstance(key, str) or not key.startswith(prefixes) for key in current):
        raise ValueError('Unexpected model tensor group')
    result = {}
    for prefix in prefixes:
        keys = sorted(key for key in current if key.startswith(prefix))
        if not keys:
            raise ValueError('Required model tensor group is missing')
        identical = True
        maximum = 0.0
        elements = 0
        for key in keys:
            a, b = current[key], reference[key]
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise ValueError('Model state contains a non-tensor')
            if a.device.type != 'cpu' or b.device.type != 'cpu':
                raise ValueError('Control audit is CPU-only')
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError('Checkpoint tensor shape or dtype differs: ' + key)
            if not a.is_floating_point() or a.is_complex():
                raise ValueError('Expected real floating-point model weights: ' + key)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Checkpoint has nonfinite model weights: ' + key)
            identical &= torch.equal(a, b)
            elements += a.numel()
            if a.numel():
                maximum = max(maximum, float((a.double() - b.double()).abs().max()))
        result[prefix[:-1]] = dict(
            tensors=len(keys), elements=elements, exactly_unchanged=identical,
            maximum_absolute_difference=maximum)
    return result


def fresh_tokenizer_provenance(document, reference):
    """Only a matching recorded fresh run supports an untrained-weights claim."""
    if not isinstance(document, dict) or not isinstance(document.get('arguments'), dict):
        return False
    arguments = document['arguments']
    return (
        document.get('contract') == reference['contract']
        and arguments.get('stage') == 'tokenizer'
        and 'initial' in arguments and arguments['initial'] is None
        and 'resume' in arguments and arguments['resume'] is None)


def control_findings(modules, fresh_reference):
    frozen = all(modules[name]['exactly_unchanged'] for name in ('lam', 'dynamics'))
    untrained = bool(fresh_reference and frozen)
    reason = ('Action and dynamics weights are unchanged from a recorded fresh '
              'tokenizer-only run; neither module has been trained.' if untrained else
              'A tokenizer-stage checkpoint provides no evidence of trained action '
              'semantics or dynamics; untrained initialization provenance is incomplete.'
              if frozen else 'Action or dynamics tensors changed despite the tokenizer '
              'stage contract; investigate provenance before evaluating control.')
    return dict(
        status='not_eligible_for_playable_control',
        frozen_modules_consistent=frozen,
        action_and_dynamics_untrained=untrained,
        reason=reason,
        G2=dict(status='unmeasured', eligible=False, reason=reason),
        G3=dict(status='unmeasured', eligible=False, reason=reason),
        commands_evaluated=0, interventions_evaluated=0, rollout_frames_evaluated=0,
        semantic_mapping='unmeasured',
        seed_initialization_equivalence='not established; no model was constructed',
        prerequisites=[
            'A production-eligible tokenizer with scored G1 evidence bound to its exact checkpoint.',
            'A trained action stage; calibrate code meanings using training footage only.',
            'At least 32 held-out intervention prefixes from eight validation sources, '
            'with at least eight prefixes and three sources per behavior.',
            'Compare all single-group alternatives with copy-last and past-only velocity; '
            'score directional transfer and movement/aim/fire leakage separately.',
            'A trained dynamics stage and future-free commanded 1/5/10/30-second rollouts '
            'with player, geometry, trajectory and wall-crossing checks.'])


def run(root, checkpoint, reference, out):
    root, checkpoint, reference, out = map(Path, (root, checkpoint, reference, out))
    if out.exists():
        raise FileExistsError('Choose a new audit output')
    spec_path = root / 'configs/training-baseline-v1.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    binding_hash = sha256(root / spec['data_binding'])
    selection_path = root / spec['evaluation_selection']
    selection_hash = sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('binding_sha256') != binding_hash:
        raise ValueError('Evaluation selection uses a different dataset binding')
    current, current_hash = load_bound_checkpoint(checkpoint)
    prior, prior_hash = load_bound_checkpoint(reference)
    for state in (current, prior):
        validate_checkpoint(state, spec['model'], binding_hash, selection_hash)
    if current['step'] < prior['step']:
        raise ValueError('Reference is newer than the audited checkpoint')
    modules = compare_modules(current['model'], prior['model'])
    provenance_path = reference.parent / 'run.json'
    provenance = {}
    fresh = False
    if provenance_path.is_file():
        provenance_hash = sha256(provenance_path)
        document = json.loads(provenance_path.read_text(encoding='utf-8'))
        if sha256(provenance_path) != provenance_hash:
            raise ValueError('Reference provenance changed while being read')
        fresh = fresh_tokenizer_provenance(document, prior)
        provenance = dict(path=str(provenance_path.resolve()), sha256=provenance_hash,
                          records_fresh_tokenizer_run=fresh)
    report = dict(
        schema_version=1, purpose='Read-only control eligibility audit; no quality pass',
        audit_implementation_sha256=sha256(__file__), torch_version=str(torch.__version__),
        checkpoint=dict(path=str(checkpoint.resolve()), sha256=current_hash,
                        step=current['step'], contract=current['contract']),
        reference=dict(path=str(reference.resolve()), sha256=prior_hash,
                       step=prior['step'], contract=prior['contract']),
        specification_sha256=sha256(spec_path), binding_sha256=binding_hash,
        evaluation_sha256=selection_hash, reference_provenance=provenance,
        module_comparison=modules, control=control_findings(modules, fresh),
        preflight_weights_promotion_allowed=False,
        reserved_test_access=False, optimizer_updates=0, device='cpu',
        calibration_windows=len(selection['training_calibration']),
        calibration_sources=len({item['source_id'] for item in selection['training_calibration']}),
        validation_windows=len(selection['validation']),
        validation_sources=len({item['source_id'] for item in selection['validation']}))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report['control'], indent=2))


if __name__ == '__main__':
    main()
