"""Audit downloaded continuation evidence on CPU without changing its artifacts.

This verifies saved evidence and independently replays sampling metadata. It does
not train, decode images, repeat the CUDA resume experiment, or award a quality
gate. The optional output receipt must be outside the audited run directory.
Stops before both trainers initialize lack the evidence for this full audit.
"""
import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import torch

from genie_diep.cohort import Cohort, read
from genie_diep.coverage_sampler import CoverageSampler, sampler_state_sha256
from genie_diep.data import file_hash
from scripts.run_tokenizer_comparison import ARMS, compare_exact, cpu_copy
from scripts.run_tokenizer_continuation import inside, validate_experiment, validate_parent
from scripts.validate_gpu_recovery_artifacts import finite_tensors, require


ORIGINAL_RUN = 'runs/tokenizer-comparison-v1-2026-11-10/comparison'
METRICS = ('mse', 'foreground_mse', 'temporal_mse')
STATE_FIELDS = ('model', 'optimizer', 'scheduler', 'scaler', 'sampler', 'rng', 'step', 'config', 'contract', 'best')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def number(value, label, *, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum), 'Invalid '+label)


def integer(value, label, *, minimum=0, maximum=None):
    require(type(value) is int, 'Invalid '+label)
    number(value, label, minimum=minimum, maximum=maximum)


def validate_report(report, plan, experiment_sha, updates):
    """Check paired accounting, including a retained but uncommitted half-pair."""
    require(report['status'] in ('completed', 'budget_exhausted'), 'Run failed or is still running')
    require(report['preflight'] is False and report['promotion_approved'] is False
            and report['quality_gates'] == 'unmeasured',
            'Expected unpromoted production research')
    require(report['device'].startswith('cuda'), 'Run did not execute on CUDA')
    require(report['experiment_sha256'] == experiment_sha and report['limits'] == plan['limits'],
            'Report experiment or authorization changed')
    require(report['parent_step'] == 999 and report['target_step'] == 5000
            and report['sampling_change'] == plan['sampling_policy'], 'Continuation trajectory changed')
    common = report['latest_common_step']
    integer(common, 'common step', minimum=999, maximum=5000)
    if report['status'] == 'completed':
        require(common == 5000, 'Completed run stopped before step 5000')
    require(not report.get('checkpoint_save_error') and not report.get('checkpoint_save_errors')
            and not report.get('failures'), 'Run contains failed checkpoint or update evidence')
    for field, cap in [('spent_seconds', 10800), ('estimated_units', 4)]:
        number(report[field], field, maximum=cap)
    integer(report['peak_allocated_bytes'], 'peak CUDA allocation', minimum=1)
    require([row['step'] for row in updates] == list(range(1000, common+1)),
            'Paired update log has missing or duplicate steps')
    for row in updates:
        require(set(row['arms']) == set(ARMS), 'Update is missing an arm')
        for value in row['arms'].values():
            require(value['step'] == row['step'] and value['scale'] == 1024, 'Update boundary or scale changed')
            number(value['loss'], 'update loss')
            number(value['seconds'], 'update duration', minimum=1e-12)
    partials = report.get('partial_checkpoints', [])
    require(len(partials) <= 1, 'Multiple unmatched partial updates')
    for partial in partials:
        first = ARMS[0] if common % 2 == 0 else ARMS[1]
        require(report['status'] != 'completed' and common < 5000 and partial['arm'] == first
                and partial['step'] == common+1 and partial['comparable'] is False,
                'Partial checkpoint was promoted or is not the unmatched next update')
        require(partial['path'] == f"{partial['arm']}/partial-step-{partial['step']}.pt",
                'Unexpected partial checkpoint path')
    measurement = report['measurement']
    passed = [arm for arm in ARMS if arm in measurement]
    require(set(measurement) <= set(ARMS) | {'both_arms_passed'} and passed == list(ARMS)[:len(passed)],
            'Unexpected resume measurement order')
    require(measurement.get('both_arms_passed', False) is (len(passed) == 2),
            'Both-arm resume flag disagrees with measurements')
    if common > 1001 or report['status'] == 'completed':
        require(len(passed) == 2, 'Training advanced without both exact resume checks')
    require(common >= 1001 or not passed, 'Resume measurement precedes its boundary')
    for field in ('optimizer_calls', 'replay_optimizer_calls', 'unique_new_updates_per_arm', 'migrations', 'frozen_modules', 'coverage'):
        require(set(report[field]) == set(ARMS), 'Missing arm in '+field)
    for arm in ARMS:
        calls = report['optimizer_calls'][arm]
        replays = report['replay_optimizer_calls'][arm]
        discarded = sum(item['arm'] == arm for item in partials)
        integer(calls, 'optimizer calls', maximum=4002)
        integer(replays, 'replay calls', maximum=1)
        require(replays == int(arm in passed) and calls == common-999+replays+discarded,
                'Optimizer/replay/partial accounting inconsistent')
        require(report['unique_new_updates_per_arm'][arm] == common-999,
                'Unique updates do not match the common boundary')
        if arm in passed:
            item = measurement[arm]
            require(item['status'] == 'passed' and item['maximum_state_difference'] == 0 and item['step'] == 1001,
                    'GPU exact-resume evidence did not pass')
            number(item['seconds'], 'resume duration', minimum=1e-12)
            require(item['loss'] == updates[1]['arms'][arm]['loss'], 'Resume loss differs from update 1001')
        if report['status'] == 'completed':
            require(calls == 4002 and report['unique_new_updates_per_arm'][arm] == 4001,
                    'Completed continuation call cap differs')
        frozen = report['frozen_modules'][arm]
        require(frozen['unchanged'] is True and frozen['tensors'] == 98, 'Frozen-module evidence changed')
        migration = report['migrations'][arm]
        expected = dict(model_exact=True, optimizer_exact=True, scheduler_exact=True, scaler_exact=True,
                        rng_exact=True, parent_step=999, parent_sha256=plan['parents'][arm]['sha256'],
                        only_training_change='shared complete-coverage sampling schedule', preflight=False)
        require(migration == expected, 'Migration evidence changed')
    saved = [step for step in plan['save_steps'] if step <= common and (step != 1001 or len(passed) == 2)]
    require(report['saved_common_steps'] == saved, 'Saved common boundaries are missing or out of order')
    return common, saved, partials


def validate_evaluations(report, plan, selection):
    """Require fixed descriptor prefixes and complete assessments before advancing."""
    descriptors = {'train': selection['training_calibration'], 'val': selection['validation']}
    require(len(descriptors['train']) == 84 and len(descriptors['val']) == 64,
            'Frozen assessment window counts changed')
    require(len({d['source_id'] for d in descriptors['train']}) == 84
            and len({d['source_id'] for d in descriptors['val']}) == 16, 'Assessment source counts changed')
    common = report['latest_common_step']
    expected = [(step, arm, split) for step in plan['evaluate_steps'] if step <= common
                for arm in ARMS for split in ('train', 'val')]
    actual = [(item['common_step'], item['arm'], item['report']['split']) for item in report['evaluations']]
    require(actual == expected[:len(actual)], 'Assessment order or boundary differs')
    if report['status'] == 'completed':
        require(actual == expected, 'Completed run omitted an assessment')
    else:
        required = [key for key in expected if key[0] < common]
        require(actual[:len(required)] == required, 'Training advanced past an omitted assessment')
    for index, item in enumerate(report['evaluations']):
        metric = item['report']; split = metric['split']; n = metric['samples']
        integer(n, 'assessment sample count', maximum=len(descriptors[split]))
        require(metric['step'] == item['common_step'], 'Assessment checkpoint differs')
        require(metric['complete'] is (n == len(descriptors[split])), 'Assessment completion flag disagrees')
        counts = Counter(d['source_id'] for d in descriptors[split][:n])
        require(set(metric['by_source']) == set(counts) and metric['sources'] == len(counts),
                'Assessment descriptor-prefix sources disagree')
        for source, count in counts.items():
            require(metric['by_source'][source]['samples'] == count, 'Assessment per-source sample count disagrees')
        for values in [metric, *metric['by_source'].values()]:
            for name in METRICS:
                value = values[name]
                if value is None:
                    require(name == 'foreground_mse' or n == 0, 'Missing assessment metric')
                else:
                    number(value, 'assessment '+name)
        if n:
            for name in ('mse', 'temporal_mse'):
                average = sum(metric['by_source'][source][name]*count for source, count in counts.items())/n
                require(math.isclose(metric[name], average, rel_tol=1e-6, abs_tol=1e-10),
                        'Pooled assessment metric disagrees with sources')
        else:
            require(all(metric[name] is None for name in METRICS), 'Empty assessment has a metric')
        if not metric['complete']:
            require(report['status'] == 'budget_exhausted' and index == len(report['evaluations'])-1
                    and item['common_step'] == common, 'Incomplete assessment was followed by more work')


def replay_sampling(data, parent_state, seed, steps):
    """Reconstruct expected states from draws only; no proxy images are decoded."""
    sampler = CoverageSampler(data, parent_state, seed=seed,
                              expected_parent_sampler_sha256=sampler_state_sha256(parent_state))
    states = {}; stats = {}
    for step in sorted(steps):
        target = parent_state['draws']+(step-999)*16
        while sampler.draws < target:
            sampler.draw()
        require(sampler.draws == target, 'Invalid requested replay boundary')
        states[step] = cpu_copy(sampler.state_dict())
        stats[step] = copy.deepcopy(sampler.plan_stats())
    return states, stats


def expected_contract(parent, plan, experiment_sha, arm):
    contract = copy.deepcopy(parent['contract'])
    contract['continuation'] = dict(experiment_sha256=experiment_sha,
        parent_sha256=plan['parents'][arm]['sha256'], parent_step=999, sampling_policy=plan['sampling_policy'],
        coverage_seed=plan['coverage_seed'], source_hashes=plan['source_hashes'],
        parent_sampler_sha256=sampler_state_sha256(parent['sampler']),
        accounting='New authorization; old allocation totals retained in parent checkpoint only',
        migration='CUDA state-preserving sampling-plan branch')
    return contract


def validate_state(state, parent, contract, sampler, step, report, frozen):
    require(state['step'] == step and state['contract'] == contract, 'Checkpoint boundary or contract changed')
    require(state['config'] == parent['config'], 'Checkpoint architecture changed')
    require(state.get('last_failure') is None, 'Checkpoint retains a failed update')
    require(state['best'] == parent['best'], 'Continuation unexpectedly changed checkpoint selection')
    for key, maximum in [('spent_seconds', report['spent_seconds']), ('spent_units', report['estimated_units'])]:
        number(state[key], 'checkpoint '+key, maximum=maximum)
    require(state['model'].keys() == parent['model'].keys(), 'Model tensors missing or added')
    for name, tensor in state['model'].items():
        require(tensor.shape == parent['model'][name].shape and tensor.dtype == parent['model'][name].dtype,
                'Model tensor layout changed: '+name)
    compare_exact({name:state['model'][name] for name in frozen}, frozen)
    compare_exact(state['sampler'], sampler)
    expected_scaler = copy.deepcopy(parent['scaler'])
    expected_scaler['_growth_tracker'] = step % expected_scaler['growth_interval']
    compare_exact(state['scaler'], expected_scaler)
    require(state['scaler']['scale'] == 1024, 'FP16 scale cap changed')
    scheduler = copy.deepcopy(parent['scheduler'])
    # Keep the original scheduler's operation order: arbitrary budget-stop
    # boundaries otherwise differ by one floating-point rounding bit.
    progress = min(1, (step-500)/max(1, 20000-500))
    factor = .1+.9*(1+math.cos(math.pi*progress))/2
    rates = [base*factor for base in scheduler['base_lrs']]
    scheduler.update(last_epoch=step, _step_count=step+1, _last_lr=rates)
    compare_exact(state['scheduler'], scheduler)
    require(state['optimizer']['state'].keys() == parent['optimizer']['state'].keys(), 'Adam slots missing or added')
    require(len(state['optimizer']['param_groups']) == len(parent['optimizer']['param_groups']), 'Adam groups changed')
    for index, group in enumerate(state['optimizer']['param_groups']):
        expected = copy.deepcopy(parent['optimizer']['param_groups'][index]); expected['lr'] = rates[index]
        compare_exact(group, expected)
    for key, slot in state['optimizer']['state'].items():
        require(slot.keys() == parent['optimizer']['state'][key].keys(), 'Adam slot layout changed')
        require(float(slot['step']) == step, 'Adam step disagrees with checkpoint')
        for name, tensor in slot.items():
            require(tensor.shape == parent['optimizer']['state'][key][name].shape
                    and tensor.dtype == parent['optimizer']['state'][key][name].dtype, 'Adam tensor layout changed')
    # The bound tokenizer/objectives have no stochastic forward/update operation.
    # Both samplers use private generators. Global Python/NumPy/CPU/CUDA RNG must
    # therefore retain the authenticated parent state throughout this experiment.
    compare_exact(state['rng'], parent['rng'])
    if step == 999:
        for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng'):
            compare_exact(state[key], parent[key])
    return finite_tensors(state['model'], 'model')+finite_tensors(state['optimizer'], 'optimizer')


def validate(root, run, experiment_path='configs/tokenizer-continuation-v1.json'):
    root = Path(root).resolve(); run = Path(run).resolve()
    plan = validate_experiment(root, experiment_path)
    experiment_sha = file_hash(inside(root, experiment_path))
    prior = read(inside(root, plan['parent_experiment']['path']))
    parents = {}; original_receipts = []
    for arm in ARMS:
        entry = plan['parents'][arm]
        original = inside(root, f'{ORIGINAL_RUN}/{arm}/last-common.pt')
        require(file_hash(original) == entry['sha256'], 'Original parent checkpoint changed: '+arm)
        parents[arm] = load(inside(root, entry['path']))
        validate_parent(parents[arm], plan, arm, prior, root)
        finite_tensors(parents[arm]['model'], 'parent.model'); finite_tensors(parents[arm]['optimizer'], 'parent.optimizer')
        original_receipts.append(dict(arm=arm, original_path=original.relative_to(root).as_posix(),
                                     input_path=entry['path'], sha256=entry['sha256'], bytes=original.stat().st_size))
    compare_exact(parents['baseline']['sampler'], parents['detail']['sampler'])
    report_path = run/'continuation-report.json'; report = read(report_path)
    log = run/'updates.jsonl'
    updates = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line] if log.exists() else []
    common, saved, partials = validate_report(report, plan, experiment_sha, updates)
    require(all(report['torch_version'] == parent['contract']['torch_version'] for parent in parents.values()),
            'Runtime differs from the authenticated parent')
    spec = read(root/prior['arms']['baseline']['spec'])
    validate_evaluations(report, plan, read(root/spec['evaluation_selection']))
    data = Cohort(root, spec['data_binding'], 'train', spec['model']['context'])
    steps = {999, common, *saved, *(p['step'] for p in partials)}
    if common >= 1000:
        steps.add(1000)
    samplers, stats = replay_sampling(data, parents['baseline']['sampler'], plan['coverage_seed'], steps)
    require(stats[common]['total_training_frames'] == 315320 and stats[common]['training_sources'] == 84,
            'Frozen training-frame/source inventory changed')
    for arm in ARMS:
        require(report['coverage'][arm] == stats[common], 'Coverage report differs from independently replayed metadata')
    if report['status'] == 'completed':
        require(stats[common]['full_coverage'] is True and stats[common]['coverage_unique_frames'] == 315320,
                'Completed run did not cover every prepared training frame')
    frozen = {name:tensor for name, tensor in parents['baseline']['model'].items() if name.startswith(('lam.', 'dynamics.'))}
    require(len(frozen) == 98, 'Parent frozen-module tensor inventory changed')
    compare_exact({name:parents['detail']['model'][name] for name in frozen}, frozen)
    checkpoints = []; checked = 0; expected_paths = set()
    for arm in ARMS:
        names = {'branch-start.pt':999, 'last-common.pt':common, 'last.pt':common}
        if common >= 1000:
            names['measurement-boundary.pt'] = 1000
        names.update({f'step-{step:06d}.pt':step for step in saved})
        names.update({f'partial-step-{p["step"]}.pt':p['step'] for p in partials if p['arm'] == arm})
        expected_paths.update(f'{arm}/{name}' for name in names)
        contract = expected_contract(parents[arm], plan, experiment_sha, arm)
        final = load(run/arm/'last-common.pt')
        for name, step in sorted(names.items()):
            path = run/arm/name; state = load(path)
            count = validate_state(state, parents[arm], contract, samplers[step], step, report, frozen)
            checked += count
            if name == 'last.pt' or (name.startswith('step-') and step == common):
                for key in STATE_FIELDS:
                    compare_exact(state[key], final[key])
            checkpoints.append(dict(path=path.relative_to(run).as_posix(), step=step,
                                    sha256=file_hash(path), bytes=path.stat().st_size, finite_tensors=count,
                                    comparable=not name.startswith('partial-')))
    require({p.relative_to(run).as_posix() for p in run.rglob('*.pt')} == expected_paths,
            'Unexpected or missing checkpoint artifacts')
    # Recheck lineage after reading all artifacts. Audit never rewrites old files.
    for entry in original_receipts:
        require(file_hash(root/entry['original_path']) == entry['sha256']
                and file_hash(root/entry['input_path']) == entry['sha256'], 'Parent changed during audit')
    files = [dict(path=p.relative_to(run).as_posix(), bytes=p.stat().st_size, sha256=file_hash(p))
             for p in sorted(run.rglob('*')) if p.is_file()]
    return dict(schema_version=1, status='validated', run_status=report['status'],
                experiment_sha256=experiment_sha, latest_common_step=common,
                optimizer_calls=report['optimizer_calls'], unique_new_updates_per_arm=report['unique_new_updates_per_arm'],
                parent_checkpoints_unchanged=True, parents=original_receipts,
                branch_state_exact_to_parents=True, sampling_independently_replayed=True,
                coverage=stats[common], frozen_lam_dynamics_exact=True, frozen_tensors=98,
                finite_tensors_checked=checked, assessments_checked=len(report['evaluations']),
                gpu_resume_evidence='Saved CUDA replay evidence checked; this CPU audit does not repeat CUDA execution.',
                accounting_limit='Shared harness totals checked; live charges and runtime release require separate evidence.',
                quality_gates='unmeasured', promotion_approved=False, checkpoints=checkpoints, files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--experiment', default='configs/tokenizer-continuation-v1.json')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.out:
        destination = Path(args.out).resolve()
        require(not destination.is_relative_to(Path(args.run).resolve()), 'Receipt must be outside the audited run')
        require(not destination.exists(), 'Do not overwrite an existing validation receipt')
    torch.set_num_threads(2)
    result = validate(args.root, args.run, args.experiment)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.out:
        destination.write_text(encoded, encoding='utf-8')
    else:
        print(encoded, end='', flush=True)


if __name__ == '__main__':
    main()
