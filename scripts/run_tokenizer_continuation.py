"""Matched step-999 continuation with a complete train-frame coverage pass.

This is an explicit sampling-plan branch, not an exact continuation of the old
random sampling trajectory. Model, optimizer, scheduler and objectives survive.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch

from genie_diep.baseline import Budget, restore_rng, rng_state
from genie_diep.cohort import read
from genie_diep.coverage_sampler import CoverageSampler, sampler_state_sha256
from genie_diep.data import file_hash
from scripts.run_tokenizer_comparison import (
    ARMS, ComparisonTrainer, check, compare_exact, cpu_copy, paired_update,
    source_hash, validate_experiment as validate_parent_experiment,
)


LIMITS = dict(additional_optimizer_calls_per_arm=4002, additional_unique_updates_per_arm=4001,
              maximum_compute_units=4, maximum_allocated_hours=3, minimum_remaining_units=50)
SAVE_STEPS = [1001, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
EVALUATE_STEPS = [999, 2000, 3000, 4000, 5000]


def inside(root, relative):
    path = (Path(root)/relative).resolve()
    check(path.is_relative_to(Path(root).resolve()), 'Input escapes the project')
    return path


def validate_experiment(root, experiment_path):
    root = Path(root).resolve()
    plan = read(inside(root, experiment_path))
    check(plan['parent_step'] == 999 and plan['target_step'] == 5000, 'Continuation boundaries changed')
    check(plan['limits'] == LIMITS, 'Continuation budget changed')
    check(plan['save_steps'] == SAVE_STEPS and plan['evaluate_steps'] == EVALUATE_STEPS,
          'Predeclared assessment boundaries changed')
    check(plan['schedule_max_updates'] == 20000 and plan['coverage_seed'] == 20261110,
          'Schedule or coverage seed changed')
    check(plan['sampling_policy'] == 'full_train_frame_coverage_then_parent_mixed', 'Unknown coverage policy')
    parent = plan['parent_experiment']
    check(file_hash(inside(root, parent['path'])) == parent['sha256'], 'Parent experiment changed')
    prior = read(inside(root, parent['path']))
    validate_parent_experiment(root, parent['path'], prior['arms']['baseline']['spec'], prior['arms']['detail']['spec'])
    for key, name in [('runner_sha256', 'scripts/run_tokenizer_continuation.py'),
                      ('coverage_sampler_sha256', 'genie_diep/coverage_sampler.py')]:
        check(plan['source_hashes'][key] == source_hash(root/name), 'Continuation source changed: '+name)
    check(set(plan['parents']) == set(ARMS), 'Both parent checkpoints are required')
    for arm in ARMS:
        entry = plan['parents'][arm]
        check(entry['path'] == f'continuation-inputs/{arm}-step999.pt', 'Unexpected parent destination')
        check(file_hash(inside(root, entry['path'])) == entry['sha256'], 'Parent checkpoint changed: '+arm)
    return plan


def validate_parent(state, plan, arm, parent_experiment, root):
    contract = state['contract']
    check(state['step'] == plan['parent_step'], 'Parent must be step 999')
    check(contract['stage'] == 'tokenizer' and not contract['preflight'], 'Parent is not a research tokenizer')
    check(contract['microbatch'] == 4 and contract['accumulation'] == 4, 'Parent batch changed')
    expected = dict(experiment_sha256=plan['parent_experiment']['sha256'], arm=arm,
                    objective=parent_experiment['arms'][arm]['objective'],
                    source_hashes=parent_experiment['source_hashes'])
    check(contract['comparison'] == expected, 'Parent objective/experiment differs')
    spec = read(Path(root)/parent_experiment['arms'][arm]['spec'])
    check(state['config'] == spec['model'], 'Parent architecture differs')
    check(contract['spec_sha256'] == parent_experiment['arms'][arm]['spec_sha256'], 'Parent specification differs')
    check(contract['binding_sha256'] == file_hash(Path(root)/spec['data_binding']), 'Parent data binding differs')
    check(contract['evaluation_sha256'] == file_hash(Path(root)/spec['evaluation_selection']), 'Parent evaluation differs')
    check(state['sampler']['draws'] == 1+999*16 and state['scheduler']['last_epoch'] == 999,
          'Parent sampling or schedule boundary differs')
    check(contract['precision'] == 'fp16' and state['scaler']['scale'] == 1024., 'Parent precision differs')


def validate_continuation_readiness(root, experiment_path):
    readiness = read(Path(root)/'configs/tokenizer-continuation-readiness-v1.json')
    check(readiness['status'] == 'passed' and readiness['experiment_sha256'] == file_hash(Path(root)/experiment_path),
          'Continuation local execution evidence is missing or stale')
    check(readiness['parents_unchanged'] is True and readiness['full_suite_passed'] is True,
          'Continuation preservation/test evidence is incomplete')
    check(readiness['exact_resume_both_arms'] is True and readiness['maximum_state_difference'] == 0,
          'Continuation local exact-resume evidence is missing')
    return readiness


class ContinuationTrainer(ComparisonTrainer):
    def __init__(self, root, experiment_path, arm, device='cpu', preflight=False, resume=None):
        root = Path(root).resolve()
        plan = validate_experiment(root, experiment_path)
        prior = read(root/plan['parent_experiment']['path'])
        super().__init__(root, prior['arms'][arm]['spec'], plan['parent_experiment']['path'],
                         arm, device=device, preflight=preflight)
        parent = torch.load(root/plan['parents'][arm]['path'], map_location='cpu', weights_only=True)
        validate_parent(parent, plan, arm, prior, root)
        if not preflight:
            check(self.device.type == 'cuda' and str(torch.__version__) == parent['contract']['torch_version'],
                  'Production continuation requires the verified parent CUDA/PyTorch version')
        self.model.load_state_dict(parent['model'])
        self.optimizer.load_state_dict(parent['optimizer'])
        self.scheduler.load_state_dict(parent['scheduler'])
        self.scaler.load_state_dict(parent['scaler'])
        self.sampler = CoverageSampler(self.data, parent['sampler'], seed=plan['coverage_seed'],
                                       expected_parent_sampler_sha256=sampler_state_sha256(parent['sampler']))
        self.step = parent['step']; self.best = parent['best']
        self.spent_seconds = 0.; self.spent_units = 0.; self.last_failure = None
        inherited_rng = cpu_copy(parent['rng'])
        if preflight:
            inherited_rng['cuda'] = []
        restore_rng(inherited_rng)
        self.arm_rng = rng_state()
        self.parent_sampler_state = cpu_copy(parent['sampler'])
        self.continuation_plan = plan
        self.contract['continuation'] = dict(
            experiment_sha256=file_hash(root/experiment_path), parent_sha256=plan['parents'][arm]['sha256'],
            parent_step=999, sampling_policy=plan['sampling_policy'], coverage_seed=plan['coverage_seed'],
            source_hashes=plan['source_hashes'], parent_sampler_sha256=sampler_state_sha256(parent['sampler']),
            accounting='New authorization; old allocation totals retained in parent checkpoint only',
            migration='CPU FP32 execution check' if preflight else 'CUDA state-preserving sampling-plan branch')
        compare_exact(self.model.state_dict(), parent['model'])
        compare_exact(self.optimizer.state_dict(), parent['optimizer'])
        compare_exact(self.scheduler.state_dict(), parent['scheduler'])
        if not preflight:
            compare_exact(self.scaler.state_dict(), parent['scaler'])
            compare_exact(self.arm_rng, parent['rng'])
        self.migration = dict(model_exact=True, optimizer_exact=True, scheduler_exact=True,
                              scaler_exact=not preflight, rng_exact=not preflight,
                              parent_step=999, parent_sha256=plan['parents'][arm]['sha256'],
                              only_training_change='shared complete-coverage sampling schedule',
                              preflight=preflight)
        if resume:
            state = torch.load(resume, map_location='cpu', weights_only=True)
            check(state['contract'] == self.contract, 'Continuation resume contract differs')
            self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler']); self.scaler.load_state_dict(state['scaler'])
            self.sampler.load_state_dict(state['sampler']); self.step = state['step']; self.best = state['best']
            check(999 <= self.step <= 5000 and state['scheduler']['last_epoch'] == self.step,
                  'Continuation resume step/scheduler boundary differs')
            check(self.sampler.draws-self.sampler.parent_draws == (self.step-999)*16,
                  'Continuation resume sampling/step boundary differs')
            self.spent_seconds = state['spent_seconds']; self.spent_units = state['spent_units']
            self.last_failure = state.get('last_failure'); restore_rng(state['rng']); self.arm_rng = rng_state()

    def _failure_diagnostics(self, boundary, phase, microbatches, error):
        affected = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                affected.append(dict(name=name, nonfinite=int((~torch.isfinite(parameter.grad)).sum())))
        replay = CoverageSampler(self.data, self.parent_sampler_state, seed=self.continuation_plan['coverage_seed'],
                                 expected_parent_sampler_sha256=sampler_state_sha256(self.parent_sampler_state))
        replay.load_state_dict(boundary['sampler'])
        windows = [self.data.descriptor(*replay.draw()) for _ in range(self.sampler.draws-boundary['sampler']['draws'])]
        return dict(attempted_step=boundary['step']+1, phase=phase, completed_microbatches=microbatches,
                    exception=type(error).__name__, message=str(error), precision=self.precision,
                    scale=self.scaler.get_scale(), sampled_windows=windows, affected_gradients=affected,
                    restored_pre_update_boundary=False)

    @torch.no_grad()
    def assess(self, split, budget=None, limit=None):
        check(split in {'train', 'val'}, 'Assessment split must exclude reserved test')
        descriptors = self.evaluation['validation' if split == 'val' else 'training_calibration'][:limit]
        cohort = self.validation if split == 'val' else self.data
        self.model.eval(); records = []; estimate = 0.
        try:
            for item in descriptors:
                if budget is not None and not budget.permits(estimate):
                    break
                start = time.monotonic()
                x = cohort.window(*cohort.locate(item))[None].to(self.device)
                reconstruction, _, _ = self.model.tokenizer(x)
                error = (reconstruction-x).square()
                high, low = x.amax(2), x.amin(2)
                foreground = (high >= 65/255) & ((high-low)/high.clamp_min(1e-8) >= 65/255)
                rgb_error = error.mean(2)
                count = int(foreground.sum())
                records.append(dict(source_id=item['source_id'], mse=float(error.mean()),
                    foreground_error_sum=float((rgb_error*foreground).sum()), foreground_count=count,
                    temporal_mse=float(((reconstruction[:,1:]-reconstruction[:,:-1])-(x[:,1:]-x[:,:-1])).square().mean())))
                estimate = max(estimate, (time.monotonic()-start)*1.25)
        finally:
            self.module.train()
        by_source = {}
        for source in sorted({row['source_id'] for row in records}):
            rows = [r for r in records if r['source_id'] == source]
            pixels = sum(r['foreground_count'] for r in rows)
            by_source[source] = dict(samples=len(rows), mse=sum(r['mse'] for r in rows)/len(rows),
                temporal_mse=sum(r['temporal_mse'] for r in rows)/len(rows),
                foreground_mse=sum(r['foreground_error_sum'] for r in rows)/pixels if pixels else None)
        n = len(records); pixels = sum(r['foreground_count'] for r in records)
        return dict(split=split, step=self.step, samples=n, sources=len(by_source), complete=n == len(descriptors),
                    mse=sum(r['mse'] for r in records)/n if n else None,
                    foreground_mse=sum(r['foreground_error_sum'] for r in records)/pixels if pixels else None,
                    temporal_mse=sum(r['temporal_mse'] for r in records)/n if n else None,
                    by_source=by_source,
                    foreground_policy='Target-only HSV saturation>=65/255 and value>=65/255; colored pixels, not semantic objects',
                    interpretation='Fixed reconstruction diagnostics; no semantic or control claim')


def allocation_budget(args):
    if args.preflight:
        check(torch.device(args.device).type == 'cpu', 'Local preflight must be CPU-only')
        return Budget(dict(max_allocated_hours=1, max_compute_units=2), reserve_seconds=0)
    check(args.authorized and torch.device(args.device).type == 'cuda', 'Authorized CUDA continuation required')
    now = time.time()
    check(all(v is not None and math.isfinite(v) for v in (args.rate, args.balance, args.allocation_start)),
          'Live allocation accounting required')
    check(args.rate > 0 and args.balance > 50 and 0 < args.allocation_start <= now, 'Invalid allocation accounting')
    return Budget(dict(max_allocated_hours=3, max_compute_units=4), rate=args.rate, balance=args.balance,
                  setup_seconds=now-args.allocation_start, reserve_seconds=180)


def run(args):
    root = Path(args.root).resolve(); out = Path(args.out).resolve()
    check(not out.exists(), 'Use a new continuation directory')
    budget = allocation_budget(args)
    plan = validate_experiment(root, args.experiment)
    if not args.preflight:
        validate_continuation_readiness(root, args.experiment)
    torch.set_num_threads(args.threads)
    target = 1001 if args.preflight else plan['target_step']
    if torch.device(args.device).type == 'cuda':
        check(torch.cuda.is_available(), 'CUDA unavailable'); torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True)
    report = dict(schema_version=1, status='running', preflight=args.preflight, promotion_approved=False, quality_gates='unmeasured',
                  experiment_sha256=file_hash(root/args.experiment), parent_step=999, target_step=target,
                  limits=LIMITS, latest_common_step=999, optimizer_calls={a:0 for a in ARMS},
                  replay_optimizer_calls={a:0 for a in ARMS}, measurement={}, migrations={},
                  saved_common_steps=[], evaluations=[], allocation_accounting='Shared new-run totals; never sum the two arms',
                  sampling_change=plan['sampling_policy'], device=args.device, torch_version=str(torch.__version__))
    trainers = {}; frozen = None

    def write_report():
        report['spent_seconds'], report['estimated_units'] = budget.spent()
        if trainers:
            report['coverage'] = {a:t.sampler.plan_stats() for a,t in trainers.items()}
        temporary = out/'continuation-report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(out/'continuation-report.json')

    def save_common(name):
        for arm in ARMS:
            trainers[arm].save(out/arm/name, budget)

    def on_success(arm, result):
        report['optimizer_calls'][arm] += 1
        check(report['optimizer_calls'][arm] <= (3 if args.preflight else LIMITS['additional_optimizer_calls_per_arm']),
              'Optimizer call allowance exceeded')

    def on_partial(arm, trainer):
        path = out/arm/f'partial-step-{trainer.step}.pt'; trainer.save(path, budget)
        report.setdefault('partial_checkpoints', []).append(dict(arm=arm, step=trainer.step, comparable=False,
                                                               path=path.relative_to(out).as_posix()))

    def assess_common(step):
        for arm in ARMS:
            for split in ('train', 'val'):
                metric = trainers[arm].assess(split, budget, limit=2 if args.preflight else None)
                report['evaluations'].append(dict(arm=arm, common_step=step, report=metric)); write_report()
                check(metric['samples'] <= (84 if split == 'train' else 64), 'Unexpected assessment size')
                if not metric['complete']:
                    raise TimeoutError('Budget ended during fixed assessment')

    try:
        if not budget.permits(120):
            raise TimeoutError('Setup exhausted continuation budget')
        for arm in ARMS:
            trainers[arm] = ContinuationTrainer(root, args.experiment, arm, args.device, args.preflight)
            report['migrations'][arm] = trainers[arm].migration
        compare_exact(trainers['baseline'].sampler.state_dict(), trainers['detail'].sampler.state_dict())
        frozen = {n:p.detach().cpu().clone() for n,p in trainers['baseline'].model.state_dict().items()
                  if n.startswith(('lam.', 'dynamics.'))}
        compare_exact({n:trainers['detail'].model.state_dict()[n] for n in frozen}, frozen)
        check(len(trainers['baseline'].evaluation['validation']) == 64 and
              len(trainers['baseline'].evaluation['training_calibration']) == 84, 'Frozen probes changed')
        save_common('branch-start.pt'); write_report(); assess_common(999)
        for step in range(1000, target+1):
            if not budget.permits(max(report.get('recent_pair_seconds', 1)*2, 30)):
                raise TimeoutError('Combined continuation allowance/export reserve reached')
            results = paired_update(trainers, budget, on_success, on_partial)
            report['latest_common_step'] = step
            report['recent_pair_seconds'] = sum(row['seconds'] for row in results.values())
            with (out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step, arms=results))+'\n')
            if step == 1000:
                save_common('measurement-boundary.pt')
            if step == 1001:
                for arm in ARMS:
                    if not budget.permits(max(results[arm]['seconds']*2, 30)):
                        raise TimeoutError('Budget ended before exact-resume verification')
                    expected = cpu_copy(trainers[arm].snapshot())
                    replay = ContinuationTrainer(root, args.experiment, arm, args.device, args.preflight,
                                                 resume=out/arm/'measurement-boundary.pt')
                    started = time.monotonic(); loss = replay.update(budget)
                    if loss is None:
                        raise TimeoutError('Budget ended during exact-resume verification')
                    report['optimizer_calls'][arm] += 1; report['replay_optimizer_calls'][arm] += 1
                    compare_exact(replay.snapshot(), expected); check(loss == results[arm]['loss'], 'Replay loss differs')
                    report['measurement'][arm] = dict(status='passed', maximum_state_difference=0, step=1001,
                                                     seconds=time.monotonic()-started, loss=loss)
                    trainers[arm] = replay
                report['measurement']['both_arms_passed'] = True
            if step in SAVE_STEPS:
                save_common(f'step-{step:06d}.pt'); save_common('last-common.pt')
                report['saved_common_steps'].append(step); write_report()
            if step in EVALUATE_STEPS or (args.preflight and step == target):
                assess_common(step)
            if step % 25 == 0 or step == target:
                write_report()
                print(json.dumps(dict(step=step, losses={a:r['loss'] for a,r in results.items()},
                                      remaining_seconds=budget.remaining())), flush=True)
        if not args.preflight:
            for arm, trainer in trainers.items():
                check(trainer.sampler.plan_stats()['full_coverage'], 'Training-frame coverage remains incomplete')
                check(report['optimizer_calls'][arm] == LIMITS['additional_optimizer_calls_per_arm'],
                      'Completed optimizer-call accounting differs')
        report['status'] = 'completed'
    except TimeoutError as error:
        report.update(status='budget_exhausted', error=str(error))
    except BaseException as error:
        report.update(status='failed', exception=type(error).__name__, error=str(error))
        report['failures'] = {a:t.last_failure for a,t in trainers.items() if t.last_failure}
        raise
    finally:
        if frozen is not None:
            report['frozen_modules'] = {}
            for arm, trainer in trainers.items():
                try:
                    compare_exact({n:trainer.model.state_dict()[n] for n in frozen}, frozen)
                    report['frozen_modules'][arm] = dict(unchanged=True, tensors=len(frozen))
                except Exception as error:
                    report['frozen_modules'][arm] = dict(unchanged=False, error=str(error)); report['status'] = 'failed'
        if len(trainers) == 2 and trainers['baseline'].step == trainers['detail'].step:
            report['latest_common_step'] = trainers['baseline'].step
            try:
                save_common('last-common.pt')
            except Exception as error:
                report.update(status='failed', checkpoint_save_error=str(error))
        for arm, trainer in trainers.items():
            try:
                trainer.save(out/arm/'last.pt', budget)
            except Exception as error:
                report.setdefault('checkpoint_save_errors', {})[arm] = str(error); report['status'] = 'failed'
        if torch.device(args.device).type == 'cuda':
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['unique_new_updates_per_arm'] = {a:t.step-999 for a,t in trainers.items()}
        write_report()
    check(report['status'] != 'failed', 'Continuation failed validation or checkpoint saving')
    print(json.dumps(dict(status=report['status'], latest_common_step=report['latest_common_step'],
                         optimizer_calls=report['optimizer_calls'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--experiment', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu'); parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--authorized', action='store_true'); parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rate', type=float, default=0); parser.add_argument('--balance', type=float)
    parser.add_argument('--allocation-start', type=float)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
