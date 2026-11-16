"""Sequential fresh CNN32 and ST32 experiments under one allocation ceiling."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

# Required before a CUDA process/context creates cuBLAS workspace state.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from genie_diep.baseline import Trainer, Budget, implementation_hash, restore_rng, rng_state
from genie_diep.cohort import MixedSampler, read
from genie_diep.coverage_sampler import CoverageSampler, sampler_state_sha256
from genie_diep.data import file_hash
from scripts.run_tokenizer_comparison import compare_exact, source_hash


ARMS = ('cnn', 'transformer')
BACKEND_POLICY = dict(deterministic_algorithms=True, warn_only=False,
    cudnn_benchmark=False, cudnn_deterministic=True, cublas_workspace_config=':4096:8')


def configure_backend():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(root, plan_path):
    root = Path(root); plan = read(root/plan_path)
    check(plan['implementation_sha256'] == implementation_hash(), 'Implementation changed')
    check(plan['runner_sha256'] == source_hash(__file__), 'Runner changed')
    check(plan.get('backend_policy') == BACKEND_POLICY, 'Backend reproducibility policy differs')
    check(type(plan.get('timing_updates_per_arm', 0)) is int and
          plan.get('timing_updates_per_arm', 0) in (0, 5), 'Unsupported steady timing budget')
    for arm in ARMS:
        entry = plan['arms'][arm]
        check(file_hash(root/entry['spec']) == entry['spec_sha256'], 'Arm spec changed')
    reference = plan['initial_checkpoint']
    check(file_hash(root/reference['path']) == reference['sha256'], 'Fresh reference changed')
    return plan


class ArchitectureTrainer(Trainer):
    def __init__(self, root, plan_path, arm, device='cpu', preflight=False, resume=None):
        configure_backend()
        self.plan = validate_plan(root, plan_path)
        check(arm in ARMS, 'Unknown arm')
        self.arm = arm
        super().__init__(root, self.plan['arms'][arm]['spec'], 'tokenizer', device,
                         'fp16' if torch.device(device).type == 'cuda' else 'fp32', preflight=preflight)
        initial = torch.load(Path(root)/self.plan['initial_checkpoint']['path'], map_location='cpu', weights_only=True)
        check(initial['step'] == 0 and not initial['optimizer']['state'], 'Reference must be fresh, untrained step zero')
        check(initial['contract']['stage'] == 'tokenizer' and not initial['contract']['preflight'], 'Reference is not a production fresh tokenizer')
        check(initial['sampler']['draws'] == 1, 'Reference initialization draw changed')
        check(initial['contract']['binding_sha256'] == self.data.binding_sha256, 'Reference data differs')
        self.sampler.load_state_dict(initial['sampler'])
        self.initial_sampler_state = copy.deepcopy(initial['sampler'])
        self.initialization = dict(reference_sha256=self.plan['initial_checkpoint']['sha256'],
                                   saved_reference_step=0, sampler_exact=True)
        if arm == 'cnn':
            current = self.model.state_dict()
            copied = {}
            for key in current:
                if key.startswith(('tokenizer.encoder.', 'tokenizer.decoder.')):
                    copied[key] = initial['model'][key]
                elif key == 'tokenizer.vq.embedding.weight':
                    copied[key] = initial['model'][key][:self.model.config.vocab]
            self.model.load_state_dict(copied, strict=False)
            for key, tensor in copied.items():
                torch.testing.assert_close(self.model.state_dict()[key].cpu(), tensor, rtol=0, atol=0)
            self.initialization.update(encoder_decoder_exact=True, first_codebook_rows_exact=True)
            historical_rng = copy.deepcopy(initial['rng'])
            if self.device.type != 'cuda': historical_rng['cuda'] = []
            restore_rng(historical_rng)
        else:
            self.initialization.update(encoder_decoder_exact=False,
                codebook='Existing first flattened feature rows plus jitter, freshly initialized ST features')
        self.contract['small_codebook_architecture'] = dict(plan_sha256=file_hash(Path(root)/plan_path),
            arm=arm, reference_sha256=self.plan['initial_checkpoint']['sha256'], runner_sha256=source_hash(__file__))
        self.contract['backend_policy'] = copy.deepcopy(BACKEND_POLICY)
        if resume:
            state = torch.load(resume, map_location='cpu', weights_only=True)
            check(state['contract'] == self.contract, 'Exact experiment resume contract changed')
            if state['sampler'].get('schema') == 'train-frame-coverage-v1':
                self._attach_coverage(reconstruct=True)
            self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler']); self.scaler.load_state_dict(state['scaler'])
            self.sampler.load_state_dict(state['sampler']); self.step = state['step']; self.best = state['best']
            self.spent_seconds = state['spent_seconds']; self.spent_units = state['spent_units']
            restore_rng(state['rng']); self.last_failure = state.get('last_failure')
        self.inactive = {k: v.detach().clone() for k, v in self.model.state_dict().items() if not k.startswith('tokenizer.')}

    def _attach_coverage(self, reconstruct=False):
        if isinstance(self.sampler, CoverageSampler):
            return
        if reconstruct:
            self.sampler.load_state_dict(self.initial_sampler_state)
            for _ in range(self.plan['coverage_after_update']*self.microbatch*self.accumulation):
                self.sampler.draw()
        state = self.sampler.state_dict()
        expected = self.plan.get('historical_parent_sampler_sha256')
        if expected:
            check(sampler_state_sha256(state) == expected, 'Historical data-order boundary differs')
        self.sampler = CoverageSampler(self.data, state, seed=self.plan['coverage_seed'],
            expected_parent_sampler_sha256=sampler_state_sha256(state))
        self.coverage_parent_state = copy.deepcopy(state)
        expected_plan = self.plan.get('historical_coverage_plan_sha256')
        if expected_plan:
            check(self.sampler.plan_sha256 == expected_plan, 'Historical coverage plan differs')

    def update(self, budget=None):
        if self.step == self.plan['coverage_after_update']:
            self._attach_coverage()
        return super().update(budget)

    def verify_inactive(self):
        for key, tensor in self.inactive.items():
            torch.testing.assert_close(self.model.state_dict()[key], tensor, rtol=0, atol=0)

    def _failure_diagnostics(self, boundary, phase, microbatches, error):
        if not isinstance(self.sampler, CoverageSampler):
            return super()._failure_diagnostics(boundary, phase, microbatches, error)
        replay = CoverageSampler(self.data, self.coverage_parent_state, seed=self.plan['coverage_seed'],
            expected_parent_sampler_sha256=sampler_state_sha256(self.coverage_parent_state))
        replay.load_state_dict(boundary['sampler'])
        windows = [self.data.descriptor(*replay.draw()) for _ in range(self.sampler.draws-boundary['sampler']['draws'])]
        affected = [dict(name=name, nonfinite=int((~torch.isfinite(p.grad)).sum()))
                    for name, p in self.model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        return dict(attempted_step=boundary['step']+1, phase=phase, completed_microbatches=microbatches,
            exception=type(error).__name__, message=str(error), precision=self.precision,
            scale=self.scaler.get_scale(), sampled_windows=windows, affected_gradients=affected,
            restored_pre_update_boundary=False)


def precision_check(trainer, budget):
    from genie_diep.gpu_precision import _backward_case
    check(budget.permits(120), 'Insufficient precision-pilot budget')
    state = copy.deepcopy(trainer.sampler.state_dict()); random = rng_state()
    try:
        batches = [trainer.sampler.batch(trainer.microbatch) for _ in range(trainer.accumulation)]
        reports = {}
        for precision, scale in (('fp32', 1.), ('fp16', 1024.)):
            reports[precision] = _backward_case(trainer.model, batches, precision, scale,
                lambda model, x: trainer.loss(x), budget)
            check(reports[precision]['status'] == 'finite', 'CUDA precision pilot failed: '+precision)
        relative = abs(reports['fp16']['unscaled_gradient_norm']/reports['fp32']['unscaled_gradient_norm']-1)
        check(relative <= trainer.plan['maximum_fp16_gradient_norm_relative_error'], 'FP16 gradient norm differs beyond pilot bound')
        return dict(passed=True, cases=reports, gradient_norm_relative_error=relative,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        trainer.sampler.load_state_dict(state); restore_rng(random); trainer.optimizer.zero_grad(set_to_none=True)


def replay_check(trainer, out, budget):
    """One provisional update, saved-state replay, then restore the saved boundary.

    All replay work counts against the shared allocation. Only the replayed
    update remains in the eventual sequence of unique training updates.
    """
    before = out/'replay-before.pt'; trainer.save(before)
    start = time.monotonic(); loss = trainer.update(budget)
    check(loss is not None, 'Budget exhausted during replay preflight')
    if trainer.device.type == 'cuda': torch.cuda.synchronize()
    seconds = time.monotonic()-start
    after = out/'replay-after.pt'; trainer.save(after)
    expected = torch.load(after, map_location='cpu', weights_only=True)
    state = torch.load(before, map_location='cpu', weights_only=True)
    trainer.model.load_state_dict(state['model']); trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.scheduler.load_state_dict(state['scheduler']); trainer.scaler.load_state_dict(state['scaler'])
    trainer.sampler.load_state_dict(state['sampler']); trainer.step = state['step']; restore_rng(state['rng'])
    actual_loss = trainer.update(budget)
    check(actual_loss == loss, 'Replay loss changed')
    trainer.save(out/'replay-actual.pt')
    actual = torch.load(out/'replay-actual.pt', map_location='cpu', weights_only=True)
    for key in ('model', 'optimizer', 'scheduler', 'scaler', 'sampler', 'rng', 'step'):
        compare_exact(actual[key], expected[key])
    return dict(passed=True, seconds=seconds, loss=loss, maximum_state_difference=0,
                unique_updates=1, extra_optimizer_calls=1)


def measure_training_updates(trainer, budget, count):
    """Time genuine counted updates after replay; retain their resulting state."""
    first_step = trainer.step
    seconds = []
    for _ in range(count):
        if not budget.permits((sum(seconds)/len(seconds))*1.3 if seconds else 0):
            raise TimeoutError('Shared budget exhausted during steady timing')
        if trainer.device.type == 'cuda': torch.cuda.synchronize()
        start = time.monotonic()
        value = trainer.update(budget)
        if trainer.device.type == 'cuda': torch.cuda.synchronize()
        if value is None: raise TimeoutError('Timing update rolled back at shared budget boundary')
        seconds.append(time.monotonic()-start)
    return dict(first_step=first_step+1, last_step=trainer.step, unique_updates=len(seconds),
                seconds=seconds, mean_seconds=sum(seconds)/len(seconds))


def update_seconds(entry):
    return entry['timing']['mean_seconds'] if 'timing' in entry else entry['replay']['seconds']


def run(args):
    configure_backend()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    plan = validate_plan(root, args.plan)
    check(args.resume or not out.exists(), 'Choose a fresh output directory')
    if args.preflight:
        check(args.device == 'cpu', 'Local preflight is CPU-only')
    else:
        check(args.authorized and args.device == 'cuda' and torch.cuda.is_available(), 'Authorized CUDA required')
        check(math.isfinite(args.rate) and args.rate > 0 and args.balance is not None and args.balance > 50,
              'Live rate and reserve required')
        check(args.allocation_start is not None and 0 < args.allocation_start <= time.time(), 'Allocation timestamp required')
        check('T4' in torch.cuda.get_device_name(), 'This plan requires the measured T4 class')
    torch.set_num_threads(args.threads); out.mkdir(parents=True, exist_ok=args.resume)
    budget = Budget(dict(max_allocated_hours=plan['limits']['maximum_allocated_hours'],
                         max_compute_units=plan['limits']['maximum_compute_units']),
                    rate=args.rate, balance=args.balance,
                    setup_seconds=max(0, time.time()-args.allocation_start) if args.allocation_start else 0,
                    reserve_seconds=240 if not args.preflight else 0)
    if args.resume:
        report = read(out/'experiment-report.json')
        check(report['plan_sha256'] == file_hash(root/args.plan) and report['preflight'] == args.preflight,
              'Resume plan or execution mode differs')
        check(report['implementation_sha256'] == implementation_hash() and report['limits'] == plan['limits'],
              'Resume source or allocation bounds differ')
        check(args.preflight or report['displayed_units_per_hour'] == args.rate,
              'Resume live rate changed; explicit prior-unit accounting is required before continuing')
        check(args.preflight or time.time()-args.allocation_start >= report['spent_seconds'],
              'Resume must retain all prior allocation debits in accounting start')
        for entry in report['arms'].values():
            if 'checkpoint' in entry:
                check(file_hash(out/entry['checkpoint']['path']) == entry['checkpoint']['sha256'], 'Resume checkpoint changed')
        report['status'] = 'running'; report.pop('error', None)
    else:
        report = dict(status='running', preflight=args.preflight, plan_sha256=file_hash(root/args.plan),
            implementation_sha256=implementation_hash(), device=args.device,
            gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
            limits=plan['limits'], arms={}, quality_gate='unmeasured', promotion_approved=False)
        report['displayed_units_per_hour'] = args.rate
        report['backend_policy'] = copy.deepcopy(BACKEND_POLICY)
    def write_report():
        report['spent_seconds'], report['estimated_units'] = budget.spent()
        path = out/'experiment-report.json'; temporary = out/'experiment-report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(path)
    def checkpoint(trainer, folder, entry):
        path = folder/f'step-{trainer.step:06d}.pt'
        trainer.save(path, budget)
        entry['checkpoint'] = dict(path=path.relative_to(out).as_posix(), sha256=file_hash(path))
        entry['latest_step'] = trainer.step
        write_report()
    target = 2 if args.preflight else plan['unique_updates_per_arm']
    try:
        # Pilot BOTH architectures before spending on either full training run.
        for arm in ARMS:
            if report['arms'].get(arm, {}).get('pilot_passed'): continue
            if not budget.permits(180): raise TimeoutError('Insufficient budget for next arm')
            folder = out/arm; folder.mkdir(exist_ok=args.resume)
            trainer = ArchitectureTrainer(root, args.plan, arm, args.device, args.preflight)
            trainer.save(folder/'initial.pt')
            arm_report = report['arms'][arm] = dict(initialization=trainer.initialization, latest_step=0,
                extra_replay_calls=0, saved_steps=[], evaluations=[], frozen_action_dynamics=True)
            if not args.preflight:
                torch.cuda.reset_peak_memory_stats()
                arm_report['precision'] = precision_check(trainer, budget)
            arm_report['replay'] = replay_check(trainer, folder, budget)
            arm_report['extra_replay_calls'] = 1
            timing_updates = min(plan.get('timing_updates_per_arm', 0), target-trainer.step)
            if timing_updates:
                arm_report['timing'] = measure_training_updates(trainer, budget, timing_updates)
            arm_report['pilot_passed'] = True
            checkpoint(trainer, folder, arm_report)
            del trainer
            if args.device == 'cuda': torch.cuda.empty_cache()
        remaining = sum(max(0, target-r['latest_step'])*update_seconds(r)*1.3 for r in report['arms'].values())
        if not args.preflight and not budget.permits(remaining+300):
            raise TimeoutError('Both measured arm timings cannot fit shared cap')
        report['measured_training_seconds_with_margin'] = remaining
        write_report()
        for arm in ARMS:
            arm_report = report['arms'][arm]
            if arm_report.get('completed'): continue
            folder = out/arm
            trainer = ArchitectureTrainer(root, args.plan, arm, args.device, args.preflight,
                resume=out/arm_report['checkpoint']['path'])
            last_save = time.monotonic()
            while trainer.step < target:
                if not budget.permits(update_seconds(arm_report)*1.3): raise TimeoutError('Shared budget exhausted')
                value = trainer.update(budget)
                if value is None: raise TimeoutError('Shared budget ended mid-update; rollback applied')
                arm_report['latest_step'] = trainer.step
                if trainer.step in plan['save_steps'] or trainer.step == target or time.monotonic()-last_save >= 300:
                    trainer.verify_inactive(); checkpoint(trainer, folder, arm_report); last_save = time.monotonic()
                    arm_report['saved_steps'].append(trainer.step)
                    arm_report['evaluations'].append(trainer.evaluate(limit=plan['progress_validation_windows'], budget=budget))
                    if isinstance(trainer.sampler, CoverageSampler): arm_report['coverage'] = trainer.sampler.plan_stats()
                    write_report()
                if trainer.step % 25 == 0 or trainer.step == target:
                    print(json.dumps(dict(arm=arm, step=trainer.step, loss=value, remaining_seconds=budget.remaining())), flush=True)
            trainer.verify_inactive(); trainer.save(folder/'last.pt', budget)
            arm_report['completed'] = True; write_report()
            del trainer
            if args.device == 'cuda': torch.cuda.empty_cache()
        report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'budget_exhausted' if isinstance(error, TimeoutError) else 'failed'
        report['error'] = repr(error)
        if 'trainer' in locals():
            trainer.verify_inactive(); checkpoint(trainer, folder, report['arms'][arm])
            trainer.save(folder/'last.pt', budget)
            report['arms'][arm]['latest_step'] = trainer.step
            report['arms'][arm]['failure'] = trainer.last_failure
        raise
    finally:
        write_report()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.'); p.add_argument('--plan', default='configs/small-codebook-architecture-v1.json')
    p.add_argument('--out', required=True); p.add_argument('--device', default='cpu'); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--preflight', action='store_true'); p.add_argument('--authorized', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--rate', type=float, default=0); p.add_argument('--balance', type=float); p.add_argument('--allocation-start', type=float)
    run(p.parse_args())
