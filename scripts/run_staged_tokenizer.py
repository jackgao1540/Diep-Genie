"""Bounded 16-code staged tokenizer execution; quality remains unmeasured.

Pilot and full use the same transactional updates. Full requires an independent
budget receipt and starts fresh; pilot decoder weights are never promoted.
The notebook owns the outer watchdog, export and runtime release.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from genie_diep.baseline import Trainer, Budget, implementation_hash, restore_rng, rng_state
from genie_diep.cohort import read
from genie_diep.coverage_sampler import CoverageSampler, sampler_state_sha256
from genie_diep.data import file_hash
from genie_diep.model import image_loss
from scripts.run_small_codebook_architecture import (
    BACKEND_POLICY, configure_backend, check, replay_check,
    measure_training_updates,
)
from scripts.run_tokenizer_comparison import source_hash, compare_exact

PHASES = ('joint', 'decoder_only')
BOUND_SOURCES = (
    'genie_diep/baseline.py', 'genie_diep/cohort.py', 'genie_diep/coverage_sampler.py',
    'genie_diep/data.py', 'genie_diep/model.py', 'genie_diep/transformer_tokenizer.py',
    'genie_diep/gpu_precision.py', 'scripts/diagnose_tokenizer_precision.py',
    'scripts/run_small_codebook_architecture.py', 'scripts/run_tokenizer_comparison.py',
)
PILOT_LIMITS = dict(max_compute_units=1.0, max_allocated_hours=.75,
                    minimum_remaining_units=50)


def precision_check(trainer, budget):
    from genie_diep.gpu_precision import _backward_case
    sampler, random = copy.deepcopy(trainer.sampler.state_dict()), rng_state()
    try:
        batches = [trainer.sampler.batch(trainer.microbatch) for _ in range(trainer.accumulation)]
        cases = {}
        for precision, scale in (('fp32', 1.), ('fp16', 1024.)):
            case = cases[precision] = _backward_case(trainer.model, batches, precision, scale,
                lambda model, x: trainer.loss(x), budget)
            if case['status'] == 'runtime_error' and 'out of memory' in case.get('error', '').lower():
                raise torch.cuda.OutOfMemoryError(case['error'])
            check(case['status'] == 'finite', 'Precision comparison failed: ' + json.dumps(case))
        relative = abs(cases['fp16']['unscaled_gradient_norm'] / cases['fp32']['unscaled_gradient_norm'] - 1)
        check(relative <= trainer.plan['maximum_fp16_gradient_norm_relative_error'], 'FP16 gradient norm differs')
        return dict(passed=True, cases=cases, gradient_norm_relative_error=relative,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        trainer.sampler.load_state_dict(sampler); restore_rng(random); trainer.optimizer.zero_grad(set_to_none=True)


def validate_plan(root, plan_path):
    root = Path(root); plan = read(root / plan_path)
    check(plan['schema_version'] == 1 and plan['phases'] == list(PHASES), 'Plan schema changed')
    check(plan['implementation_sha256'] == implementation_hash(), 'Implementation changed')
    check(plan['runner_sha256'] == source_hash(__file__), 'Runner changed')
    check(set(plan['source_sha256']) == set(BOUND_SOURCES), 'Source inventory changed')
    for name in BOUND_SOURCES:
        check(source_hash(root / name) == plan['source_sha256'][name], 'Bound source changed: ' + name)
    check(plan['backend_policy'] == BACKEND_POLICY, 'Backend policy changed')
    check(plan['pilot_limits'] == PILOT_LIMITS, 'Pilot authorization limits changed')
    check(plan['full_updates_per_phase'] == 5000 and plan['timing_updates'] == 5, 'Update targets changed')
    check(plan['coverage_after_update'] == 999 and plan['coverage_seed'] == 20261110, 'Data schedule changed')
    check(plan['microbatch_candidates'] == [4, 2, 1] and plan['sequences_per_update'] == 16,
          'Effective update batch changed')
    for phase in PHASES:
        entry = plan['stages'][phase]
        check(file_hash(root / entry['spec']) == entry['spec_sha256'], 'Stage specification changed')
        spec = read(root / entry['spec'])
        check(spec['model']['vocab'] == 16 and spec['model']['context'] == 16, 'Code/context changed')
        check(spec['stages'][0]['max_updates'] == 20000, 'Historical scheduler horizon changed')
    return plan


class StagedTrainer(Trainer):
    def __init__(self, root, plan_path, phase, device='cpu', microbatch=4,
                 handoff=None, resume=None, local_check=False, mode='pilot'):
        configure_backend()
        self.plan = validate_plan(root, plan_path)
        check(phase in PHASES and microbatch in self.plan['microbatch_candidates'], 'Invalid phase/batch')
        self.phase = phase
        # The explicit staged plan/authorization replaces the older readiness
        # document gate. Baseline preflight bypass is internal only; the saved
        # contract below records the real mode and local-check status.
        super().__init__(root, self.plan['stages'][phase]['spec'], 'tokenizer', device,
                         'fp16' if torch.device(device).type == 'cuda' else 'fp32',
                         microbatch=microbatch, accumulation=16 // microbatch, preflight=True)
        self.preflight = local_check
        self.contract.update(preflight=local_check, staged_mode=mode, phase=phase,
            plan_sha256=file_hash(Path(root) / plan_path), runner_sha256=source_hash(__file__),
            backend_policy=copy.deepcopy(BACKEND_POLICY), execution_gate='staged-plan-v1')
        self.initial_sampler_state = copy.deepcopy(self.sampler.state_dict())
        self.handoff = None
        if phase == 'decoder_only':
            check(handoff is not None, 'Decoder requires the actual joint checkpoint')
            state = torch.load(handoff, map_location='cpu', weights_only=True)
            check(state['contract']['phase'] == 'joint' and state['contract']['staged_mode'] == mode,
                  'Handoff phase/mode differs')
            check(state['contract']['plan_sha256'] == self.contract['plan_sha256'] and
                  state['contract']['binding_sha256'] == self.data.binding_sha256, 'Handoff provenance changed')
            check(mode != 'full' or state['step'] == 5000, 'Full decoder requires final full joint step 5000')
            self.handoff = file_hash(handoff)
            for component in ('encoder', 'vq'):
                prefix = 'tokenizer.' + component + '.'
                module = getattr(self.module, component)
                module.load_state_dict({k[len(prefix):]: v for k, v in state['model'].items()
                                        if k.startswith(prefix)}, strict=True)
                module.requires_grad_(False).eval()
            self.parameters = list(self.module.decoder.parameters())
            self.optimizer = torch.optim.AdamW(self.parameters, lr=self.stage_spec['learning_rate'],
                                              weight_decay=self.spec['optimization']['weight_decay'])
            # Historical schedule: 500-step warmup / 20,000-step cosine horizon;
            # the orchestrator separately stops each probe at 5,000 updates.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                lambda s: (s+1)/500 if s < 500 else .1+.9*(1+math.cos(math.pi*min(1,(s-500)/19500)))/2)
        self.contract['handoff_sha256'] = self.handoff
        if resume:
            state = torch.load(resume, map_location='cpu', weights_only=True)
            check(state['contract'] == self.contract, 'Exact staged resume contract differs')
            if state['sampler'].get('schema') == 'train-frame-coverage-v1':
                self._attach_coverage(reconstruct=True)
            self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler']); self.scaler.load_state_dict(state['scaler'])
            self.sampler.load_state_dict(state['sampler']); self.step = state['step']; self.best = state['best']
            self.spent_seconds = state['spent_seconds']; self.spent_units = state['spent_units']
            restore_rng(state['rng']); self.last_failure = state.get('last_failure')
        self.inactive = {k: v.detach().clone() for k, v in self.model.state_dict().items()
            if not k.startswith('tokenizer.') or
            (phase == 'decoder_only' and k.startswith(('tokenizer.encoder.', 'tokenizer.vq.')))}
        check(sum(p.numel() for p in self.module.encoder.parameters()) == self.spec['parameter_counts']['encoder'], 'Encoder size changed')
        check(sum(p.numel() for p in self.module.decoder.parameters()) == self.spec['parameter_counts']['decoder'], 'Decoder size changed')

    def _attach_coverage(self, reconstruct=False):
        if isinstance(self.sampler, CoverageSampler): return
        if reconstruct:
            self.sampler.load_state_dict(self.initial_sampler_state)
            for _ in range(self.plan['coverage_after_update'] * 16): self.sampler.draw()
        parent = self.sampler.state_dict()
        self.sampler = CoverageSampler(self.data, parent, seed=self.plan['coverage_seed'],
            expected_parent_sampler_sha256=sampler_state_sha256(parent))

    def update(self, budget=None):
        # Attach before baseline snapshots the transaction, so rollback keeps the
        # exact coverage cursor even at the mixed -> coverage boundary.
        if self.step == self.plan['coverage_after_update']: self._attach_coverage()
        value = super().update(budget)
        self.verify_inactive()
        return value

    def loss(self, x):
        if self.phase == 'joint': return super().loss(x)
        self.module.encoder.eval(); self.module.vq.eval()
        # FP32 frozen encoding prevents AMP rounding from changing the actual
        # discrete IDs seen by the replacement decoder across precision checks.
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            _, ids, _ = self.module.encode(x.float())
        return 10 * image_loss(self.module.decode_ids(ids), x, color_balance=True)

    def verify_inactive(self):
        for key, tensor in self.inactive.items():
            torch.testing.assert_close(self.model.state_dict()[key], tensor, rtol=0, atol=0)

    def freeze_reference(self):
        descriptor = self.evaluation['training_calibration'][0]
        x = self.data.window(*self.data.locate(descriptor))[None].to(self.device)
        was_training = self.module.training
        self.module.eval()
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            _, ids, _ = self.module.encode(x.float())
            embeddings = self.module.vq.lookup(ids)
        self.module.train(was_training)
        if self.phase == 'decoder_only': self.module.encoder.eval(); self.module.vq.eval()
        return dict(source=descriptor, ids=ids.cpu(), embeddings=embeddings.cpu())


def allocation_budget(args):
    check(args.mode in ('pilot', 'full'), 'Unknown run mode')
    limits = PILOT_LIMITS
    if args.mode == 'full':
        check(args.full_budget is not None, 'Full training inaccessible without an independent budget receipt')
        receipt = read(args.full_budget)
        check(receipt.get('authorized') is True and receipt.get('purpose') == 'staged16-full-5000-each',
              'Full budget receipt is not this experiment authorization')
        limits = {k: receipt[k] for k in PILOT_LIMITS}
        check(limits['minimum_remaining_units'] == 50 and all(
              math.isfinite(limits[k]) and limits[k] > 0 for k in ('max_compute_units','max_allocated_hours')),
              'Invalid full budget')
        check(limits['max_compute_units'] <= 30 and limits['max_allocated_hours'] <= 26,
              'Full budget exceeds the independent 30-CU/26-hour authorization')
    check(all(math.isfinite(v) and v >= 0 for v in (args.prior_seconds, args.prior_units)), 'Invalid prior debits')
    if args.local_check:
        check(args.device == 'cpu' and not args.authorized, 'Local check is CPU only')
        return Budget(dict(max_allocated_hours=.75, max_compute_units=1), reserve_seconds=0), limits
    check(args.authorized and args.device == 'cuda' and torch.cuda.is_available(), 'Authorized CUDA required')
    check(math.isfinite(args.rate) and args.rate > 0 and args.balance is not None and args.balance > 50,
          'Live rate/balance required')
    check(args.allocation_start is not None and math.isfinite(args.allocation_start) and
          0 < args.allocation_start <= time.time(), 'Actual allocation start required')
    return Budget(limits, prior_seconds=args.prior_seconds, prior_units=args.prior_units,
        rate=args.rate, balance=args.balance, setup_seconds=time.time()-args.allocation_start,
        reserve_seconds=180), limits


def full_pilot_gate(root, args, plan, budget, phase_steps=None):
    """Bind measured fit/speed to current conditions before full compute work."""
    receipt = read(args.full_budget)
    reference = receipt['pilot']
    path = Path(reference['report_path'])
    if not path.is_absolute(): path = Path(root) / path
    check(file_hash(path) == reference['sha256'], 'Pilot report changed')
    pilot = read(path)
    check(pilot['status'] == 'pilot_complete' and not pilot['identity']['local_check'],
          'A completed real CUDA pilot is required')
    check(pilot['identity']['plan_sha256'] == file_hash(Path(root) / args.plan), 'Pilot plan differs')
    check(pilot['identity']['mode'] == 'pilot' and pilot['identity']['rate'] == args.rate,
          'Pilot rate/mode differs')
    check(pilot['gpu'] == torch.cuda.get_device_name() and
          pilot['torch_version'] == str(torch.__version__) and pilot['cuda_version'] == torch.version.cuda,
          'GPU or numerical runtime changed; repeat the bounded pilot')
    check(receipt['minimum_prior_seconds'] >= pilot['spent_seconds'] and
          receipt['minimum_prior_units'] >= pilot['spent_units'], 'Receipt omits pilot debits')
    check(args.prior_seconds >= receipt['minimum_prior_seconds'] and
          args.prior_units >= receipt['minimum_prior_units'], 'Full run loses prior allocation debits')
    forecast = 600.
    batches = {}
    for phase in PHASES:
        entry = pilot['phases'][phase]
        check(entry['complete'] and entry['pilot_passed'] and entry['precision']['passed'] and
              entry['replay']['passed'], 'Pilot checks incomplete')
        check(entry['microbatch'] in plan['microbatch_candidates'] and
              entry['microbatch'] * entry['accumulation'] == 16, 'Pilot update batch differs')
        timing = entry['timing']
        check(timing['unique_updates'] == 5 and len(timing['seconds']) == 5 and
              all(math.isfinite(v) and v > 0 for v in timing['seconds']), 'Measured timing missing')
        mean = sum(timing['seconds']) / 5
        check(mean == timing['mean_seconds'], 'Pilot timing summary changed')
        remaining = 5000 - (phase_steps or {}).get(phase, 0)
        check(0 <= remaining <= 5000, 'Invalid full resume step')
        # Allow every 500-step safe save plus initial/replay/handoff overhead.
        save_cost = max(entry.get('checkpoint_seconds', [0]))
        forecast += remaining * mean * 1.3 + (math.ceil(remaining/500) + 4) * save_cost * 1.3
        batches[phase] = entry['microbatch']
    check(budget.permits(forecast), 'Measured full forecast cannot fit remaining allocation limits')
    return dict(pilot_sha256=reference['sha256'], estimated_remaining_seconds_with_margin=forecast,
                microbatches=batches)


def validate_output(root, out, device, private_output_root=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.is_relative_to(root / 'runs'): return out
    check(device == 'cuda' and private_output_root is not None,
          'Outputs must stay private under runs or an explicit CUDA artifact root')
    private = Path(private_output_root).resolve()
    check(out != private and out.is_relative_to(private), 'Output escapes the private artifact root')
    return out


def run(args):
    configure_backend(); torch.set_num_threads(args.threads)
    root = Path(args.root).resolve()
    out = validate_output(root, args.out, args.device, args.private_output_root)
    plan = validate_plan(root, args.plan)
    budget, limits = allocation_budget(args)
    report_path = out / 'experiment-report.json'
    check(args.resume or not out.exists(), 'Use fresh outputs or explicit resume')
    out.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(plan_sha256=file_hash(root / args.plan), mode=args.mode,
        local_check=args.local_check, limits=limits, rate=args.rate,
        budget_sha256=file_hash(args.full_budget) if args.full_budget else None)
    if args.resume:
        report = read(report_path); check(report['identity'] == identity, 'Resume identity changed')
        spent_seconds, spent_units = budget.spent()
        check(args.local_check or (spent_seconds >= report['spent_seconds'] and
              spent_units + 1e-9 >= report['spent_units']), 'Resume loses allocation debits')
    else:
        report = dict(identity=identity, status='running', phases={}, quality='unmeasured',
            gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            full_training_authorized=args.mode == 'full', backend_policy=BACKEND_POLICY)
    full_gate = None
    if args.mode == 'full':
        full_gate = full_pilot_gate(root, args, plan, budget,
            {phase: entry.get('latest_step', 0) for phase, entry in report['phases'].items()})
        report['full_pilot_gate'] = full_gate
    def write():
        report['spent_seconds'], report['spent_units'] = budget.spent()
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(report_path)
    def checkpoint(trainer, folder, entry):
        started = time.monotonic()
        path = folder / f'step-{trainer.step:06d}.pt'; trainer.save(path, budget)
        entry.setdefault('checkpoint_seconds', []).append(time.monotonic() - started)
        entry.update(checkpoint=dict(path=path.relative_to(out).as_posix(), sha256=file_hash(path)),
                     latest_step=trainer.step)
        entry.setdefault('saved_checkpoints', {})[str(trainer.step)] = dict(entry['checkpoint'])
        if isinstance(trainer.sampler, CoverageSampler): entry['coverage'] = trainer.sampler.plan_stats()
        write()
    target = 1 if args.local_check else (6 if args.mode == 'pilot' else 5000)
    handoff = None
    write()
    try:
        for phase in PHASES:
            folder = out / phase; folder.mkdir(exist_ok=True)
            entry = report['phases'].setdefault(phase, {})
            if entry.get('complete'):
                handoff = out / entry['checkpoint']['path']
                check(file_hash(handoff) == entry['checkpoint']['sha256'], 'Completed checkpoint changed')
                if phase == 'joint':
                    check(file_hash(folder / 'handoff-reference.pt') == entry['reference_sha256'],
                          'Handoff reference changed')
                continue
            trainer = None
            resume = out / entry['checkpoint']['path'] if 'checkpoint' in entry else None
            if resume: check(file_hash(resume) == entry['checkpoint']['sha256'], 'Resume checkpoint changed')
            candidates = ([entry['microbatch']] if resume else
                [full_gate['microbatches'][phase]] if full_gate else plan['microbatch_candidates'])
            for microbatch in candidates:
                try:
                    trainer = StagedTrainer(root, args.plan, phase, args.device, microbatch,
                        handoff, resume, args.local_check, args.mode)
                    entry['microbatch'] = microbatch; entry['accumulation'] = 16 // microbatch
                    entry['effective_frames_per_update'] = 256
                    if not entry.get('pilot_passed'):
                        trainer.save(folder / 'initial.pt', budget)
                        entry['initial_checkpoint_sha256'] = file_hash(folder / 'initial.pt')
                        if args.device == 'cuda':
                            torch.cuda.reset_peak_memory_stats()
                            entry['precision'] = precision_check(trainer, budget)
                        reference = trainer.freeze_reference() if phase == 'decoder_only' else None
                        if reference:
                            reference_path = out / 'joint' / 'handoff-reference.pt'
                            check(file_hash(reference_path) == report['phases']['joint']['reference_sha256'],
                                  'Handoff reference changed')
                            compare_exact(reference, torch.load(reference_path, map_location='cpu', weights_only=True))
                            entry['handoff_import_ids_embeddings_exact'] = True
                        entry['replay'] = replay_check(trainer, folder, budget)
                        if reference: compare_exact(trainer.freeze_reference(), reference)
                        if not args.local_check:
                            entry['timing'] = measure_training_updates(trainer, budget, 5)
                        if reference:
                            compare_exact(trainer.freeze_reference(), reference)
                            entry['frozen_ids_embeddings_exact'] = True
                        trainer.verify_inactive()
                        entry['pilot_passed'] = True
                        if args.device == 'cuda': entry['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                        checkpoint(trainer, folder, entry)
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()
                    if not oom or full_gate or resume or microbatch == 1 or entry.get('pilot_passed'): raise
                    entry.setdefault('fit_attempts', []).append(dict(microbatch=microbatch, error=str(error)))
                    # No accepted state yet: retry from seed, same data order,
                    # same effective batch. Failed work remains in wall budget.
                    del trainer; trainer = None; gc.collect()
                    if args.device == 'cuda': torch.cuda.empty_cache()
                    write()
            check(trainer is not None, 'No fitted trainer')
            last_save = time.monotonic()
            while trainer.step < target:
                estimate = entry.get('timing', {}).get('mean_seconds', 0) * 1.3
                if not budget.permits(estimate): raise TimeoutError('Allocation limit reached')
                if trainer.update(budget) is None: raise TimeoutError('Update rolled back at allocation limit')
                if trainer.step % 500 == 0 or time.monotonic() - last_save >= 300:
                    checkpoint(trainer, folder, entry); last_save = time.monotonic()
            checkpoint(trainer, folder, entry)
            handoff = out / entry['checkpoint']['path']
            if phase == 'joint':
                reference_path = folder / 'handoff-reference.pt'
                torch.save(trainer.freeze_reference(), reference_path.with_suffix('.tmp'))
                reference_path.with_suffix('.tmp').replace(reference_path)
                entry['reference_sha256'] = file_hash(reference_path)
            else:
                check(file_hash(out / 'joint' / 'handoff-reference.pt') ==
                      report['phases']['joint']['reference_sha256'], 'Handoff reference changed')
                reference = torch.load(out / 'joint' / 'handoff-reference.pt', weights_only=True)
                compare_exact(trainer.freeze_reference(), reference)
                entry['handoff_ids_embeddings_exact'] = True
            entry['complete'] = True
            write(); del trainer; gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()
        report['status'] = 'local_check_complete' if args.local_check else args.mode + '_complete'
        if args.mode == 'pilot':
            check(args.local_check or all('timing' in r for r in report['phases'].values()),
                  'Missing measured stage timing')
            report['full_forecast_seconds_with_30pct_margin'] = sum(
                5000 * r['timing']['mean_seconds'] * 1.3 for r in report['phases'].values()
                if 'timing' in r) + 600
            report['full_forecast_units_at_current_rate'] = report['full_forecast_seconds_with_30pct_margin'] * args.rate / 3600
            report['full_training_requires_independent_budget'] = True
        write()
    except BaseException as error:
        report['status'] = 'stopped'; report['error'] = dict(type=type(error).__name__, message=str(error))
        if 'trainer' in locals() and trainer is not None and trainer._checkpoint_safe:
            checkpoint(trainer, folder, entry)
        write(); raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--mode', choices=('pilot','full'), default='pilot')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--authorized', action='store_true'); parser.add_argument('--local-check', action='store_true')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--full-budget')
    parser.add_argument('--private-output-root')
    parser.add_argument('--rate', type=float, default=0); parser.add_argument('--balance', type=float)
    parser.add_argument('--allocation-start', type=float); parser.add_argument('--prior-seconds', type=float, default=0)
    parser.add_argument('--prior-units', type=float, default=0); parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())


if __name__ == '__main__': main()
