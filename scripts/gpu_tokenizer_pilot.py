"""First bounded GPU measurement: no stage promotion and no quality claims."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from genie_diep.baseline import Budget, Trainer, validate_readiness, implementation_hash
from genie_diep.data import file_hash
from genie_diep.cohort import read


def run(args):
    if not torch.cuda.is_available():raise RuntimeError('An authorized GPU allocation is required')
    if args.rate<=0 or args.balance<=50:raise ValueError('Enter the live displayed rate/balance')
    validate_readiness('.',args.spec)
    out=Path(args.out)
    if out.exists():raise FileExistsError('Use a new pilot output directory')
    out.mkdir(parents=True)
    budget=Budget(dict(max_allocated_hours=1,max_compute_units=2),rate=args.rate,balance=args.balance,
                  setup_seconds=max(0,time.time()-args.allocation_start),reserve_seconds=180)
    report=dict(status='incomplete',gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
        implementation_sha256=implementation_hash(),spec_sha256=file_hash(args.spec),binding_sha256=file_hash(read(args.spec)['data_binding']),
        purpose='Memory, throughput, FP16 and resume measurement only; not model quality',attempts=[])
    try:
        for microbatch in [4,2,1]:
            if not budget.permits(120):raise RuntimeError('Insufficient pilot budget remains')
            trainer=None;restored=None;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            started=time.monotonic()
            try:
                trainer=Trainer('.',args.spec,'tokenizer','cuda','fp16',microbatch,16//microbatch,True)
                first=trainer.update(budget)
                if first is None:raise RuntimeError('Pilot budget reached')
                trainer.save(out/'resume.pt',budget);torch.cuda.synchronize();t=time.monotonic()
                second=trainer.update(budget)
                if second is None:raise RuntimeError('Pilot budget reached')
                torch.cuda.synchronize();update_seconds=time.monotonic()-t
                expected={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
                peak=torch.cuda.max_memory_allocated();del trainer;trainer=None;gc.collect();torch.cuda.empty_cache()
                restored=Trainer('.',args.spec,'tokenizer','cuda','fp16',microbatch,16//microbatch,True,resume=out/'resume.pt')
                repeated=restored.update(budget)
                if repeated is None:raise RuntimeError('Pilot budget reached')
                maximum=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in restored.model.state_dict().items())
                for k,v in restored.model.state_dict().items():torch.testing.assert_close(v.detach().cpu(),expected[k],rtol=1e-4,atol=1e-6)
                restored.save(out/'checked.pt',budget)
                report.update(status='passed',microbatch=microbatch,accumulation=16//microbatch,
                    optimizer_update_seconds=update_seconds,sequences_per_second=16/update_seconds,
                    peak_allocated_bytes=peak,resume_max_absolute_difference=maximum,
                    resume_kind='exact' if maximum==0 else 'within FP16 tolerance',finite_losses=[first,second,repeated],
                    quality_gates='unmeasured',training_stage_promotion=False)
                report['attempts'].append(dict(microbatch=microbatch,status='passed',seconds=time.monotonic()-started));break
            except torch.cuda.OutOfMemoryError:
                report['attempts'].append(dict(microbatch=microbatch,status='out_of_memory',seconds=time.monotonic()-started))
                del trainer,restored;gc.collect();torch.cuda.empty_cache()
        if report['status']!='passed':raise RuntimeError('No supported microbatch fits; stop without changing resolution')
    finally:
        report['spent_seconds'],report['estimated_units']=budget.spent()
        (out/'pilot-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',default='configs/training-baseline-v1.json');p.add_argument('--out',required=True)
    p.add_argument('--rate',type=float,required=True);p.add_argument('--balance',type=float,required=True);p.add_argument('--allocation-start',type=float,required=True)
    run(p.parse_args())
