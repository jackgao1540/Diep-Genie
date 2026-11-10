"""Real-data CPU execution checks, never a performance or training-success claim."""
import argparse
import gc
import time
from pathlib import Path
import torch
from genie_diep.baseline import Trainer, implementation_hash
from genie_diep.baseline_evaluation import oracle_interventions, commanded_rollout
from scripts.build_stationary_recovery import write
from genie_diep.data import file_hash


def run(output='runs/training-readiness-v1/execution'):
    torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    reports=[]
    for stage in ['tokenizer','actions','dynamics']:
        start=time.monotonic();trainer=Trainer('.','configs/training-baseline-v1.json',stage,preflight=True,microbatch=1,accumulation=2)
        frozen={n:p.detach().clone() for n,p in trainer.model.named_parameters() if not p.requires_grad}
        loss1=trainer.update();path=out/f'{stage}-resume.pt';trainer.save(path)
        loss2=trainer.update();expected={n:p.detach().clone() for n,p in trainer.model.named_parameters()};schedule=trainer.scheduler.state_dict();draws=trainer.sampler.draws
        del trainer;gc.collect()
        restored=Trainer('.','configs/training-baseline-v1.json',stage,preflight=True,microbatch=1,accumulation=2,resume=path)
        actual=restored.update();assert actual==loss2
        for n,p in restored.model.named_parameters():
            assert torch.equal(p,expected[n]),n
            if n in frozen:assert torch.equal(p,frozen[n]),'Frozen stage changed: '+n
        assert restored.scheduler.state_dict()==schedule and restored.sampler.draws==draws
        report=restored.evaluate(limit=2);restored.save(out/f'{stage}-checked.pt')
        reports.append(dict(stage=stage,finite_losses=[loss1,loss2],exact_resume=True,frozen_modules_unchanged=True,
            full_model_parameters=sum(p.numel() for p in restored.model.parameters()),microbatch=1,accumulation=2,
            validation_diagnostic=report,seconds=time.monotonic()-start))
        print(reports[-1],flush=True)
        if stage=='actions':
            item=restored.evaluation['validation'][0];x=restored.validation.window(*restored.validation.locate(item))[None]
            diagnostic=oracle_interventions(restored.model,x);assert len(diagnostic['alternatives'])==17
            with torch.no_grad():_,codes,_=restored.model.lam.infer(x[:,-2:])
            rollout=commanded_rollout(restored.model,x,codes,10,'actions');assert torch.isfinite(rollout).all()
            write(out/'action-api-check.json',dict(oracle_alternatives=17,commanded_frames=10,finite=True,semantic_quality='unmeasured'))
        del restored,expected,frozen;gc.collect()
    write(out/'report.json',dict(status='passed',purpose='CPU execution and resume only; not trained model performance',
        implementation_sha256=implementation_hash(),spec_sha256=file_hash('configs/training-baseline-v1.json'),
        binding_sha256=file_hash('configs/baseline-data-v9.json'),evaluation_sha256=file_hash('configs/baseline-evaluation-v1.json'),
        stages=reports,gpu_fp16_memory_throughput='requires bounded GPU pilot',training_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/training-readiness-v1/execution')
    run(parser.parse_args().out)
