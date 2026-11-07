"""Diagnostic upper bound with a continuous, target-inferred action channel.

This deliberately removes the discrete bottleneck. Success cannot establish
playable actions: the continuous channel can carry arbitrary future information.
Saved artifacts are intentionally incompatible with normal player checkpoints.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import ClipDataset
from .model import image_loss
from .train import load_checkpoint,seed_all


def infer(lam,x):
    features=lam.features(x)
    return F.normalize(features,dim=-1).flatten(-2)


@torch.no_grad()
def evaluate(lam,dataset):
    lam.eval();rows=[]
    for offset in range(0,len(dataset),8):
        x=torch.stack([dataset.get_clip(i,start=0) for i in range(offset,min(offset+8,len(dataset)))])
        actions=infer(lam,x)
        correct=lam.reconstruct(x[:,:-1],actions)
        wrong=lam.reconstruct(x[:,:-1],actions.roll(1,0))
        weights=.05+((x[:,1:]-x[:,:-1]).abs().mean(2,keepdim=True)>.03).float()
        denominator=weights.expand_as(correct).flatten(1).sum(1)
        error=lambda prediction:(((prediction-x[:,1:]).square()*weights).flatten(1).sum(1)/denominator)
        values=torch.stack((error(correct),error(wrong),error(x[:,:-1])),1)
        rows.extend(values.tolist())
    means=torch.tensor(rows).mean(0).tolist()
    return dict(clips=len(rows),correct_motion_mse=means[0],shuffled_motion_mse=means[1],copy_motion_mse=means[2])


def run(args):
    seed_all(args.seed);torch.set_num_threads(2)
    model,state=load_checkpoint(args.init)
    lam=model.lam;lam.books.requires_grad_(False)
    train=ClipDataset(args.data,'train',2);val=ClipDataset(args.data,'val',2)
    loader=DataLoader(train,batch_size=4,shuffle=True);iterator=iter(loader)
    optimizer=torch.optim.AdamW([p for p in lam.parameters() if p.requires_grad],lr=3e-4)
    before=evaluate(lam,val);started=time.monotonic();step=0
    for step in range(1,args.steps+1):
        if time.monotonic()-started>=args.max_seconds:
            step-=1;break
        try:x=next(iterator)
        except StopIteration:iterator=iter(loader);x=next(iterator)
        lam.train();optimizer.zero_grad(set_to_none=True)
        prediction=lam.reconstruct(x[:,:-1],infer(lam,x))
        loss=10*image_loss(prediction,x[:,1:],x[:,:-1])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(lam.parameters(),1.);optimizer.step()
        if step%500==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started)),flush=True)
    seconds=time.monotonic()-started
    report=dict(warning='Continuous target-information upper bound, NOT a discrete action model or playable checkpoint.',
                arguments=vars(args),steps=step,seconds=seconds,before=before,
                train=evaluate(lam,train),validation=evaluate(lam,val))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format='continuous-action-diagnostic-only',config=model.config_dict(),lam=lam.state_dict()),out/'diagnostic-only.pt')
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--init',required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--max-seconds',type=float,default=300);p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
