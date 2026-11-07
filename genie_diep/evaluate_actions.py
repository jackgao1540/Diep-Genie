"""Held-out interventions against EVERY alternative single-codebook action.

Unlike shuffled minibatches, alternatives cannot accidentally retain the same
code. Uses motion-region errors and code frequency, not just whole-frame MSE.
These checks measure predictive code use, not WASD/aim/fire semantics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import ClipDataset
from .train import load_checkpoint


@torch.no_grad()
def run(checkpoint,data,out):
    torch.set_num_threads(2)
    model,state=load_checkpoint(checkpoint);model.eval()
    if state['stage'] not in ('actions','world'):raise ValueError('Requires a trained action stage')
    if len(model.lam.books)!=1:raise ValueError('This diagnostic currently requires one action codebook')
    k=len(model.lam.books[0].embedding.weight)
    dataset=ClipDataset(data,'val',16);counts=np.zeros(k,dtype=int);rows=[]
    for i in range(len(dataset)):
        clip=dataset.get_clip(i,start=0)
        for t in (0,7,14):
            x=clip[t:t+2][None]
            _,ids,_=model.lam.infer(x);chosen=int(ids[0,0,0]);counts[chosen]+=1
            alternatives=torch.arange(k).reshape(k,1,1)
            actions=model.lam.lookup(alternatives)
            prediction=model.lam.reconstruct(x[:,:1].expand(k,-1,-1,-1,-1),actions)
            target=x[:,1:];current=x[:,:1]
            motion=(target-current).abs().mean(2,keepdim=True)>.03
            weight=.05+motion.float()
            error=((prediction-target).square()*weight).flatten(1).sum(1)/weight.expand_as(target).sum()
            copy_error=(((current-target).square()*weight).sum()/weight.expand_as(target).sum()).item()
            other=error[torch.arange(k)!=chosen]
            rows.append(dict(clip=i,frame=t,code=chosen,correct=float(error[chosen]),
                             other_mean=float(other.mean()),best=float(error.min()),copy=copy_error,
                             chosen_rank=1+int((error<error[chosen]-1e-8).sum()),
                             effect=float((prediction-prediction[chosen:chosen+1]).square().mean()),
                             motion_fraction=float(motion.float().mean())))
    freq=counts/counts.sum();nonzero=freq[freq>0]
    mean=lambda key:float(np.mean([r[key] for r in rows]))
    report=dict(checkpoint=str(checkpoint),stage=state['stage'],transitions=len(rows),
                code_counts=counts.tolist(),code_perplexity=float(np.exp(-(nonzero*np.log(nonzero)).sum())),
                correct_motion_mse=mean('correct'),other_motion_mse=mean('other_mean'),
                best_motion_mse=mean('best'),copy_motion_mse=mean('copy'),
                action_effect_mse=mean('effect'),mean_chosen_rank=mean('chosen_rank'),
                warning='Target-inferred action interventions. Predictive sensitivity is not evidence of semantic controllability; observations repeat within source videos.',
                transitions_detail=rows)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'interventions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='transitions_detail'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True);run(**vars(parser.parse_args()))
