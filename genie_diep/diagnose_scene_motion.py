"""Unlabeled scene scrolling with a fixed player; synthetic capacity check only."""
import argparse
import json
import time
from pathlib import Path
import torch
from .model import Config,LatentActionModel
from .diagnose_actions import metrics,motion_loss


def scenes(seed,count):
    rng=torch.Generator().manual_seed(seed);pairs=[];labels=[]
    for _ in range(count):
        scene=torch.full((3,40,40),.65)
        for _ in range(12):
            x,y=torch.randint(4,34,(2,),generator=rng).tolist()
            scene[:,y:y+3,x:x+3]=torch.rand(3,1,1,generator=rng)*.8+.1
        for label,(dx,dy) in enumerate(((3,0),(-3,0),(0,3),(0,-3))):
            first=scene[:,4:36,4:36].clone()
            second=scene[:,4-dy:36-dy,4-dx:36-dx].clone()
            # Player remains fixed while the environment scrolls.
            for frame in (first,second):
                frame[:,14:19,14:19]=torch.tensor([.05,.7,.9])[:,None,None]
                frame[:,15:18,19:22]=.45
            pairs.append(torch.stack((first,second)));labels.append(label)
    return torch.stack(pairs),torch.tensor(labels)


def run(args):
    torch.manual_seed(42);torch.set_num_threads(1)
    train,labels=scenes(10,12);val,val_labels=scenes(91,8)
    model=LatentActionModel(Config(size=32,width=16,action_dim=8,action_codes=(4,),
        lam_variant='delta',lam_dilations=(1,2,3),lam_motion_warp=True))
    model.books[0].initialize(model.features(train)[:,:,0])
    opt=torch.optim.AdamW(model.parameters(),lr=.001);started=time.monotonic();step=0
    for step in range(1,args.steps+1):
        if time.monotonic()-started>args.max_seconds:step-=1;break
        indices=torch.randperm(len(train))[:16];x=train[indices]
        opt.zero_grad();pred,_,_,vq=model(x)
        loss=motion_loss(pred,x[:,1:],x[:,:-1])+vq
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step in (200,400):
            with torch.no_grad():changed=model.books[0].revive_unused(model.features(train)[:,:,0])
            for key in ('exp_avg','exp_avg_sq'):
                if key in opt.state[model.books[0].embedding.weight]:opt.state[model.books[0].embedding.weight][key][changed]=0
    with torch.no_grad():
        heldout=metrics(model,val,val_labels);actions,_,_=model.infer(val)
        shifts=model.motion_shifts(actions,32,32)[:,0]
        inferred=[shifts[val_labels==i].mean(0).tolist() for i in range(4)]
    report=dict(steps=step,seconds=time.monotonic()-started,arguments=vars(args),
        train=metrics(model,train,labels),validation=heldout,
        expected_scene_shifts=[[3,0],[-3,0],[0,3],[0,-3]],mean_inferred_shifts=inferred,
        passed=heldout['codes_used']==4 and heldout['wrong_loss']>heldout['true_loss']*1.5
               and heldout['true_loss']<heldout['copy_loss']*.5,
        warning='Synthetic scene motion only; no evidence of real-video or independent aim/fire control.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    torch.save(dict(config=vars(model.c),lam=model.state_dict()),out/'diagnostic-only.pt')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=800)
    p.add_argument('--max-seconds',type=float,default=90);p.add_argument('--out',default='runs/scene-motion-capacity')
    run(p.parse_args())
