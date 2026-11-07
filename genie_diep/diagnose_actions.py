"""Unlabeled LAM training on ambiguous four-way motion; labels used only to score.

Every starting frame has four possible futures, so copying or memorizing the
starting image cannot solve the task. Validation uses unseen square positions.
"""
import argparse
import json
from pathlib import Path
import torch
from .model import Config, LatentActionModel


def examples(positions,size=32):
    data=[]; labels=[]
    for x,y in positions:
        for label,(dx,dy) in enumerate([(3,0),(-3,0),(0,3),(0,-3)]):
            first=torch.full((3,size,size),.15)
            second=first.clone(); first[:,y:y+6,x:x+6]=.85
            second[:,y+dy:y+dy+6,x+dx:x+dx+6]=.85
            data.append(torch.stack([first,second])); labels.append(label)
    return torch.stack(data),torch.tensor(labels)


def motion_loss(pred,target,previous):
    motion=(target-previous).abs().mean(2,keepdim=True)>.03
    weight=.05+motion.float()
    return ((pred-target).square()*weight).sum()/weight.expand_as(target).sum()


@torch.no_grad()
def metrics(model,x,labels):
    pred,actions,ids,_=model(x)
    # Shift within each group of four: always another direction for the same image.
    wrong=actions.reshape(-1,4,*actions.shape[1:]).roll(1,1).reshape_as(actions)
    other=model.reconstruct(x[:,:-1],wrong)
    used=int(ids.unique().numel()); table=[]
    for i in range(len(model.books[0].embedding.weight)):
        counts=torch.bincount(labels[ids[:,0,0]==i],minlength=4)
        table.append(counts.tolist())
    return dict(codes_used=used,true_loss=motion_loss(pred,x[:,1:],x[:,:-1]).item(),
                wrong_loss=motion_loss(other,x[:,1:],x[:,:-1]).item(),
                copy_loss=motion_loss(x[:,:-1],x[:,1:],x[:,:-1]).item(),
                code_direction_counts=table)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--out",default="runs/action-oracle"); a=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    x,labels=examples([(7,7),(15,7),(7,15),(15,15)])
    val,val_labels=examples([(11,11),(6,12),(13,6)])
    model=LatentActionModel(Config(size=32,width=32,action_codes=(4,),lam_variant="delta"))
    model.books[0].initialize(model.features(x)[:,:,0])
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    before=metrics(model,x,labels)
    for step in range(a.steps):
        opt.zero_grad(); pred,_,_,vq=model(x)
        loss=motion_loss(pred,x[:,1:],x[:,:-1])+vq
        loss.backward(); opt.step()
        if (step+1)%250==0:
            with torch.no_grad():
                replaced=model.books[0].revive_unused(model.features(x)[:,:,0])
                state=opt.state[model.books[0].embedding.weight]
                for key in ("exp_avg","exp_avg_sq"):
                    if key in state: state[key][replaced]=0
            print(json.dumps(dict(step=step+1,loss=loss.item(),revived=replaced)),flush=True)
    train=metrics(model,x,labels); heldout=metrics(model,val,val_labels)
    passed=heldout["codes_used"]==4 and heldout["wrong_loss"]>heldout["true_loss"]*1.5 and heldout["true_loss"]<heldout["copy_loss"]*.5
    report=dict(before=before,train=train,validation=heldout,passed=passed,
                warning="Four-way synthetic motion only; this does not prove disentangled gameplay actions")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    torch.save(model.state_dict(),out/"lam.pt")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
