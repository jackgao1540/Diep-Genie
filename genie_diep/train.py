"""Resumable, bounded training and honest local diagnostic metrics."""
import argparse
import json
import random
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler

from .model import Config, WorldModel, image_loss, masked_dynamics_loss
from .data import ClipDataset
from .synthetic import batch as synthetic_batch
from .action_pool import collect_features,update_books


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def save_checkpoint(path,model,optimizer,step,stage,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(config=model.config_dict(),model=model.state_dict(),optimizer=optimizer.state_dict(),
               step=step,stage=stage,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],extra=extra or {})
    temp=path.with_suffix(".tmp")
    torch.save(state,temp); temp.replace(path)


def load_checkpoint(path,device="cpu"):
    state=torch.load(path,map_location=device,weights_only=True)
    model=WorldModel(Config(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model,state


@torch.no_grad()
def initialize(model,x):
    model.tokenizer.vq.initialize(model.tokenizer.features(x))
    features=model.lam.features(x)
    for i,book in enumerate(model.lam.books):
        book.initialize(features[:,:,i])


@torch.no_grad()
def evaluate(model,x):
    model.eval()
    rec,ids,_=model.tokenizer(x)
    lam,actions,action_ids,_=model.lam(x)
    shuffled=actions.roll(1,0)
    wrong=model.lam.reconstruct(x[:,:-1],shuffled)
    corrupted=ids.clone(); corrupted[:,-1]=model.config.vocab
    logits=model.dynamics(corrupted,actions)[:,-1]
    shuffled_logits=model.dynamics(corrupted,shuffled)[:,-1]
    target=ids[:,-1]
    ce=lambda y:torch.nn.functional.cross_entropy(y.reshape(-1,model.config.vocab),target.flatten()).item()
    prediction=model.tokenizer.decode_ids(logits.argmax(-1)[:,None])[:,0]
    foreground=(x.max(2,keepdim=True).values-x.min(2,keepdim=True).values>.12).float()
    foreground_mse=((rec-x).square()*foreground).sum()/(foreground.sum()*3).clamp_min(1)
    side=x.shape[-1]; lo=side*3//8; hi=side*5//8
    return dict(tokenizer_mse=(rec-x).square().mean().item(),
                tokenizer_foreground_mse=foreground_mse.item(),
                tokenizer_center_mse=(rec[:,:,:,lo:hi,lo:hi]-x[:,:,:,lo:hi,lo:hi]).square().mean().item(),
                copy_mse=(x[:,1:]-x[:,:-1]).square().mean().item(),
                lam_mse=(lam-x[:,1:]).square().mean().item(),
                lam_shuffled_mse=(wrong-x[:,1:]).square().mean().item(),
                dynamics_ce=ce(logits),dynamics_shuffled_ce=ce(shuffled_logits),
                next_frame_mse=(prediction-x[:,-1]).square().mean().item(),
                token_codes_used=int(ids.unique().numel()),
                action_codes_used=[int(action_ids[...,i].unique().numel()) for i in range(action_ids.shape[-1])])


def train(args):
    seed_all(args.seed); torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(file.name.encode()); source_hash.update(file.read_bytes())
    (out/"run.json").write_text(json.dumps(dict(arguments=vars(args),torch_version=torch.__version__,
                                             source_sha256=source_hash.hexdigest()),indent=2),encoding="utf-8")
    if args.resume:
        model,state=load_checkpoint(args.resume,device)
        if state["stage"] != args.stage:
            raise ValueError("Resume must keep stage; use --init for a new stage")
    elif args.init:
        model,_=load_checkpoint(args.init,device); state=None
    else:
        model=WorldModel(Config(size=args.size,width=args.width,context=args.length,
                                action_codes=tuple(args.action_codes),lam_variant=args.lam_variant,
                                downsample=args.downsample)).to(device); state=None
    if args.lam_dilations:
        requested=tuple(args.lam_dilations)
        if len(requested)!=3 or any(v<1 for v in requested):raise ValueError('Invalid decoder dilations')
        if args.resume and requested!=tuple(model.config.lam_dilations):
            raise ValueError('Resume cannot change architecture; use --init')
        model.config.lam_dilations=requested
        for layer,dilation in zip((model.lam.decoder[0],model.lam.decoder[2],model.lam.decoder[4]),requested):
            layer.dilation=(dilation,dilation);layer.padding=(dilation,dilation)
    if args.motion_warp and not model.config.lam_motion_warp:
        if args.resume:raise ValueError('Resume cannot enable a new architecture; use --init')
        model.lam.enable_motion_warp()
    if args.stage in ("world","actions"):
        model.tokenizer.requires_grad_(False)
    if args.stage=="actions":
        model.dynamics.requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr)
    start=0
    if state:
        optimizer.load_state_dict(state["optimizer"]); start=state["step"]
        torch.set_rng_state(state["torch_rng"].cpu()); random.setstate(state["python_rng"])
        if device.type=="cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    c=model.config
    if args.synthetic:
        x=synthetic_batch(args.batch,c.context,c.size,args.seed).to(device)
        val=synthetic_batch(args.batch,c.context,c.size,args.seed+1000).to(device)
        loader=None
    else:
        length=min(args.length,c.context)
        data=ClipDataset(args.data,"train",length)
        validation=ClipDataset(args.data,"val",length)
        sampler=WeightedRandomSampler(data.source_balanced_weights(),len(data),replacement=True) if args.balance_sources else None
        loader=DataLoader(data,batch_size=args.batch,shuffle=sampler is None,sampler=sampler,num_workers=0,drop_last=False)
        iterator=iter(loader); x=next(iterator).to(device)
        val=torch.stack([validation[i] for i in range(min(args.batch,len(validation)))]).to(device)
    if not (args.init or args.resume):
        initialize(model,x)
    if args.stage in ("world","actions") and not args.resume and args.init_actions:
        if args.synthetic:
            features=model.lam.features(x).detach().flatten(0,1); pool_indices=[]
        else:
            features,pool_indices=collect_features(model.lam,data,device,args.action_pool_clips,args.batch)
        book_report=update_books(model.lam,features,optimizer,initialize=True)
        print(json.dumps(dict(action_initialization=book_report,pool_clips=len(pool_indices))),flush=True)
    before=evaluate(model,x)
    t0=time.monotonic(); step=start; losses=[]
    for step in range(start+1,args.steps+1):
        if time.monotonic()-t0 >= args.max_seconds:
            step-=1; break
        if loader:
            try: x=next(iterator).to(device)
            except StopIteration: iterator=iter(loader); x=next(iterator).to(device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        if args.stage=="tokenizer":
            rec,_,vq=model.tokenizer(x)
            loss=10*image_loss(rec,x,color_balance=args.color_balance)+vq
        else:
            prediction,actions,_,vq=model.lam(x)
            loss=10*image_loss(prediction,x[:,1:],x[:,:-1])+vq
            if args.stage=="world":
                with torch.no_grad(): _,ids,_=model.tokenizer.encode(x)
                loss=loss+masked_dynamics_loss(model.dynamics,ids,actions)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss: aborting before wasting compute")
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if (args.stage in ("world","actions") and args.revive_every and step%args.revive_every==0
                and (not args.revive_until_step or step<=args.revive_until_step)):
            if args.synthetic:
                features=model.lam.features(x).detach().flatten(0,1); pool_indices=[]
            else:
                features,pool_indices=collect_features(model.lam,data,device,args.action_pool_clips,args.batch)
            book_report=update_books(model.lam,features,optimizer)
            print(json.dumps(dict(step=step,action_revival=book_report,pool_clips=len(pool_indices))),flush=True)
        if step%args.log_every==0 or step==args.steps:
            print(json.dumps(dict(step=step,stage=args.stage,loss=losses[-1],seconds=round(time.monotonic()-t0,2))),flush=True)
            save_checkpoint(out/"last.pt",model,optimizer,step,args.stage,dict(synthetic=args.synthetic))
    save_checkpoint(out/"last.pt",model,optimizer,step,args.stage,dict(synthetic=args.synthetic))
    report=dict(stage=args.stage,steps=step,seconds=time.monotonic()-t0,synthetic=args.synthetic,
                parameters=sum(p.numel() for p in model.parameters()),train_before=before,
                train_after=evaluate(model,x),validation=evaluate(model,val),
                loss_first=losses[0] if losses else None,loss_last=losses[-1] if losses else None,
                warning="Synthetic diagnostics are not evidence of real-video generalization or playability" if args.synthetic else "Inspect action interventions and held-out rollouts before claiming playability")
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["tokenizer","actions","world"],required=True)
    p.add_argument("--data",default="data/clips"); p.add_argument("--out",required=True)
    p.add_argument("--synthetic",action="store_true"); p.add_argument("--init"); p.add_argument("--resume")
    p.add_argument("--steps",type=int,default=200); p.add_argument("--max-seconds",type=float,default=600)
    p.add_argument("--batch",type=int,default=4); p.add_argument("--size",type=int,default=32)
    p.add_argument("--width",type=int,default=32); p.add_argument("--length",type=int,default=5)
    p.add_argument("--action-codes",type=int,nargs="+",default=[16])
    p.add_argument("--lam-variant",choices=["pair","delta"],default="delta")
    p.add_argument("--lam-dilations",type=int,nargs=3,help="Experimental decoder receptive field; 1 3 6 gives a 21-pixel span")
    p.add_argument("--motion-warp",action="store_true",help="Experimental action-conditioned scene translation before residual reconstruction")
    p.add_argument("--downsample",type=int,choices=[4,8],default=4)
    p.add_argument("--init-actions",action="store_true")
    p.add_argument("--revive-every",type=int,default=0)
    p.add_argument("--revive-until-step",type=int,default=0,help="Stop code resets after this step; 0 keeps prior unlimited behavior")
    p.add_argument("--color-balance",action="store_true",help="Tokenizer-only equal-color-group reconstruction term")
    p.add_argument("--balance-sources",action="store_true",help="Equal total training sampling mass per source video")
    p.add_argument("--action-pool-clips",type=int,default=64)
    p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=4); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--log-every",type=int,default=50)
    args=p.parse_args()
    if args.stage=="world" and not (args.init or args.resume):
        p.error("World training requires a pretrained tokenizer checkpoint via --init or --resume")
    train(args)


if __name__=="__main__": main()
