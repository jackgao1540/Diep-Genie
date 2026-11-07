"""Whole-validation-set metrics and source/reconstruction/rollout image strips."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import DataLoader
from .data import ClipDataset
from .train import load_checkpoint,evaluate


@torch.no_grad()
def run(checkpoint,data,out,device="cpu"):
    torch.set_num_threads(4)
    model,state=load_checkpoint(checkpoint,device); model.eval()
    dataset=ClipDataset(data,"val",model.config.context)
    sums={}; count=0; used=[set() for _ in model.lam.books]; token_used=set()
    for x in DataLoader(dataset,batch_size=8,shuffle=False):
        x=x.to(device); metrics=evaluate(model,x)
        for key,value in metrics.items():
            if key not in ("token_codes_used","action_codes_used"):
                sums[key]=sums.get(key,0)+value*len(x)
        _,ids,_=model.tokenizer.encode(x); _,aids,_=model.lam.infer(x)
        token_used.update(ids.unique().tolist())
        for i,s in enumerate(used): s.update(aids[...,i].unique().tolist())
        count+=len(x)
    report={key:value/count for key,value in sums.items()}
    if state['stage']!='world':
        for key in ('dynamics_ce','dynamics_shuffled_ce','next_frame_mse'):report.pop(key,None)
    if state['stage']=='tokenizer':
        for key in ('lam_mse','lam_shuffled_mse'):report.pop(key,None)
    report.update(validation_clips=count,token_codes_used=len(token_used),action_codes_used=list(map(len,used)),
                  stage=state["stage"],checkpoint_step=state["step"],
                  warning="Inferred-action teacher-forced errors are diagnostics, not evidence of playable control")
    if state['stage']=='tokenizer':report.pop('action_codes_used',None)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    x=dataset[len(dataset)//2][None].to(device)
    rec,ids,_=model.tokenizer(x); actions,aids,_=model.lam.infer(x)
    frames=[x,rec];labels=['Source held-out clip','Tokenizer reconstruction']
    if state['stage']=='world':
        history=ids[:,:1]; generated=[rec[:,:1]]
        for t in range(1,x.shape[1]):
            nxt=model.dynamics.next_tokens(history,actions[:,:t])
            generated.append(model.tokenizer.decode_ids(nxt)); history=torch.cat([history,nxt],1)
        frames.append(torch.cat(generated,1));labels.append('Rollout: inferred reference actions')
    elif state['stage']=='actions':
        frames.append(torch.cat([x[:,:1],model.lam.reconstruct(x[:,:-1],actions)],1))
        labels.append('Action decoder: target-inferred actions')
    tile=128; sheet=Image.new("RGB",(tile*x.shape[1],(tile+24)*len(frames)),"white"); draw=ImageDraw.Draw(sheet)
    for r,(array,label) in enumerate(zip(frames,labels)):
        draw.text((3,r*(tile+24)+3),label,fill="black")
        for t in range(x.shape[1]):
            rgb=(array[0,t].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            sheet.paste(Image.fromarray(rgb).resize((tile,tile)),(t*tile,r*(tile+24)+24))
    sheet.save(out/"comparison.png")
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--data",required=True)
    p.add_argument("--out",required=True); p.add_argument("--device",default="cpu")
    run(**vars(p.parse_args()))


if __name__=="__main__":main()
