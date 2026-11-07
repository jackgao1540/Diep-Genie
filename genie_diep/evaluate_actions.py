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
from PIL import Image,ImageDraw
from .data import ClipDataset
from .train import load_checkpoint


@torch.no_grad()
def run(checkpoint,data,out):
    torch.set_num_threads(2)
    model,state=load_checkpoint(checkpoint);model.eval()
    if state['stage'] not in ('actions','world'):raise ValueError('Requires a trained action stage')
    if len(model.lam.books)!=1:raise ValueError('This diagnostic currently requires one action codebook')
    k=len(model.lam.books[0].embedding.weight)
    dataset=ClipDataset(data,'val',16);counts=np.zeros(k,dtype=int);rows=[];visuals=[]
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
            regions={};height,width=x.shape[-2:]
            center=torch.zeros_like(weight)
            center[:,:,:,height*3//8:height*5//8,width*3//8:width*5//8]=1
            for name,region in (('center',center),('surroundings',1-center)):
                rw=weight*region
                re=((prediction-target).square()*rw).flatten(1).sum(1)/rw.expand_as(target).sum().clamp_min(1e-8)
                regions[name+'_correct']=float(re[chosen])
                regions[name+'_other']=float(re[torch.arange(k)!=chosen].mean())
            rows.append(dict(clip=i,frame=t,code=chosen,correct=float(error[chosen]),
                             other_mean=float(other.mean()),best=float(error.min()),copy=copy_error,
                             chosen_rank=1+int((error<error[chosen]-1e-8).sum()),
                             effect=float((prediction-prediction[chosen:chosen+1]).square().mean()),
                             motion_fraction=float(motion.float().mean()),**regions))
            if t==7 and i in {len(dataset)//4,len(dataset)//2,len(dataset)*3//4}:
                best=int(error.argmin());worst=int(error.argmax())
                visuals.append((i,chosen,best,worst,torch.cat((current,target,prediction[chosen:chosen+1],
                                prediction[best:best+1],prediction[worst:worst+1]),0)[:,0]))
    freq=counts/counts.sum();nonzero=freq[freq>0]
    mean=lambda key:float(np.mean([r[key] for r in rows]))
    report=dict(checkpoint=str(checkpoint),stage=state['stage'],transitions=len(rows),
                code_counts=counts.tolist(),code_perplexity=float(np.exp(-(nonzero*np.log(nonzero)).sum())),
                correct_motion_mse=mean('correct'),other_motion_mse=mean('other_mean'),
                best_motion_mse=mean('best'),copy_motion_mse=mean('copy'),
                action_effect_mse=mean('effect'),mean_chosen_rank=mean('chosen_rank'),
                warning='Target-inferred action interventions. Predictive sensitivity is not evidence of semantic controllability; observations repeat within source videos.',
                transitions_detail=rows)
    report['regions']={name:dict(correct_motion_mse=mean(name+'_correct'),
                                other_motion_mse=mean(name+'_other')) for name in ('center','surroundings')}
    if model.lam.motion_head is not None:
        report['scene_shift_pixels_by_code']=model.lam.motion_shifts(actions,height,width)[:,0].tolist()
        report['scene_shift_warning']='Predicted image displacement, not physical movement labels; the centered player need not follow the scene.'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if visuals:
        sheet=Image.new('RGB',(640,len(visuals)*158),'white');draw=ImageDraw.Draw(sheet)
        for row,(index,chosen,best,worst,frames) in enumerate(visuals):
            labels=('Current','Target',f'Inferred code {chosen}',f'Best of all {best}',f'Worst of all {worst}')
            for col,(frame,label) in enumerate(zip(frames,labels)):
                rgb=(frame.permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype('uint8')
                sheet.paste(Image.fromarray(rgb).resize((128,128)),(col*128,row*158+28))
                draw.text((col*128+2,row*158+2),f'Clip {index}: {label}',fill='black')
        sheet.save(out/'interventions.png')
    (out/'interventions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='transitions_detail'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True);run(**vars(parser.parse_args()))
