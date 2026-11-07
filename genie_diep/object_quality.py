"""Color-component reconstruction diagnostics, NOT semantic object ground truth.

Each source component receives equal weight, including missing components. Hue
groups can mix tanks, food, projectiles and overlays. Inspect the paired crops;
corner counts are unreliable on tiny objects and are reported only for >=20px.
The fixed thresholds are shared by source/reconstruction, never fitted on val.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from .data import ClipDataset
from .train import load_checkpoint


def components(rgb, minimum=4):
    hsv=cv2.cvtColor(np.asarray(rgb,dtype=np.uint8),cv2.COLOR_RGB2HSV)
    hue,sat,value=hsv.transpose(2,0,1)
    foreground=(sat>=65)&(value>=65)
    groups={'yellow':(hue>=18)&(hue<=40),'red':(hue<=17)|(hue>=170),
            'cyan':(hue>=75)&(hue<105),'blue':(hue>=105)&(hue<140)}
    found=[]
    for color,selection in groups.items():
        n,labels,stats,centers=cv2.connectedComponentsWithStats((foreground&selection).astype('uint8'),8)
        for i in range(1,n):
            area=int(stats[i,cv2.CC_STAT_AREA])
            if area<minimum:continue
            x,y,w,h=map(int,stats[i,:4]);mask=labels==i
            contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=max(contours,key=cv2.contourArea)
            corners=None
            if area>=20:
                corners=len(cv2.approxPolyDP(contour,.04*cv2.arcLength(contour,True),True))
            found.append(dict(color=color,mask=mask,area=area,center=centers[i],box=(x,y,w,h),corners=corners))
    return found


def boundary_f1(a,b,tolerance=1):
    kernel=np.ones((3,3),np.uint8)
    aa=a.astype('uint8');bb=b.astype('uint8')
    ea=aa-cv2.erode(aa,kernel);eb=bb-cv2.erode(bb,kernel)
    if not ea.any() or not eb.any():return 0.
    near_a=cv2.dilate(ea,kernel,iterations=tolerance) if tolerance else ea
    near_b=cv2.dilate(eb,kernel,iterations=tolerance) if tolerance else eb
    recall=float((ea*near_b).sum()/ea.sum());precision=float((eb*near_a).sum()/eb.sum())
    return 2*recall*precision/max(recall+precision,1e-8)


def compare(source,prediction):
    targets=components(source);preds=components(prediction)
    # Greedy one-to-one matching, prioritize overlap then nearby same-color blobs.
    # Missing blobs score zero; unmatched predictions are reported separately.
    candidates=[]
    for i,a in enumerate(targets):
        for j,b in enumerate(preds):
            if a['color']!=b['color']:continue
            distance=float(np.linalg.norm(a['center']-b['center']))
            intersection=int((a['mask']&b['mask']).sum())
            iou=intersection/max(int((a['mask']|b['mask']).sum()),1)
            if intersection or distance<=max(3.,np.sqrt(a['area'])):
                candidates.append((iou,-distance,i,j))
    matches={};taken=set()
    for _,_,i,j in sorted(candidates,reverse=True):
        if i not in matches and j not in taken:matches[i]=j;taken.add(j)
    rows=[]
    for i,a in enumerate(targets):
        b=preds[matches[i]] if i in matches else None
        row=dict(color=a['color'],source_area=a['area'],box=a['box'],matched=b is not None,
                 iou=0.,boundary_f1=0.,area_ratio=0.,center_error=None,
                 source_corners=a['corners'],predicted_corners=None)
        if b is not None:
            row.update(iou=float((a['mask']&b['mask']).sum()/max((a['mask']|b['mask']).sum(),1)),
                       boundary_f1=boundary_f1(a['mask'],b['mask']),area_ratio=b['area']/a['area'],
                       center_error=float(np.linalg.norm(a['center']-b['center'])),predicted_corners=b['corners'])
        rows.append(row)
    return rows,len(preds)-len(taken)


@torch.no_grad()
def run(checkpoint,data,out,device='cpu'):
    torch.set_num_threads(2)
    model,state=load_checkpoint(checkpoint,device);model.eval()
    dataset=ClipDataset(data,'val',model.config.context)
    rows=[];crops=[];unmatched=0;frames=0
    for i in range(len(dataset)):
        x=dataset.get_clip(i,start=0)[None].to(device)
        rec,_,_=model.tokenizer(x)
        for t in sorted(set((0,x.shape[1]-1))):
            src=(x[0,t].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            pred=(rec[0,t].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype('uint8')
            objects,extra=compare(src,pred);unmatched+=extra;frames+=1
            for obj in objects:
                obj.update(clip=i,frame=t);rows.append(obj)
                bx,by,bw,bh=obj['box'];margin=3
                box=(max(0,bx-margin),max(0,by-margin),min(src.shape[1],bx+bw+margin),min(src.shape[0],by+bh+margin))
                crops.append((obj,Image.fromarray(src).crop(box),Image.fromarray(pred).crop(box)))
    def summarize(items):
        return dict(objects=len(items),matched=sum(r['matched'] for r in items),
                    mean_iou=float(np.mean([r['iou'] for r in items])) if items else None,
                    mean_boundary_f1=float(np.mean([r['boundary_f1'] for r in items])) if items else None)
    report=dict(warning='Heuristic color components, not verified shape classes. Repeated frames are correlated. Missing objects score zero; small/cropped objects and overlays can confuse these probes.',
                checkpoint=str(checkpoint),stage=state['stage'],validation_clips=len(dataset),frames=frames,
                **summarize(rows),unmatched_prediction_components=unmatched,
                by_color={c:summarize([r for r in rows if r['color']==c]) for c in ('yellow','red','cyan','blue')},
                objects_detail=rows)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'object-quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Stratified worst cases: show all color groups, not just the largest tank.
    chosen=[]
    for color in ('yellow','red','cyan','blue'):
        chosen.extend(sorted([c for c in crops if c[0]['color']==color],key=lambda c:c[0]['iou'])[:6])
    if chosen:
        sheet=Image.new('RGB',(480,128*((len(chosen)+2)//3)),'white');draw=ImageDraw.Draw(sheet)
        for n,(obj,a,b) in enumerate(chosen):
            xx=(n%3)*160;yy=(n//3)*128
            draw.text((xx+2,yy+2),f"{obj['color']} IoU {obj['iou']:.2f}\nclip {obj['clip']} frame {obj['frame']}",fill='black')
            sheet.paste(a.resize((76,76),Image.Resampling.NEAREST),(xx+2,yy+38))
            sheet.paste(b.resize((76,76),Image.Resampling.NEAREST),(xx+82,yy+38))
            draw.text((xx+2,yy+114),'source       reconstructed',fill='black')
        sheet.save(out/'worst-objects.png')
    print(json.dumps({k:v for k,v in report.items() if k!='objects_detail'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    run(**vars(p.parse_args()))
