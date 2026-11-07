"""Diagnostic visual measurements, never supplied as action labels during training.

Designed for centered cyan ordinary tanks. Confidence and missing measurements
matter: optical flow can follow enemies, barrel estimates can follow text/health
bars, and visible projectiles do not identify the fire key. These are probes, not
ground-truth controls or a substitute for learned dynamics.
"""
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def cyan_components(frame):
    r,g,b=frame.astype(np.int16).transpose(2,0,1)
    mask=((g-r>45)&(b-r>60)&(b>130)&(g>95)).astype(np.uint8)
    count,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    return [(centers[i],int(stats[i,cv2.CC_STAT_AREA])) for i in range(1,count)]


def tank_pose(frame):
    h,w=frame.shape[:2]; center=np.array([w/2,h/2])
    candidates=[(p,area) for p,area in cyan_components(frame)
                if 3<=area<h*w*.06 and np.linalg.norm(p-center)<min(h,w)*.16]
    if not candidates:return None
    p,area=min(candidates,key=lambda item:np.linalg.norm(item[0]-center))
    return dict(x=float(p[0]),y=float(p[1]),radius=float(np.sqrt(area/np.pi)),area=area)


def barrel_probe(frame):
    pose=tank_pose(frame)
    if pose is None:return dict(angle=None,confidence=0.,tank=None)
    y,x=np.indices(frame.shape[:2]);dx=x-pose['x'];dy=y-pose['y']
    radius=np.hypot(dx,dy);r=pose['radius']
    gray=frame.mean(-1);chroma=frame.max(-1).astype(float)-frame.min(-1)
    annulus=(radius>r*1.08)&(radius<r*2.6)&(chroma<30)&(gray<185)
    weights=np.clip(185-gray,0,130)*annulus
    if weights.sum()<50:return dict(angle=None,confidence=0.,tank=pose)
    angles=np.arctan2(dy,dx);bins=((angles+np.pi)*36/(2*np.pi)).astype(int)%36
    hist=np.bincount(bins.ravel(),weights=weights.ravel(),minlength=36)
    smooth=hist+np.roll(hist,1)+np.roll(hist,-1)
    peak=int(smooth.argmax());confidence=float(smooth[peak]/max(hist.sum(),1))
    sector=(bins==peak)|(bins==(peak-1)%36)|(bins==(peak+1)%36)
    selected=weights*sector
    angle=float(np.arctan2((selected*np.sin(angles)).sum(),(selected*np.cos(angles)).sum()))
    return dict(angle=angle if confidence>=.25 else None,confidence=confidence,tank=pose)


def background_shift(first,second):
    """Track background corners, require forward/backward consistency and consensus."""
    a=cv2.cvtColor(first,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(second,cv2.COLOR_RGB2GRAY)
    h,w=a.shape; y,x=np.indices(a.shape)
    mask=(((x-w/2)**2+(y-h/2)**2)>(min(h,w)*.23)**2).astype(np.uint8)*255
    mask[:3]=0;mask[-3:]=0;mask[:,:3]=0;mask[:,-3:]=0
    # Exclude likely moving blue/red tanks/projectiles; yellow food can be stationary scenery.
    rgb=first.astype(np.int16)
    moving=((rgb[:,:,2]-rgb[:,:,0]>40)&(rgb[:,:,1]-rgb[:,:,0]>40))|((rgb[:,:,0]-rgb[:,:,1]>35)&(rgb[:,:,0]-rgb[:,:,2]>15))
    mask[moving]=0
    points=cv2.goodFeaturesToTrack(a,80,.01,4,mask=mask,blockSize=3)
    empty=dict(dx=None,dy=None,confidence=0.,tracks=0,inliers=0)
    if points is None or len(points)<4:return empty
    next_points,status,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(15,15),maxLevel=2)
    if next_points is None:return empty
    back,back_status,_=cv2.calcOpticalFlowPyrLK(b,a,next_points,None,winSize=(15,15),maxLevel=2)
    if back is None:return empty
    valid=(status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=-1)<.75)
    delta=next_points[valid,0]-points[valid,0]
    if len(delta)<4:return empty
    median=np.median(delta,axis=0);error=np.linalg.norm(delta-median,axis=-1)
    inliers=error<1.;confidence=float(inliers.mean())
    if inliers.sum()<4 or confidence<.6:return {**empty,'tracks':len(delta),'confidence':confidence}
    shift=np.median(delta[inliers],axis=0)
    if np.linalg.norm(shift)>min(h,w)*.2:return empty
    return dict(dx=float(shift[0]),dy=float(shift[1]),confidence=confidence,tracks=len(delta),inliers=int(inliers.sum()))


def near_barrel_projectiles(frame,barrel):
    if barrel['angle'] is None or barrel['tank'] is None:return None
    tank=barrel['tank'];center=np.array([tank['x'],tank['y']]);count=0
    for p,area in cyan_components(frame):
        delta=p-center;distance=np.linalg.norm(delta)
        if 1<=area<tank['area']*.65 and tank['radius']*2<distance<tank['radius']*8:
            angle=math.atan2(delta[1],delta[0]);error=math.atan2(math.sin(angle-barrel['angle']),math.cos(angle-barrel['angle']))
            if abs(error)<math.pi/6:count+=1
    return count


def transition(first,second):
    background=background_shift(first,second);before=barrel_probe(first);after=barrel_probe(second)
    turn=None
    if before['angle'] is not None and after['angle'] is not None:
        turn=math.atan2(math.sin(after['angle']-before['angle']),math.cos(after['angle']-before['angle']))
    return dict(background=background,barrel_before=before,barrel_after=after,aim_delta=turn,
                near_barrel_projectiles=near_barrel_projectiles(second,after))


def inspect(root,out,limit=24):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for p in sorted(root.glob('*.jsonl')) for s in p.read_text().splitlines() if s]
    indices=np.linspace(0,len(rows)-1,min(limit,len(rows))).astype(int)
    results=[];sheet=Image.new('RGB',(800,160*math.ceil(len(indices)/4)),'white');draw=ImageDraw.Draw(sheet)
    for n,i in enumerate(indices):
        row=rows[i];clip=np.load(root/row['path'],allow_pickle=False);probe=transition(clip[0],clip[1])
        results.append(dict(path=row['path'],split=row['split'],**probe))
        image=Image.fromarray(clip[1]).resize((128,128)); painter=ImageDraw.Draw(image)
        barrel=probe['barrel_after'];tank=barrel['tank'];scale=128/clip.shape[1]
        if tank and barrel['angle'] is not None:
            cx,cy=tank['x']*scale,tank['y']*scale
            painter.line((cx,cy,cx+18*math.cos(barrel['angle']),cy+18*math.sin(barrel['angle'])),fill='lime',width=2)
        bx,by=(n%4)*200,(n//4)*160;sheet.paste(image,(bx,by+30))
        flow=probe['background'];text=f"{row['source_id']} {row['start']:.1f}s\n"
        text+=f"aim conf {barrel['confidence']:.2f}; "
        text+=f"flow {flow['dx']:.1f},{flow['dy']:.1f}" if flow['dx'] is not None else 'flow unknown'
        draw.text((bx+2,by+2),text,fill='black')
    sheet.save(out/'probes.png')
    (out/'probes.json').write_text(json.dumps(dict(warning='Heuristic visual probes, not ground-truth controls; never used as training labels.',samples=results),indent=2),encoding='utf-8')
    print(json.dumps(dict(samples=len(results),flow_available=sum(r['background']['dx'] is not None for r in results),aim_available=sum(r['aim_delta'] is not None for r in results))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--limit',type=int,default=24)
    inspect(**vars(p.parse_args()))
