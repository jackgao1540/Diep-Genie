"""Small camera-following tank scene for pipeline tests, not training evidence."""
import math
import numpy as np
from PIL import Image, ImageDraw
import torch


def batch(count=8, length=5, size=32, seed=0):
    rng=np.random.default_rng(seed)
    clips=[]
    for _ in range(count):
        vx,vy=rng.integers(-2,3,2)
        angle=float(rng.uniform(-math.pi,math.pi)); turn=float(rng.choice([-.15,0,.15]))
        fire=bool(rng.integers(2)); frames=[]
        for t in range(length):
            im=Image.new("RGB",(size,size),(205,205,205)); d=ImageDraw.Draw(im)
            for p in range(-size,size*2,8):
                d.line((p-vx*t,0,p-vx*t,size),fill=(185,185,185))
                d.line((0,p-vy*t,size,p-vy*t),fill=(185,185,185))
            cx=cy=size//2; a=angle+turn*t
            end=(cx+math.cos(a)*7,cy+math.sin(a)*7)
            d.line((cx,cy,*end),fill=(70,70,70),width=3)
            d.ellipse((cx-3,cy-3,cx+3,cy+3),fill=(30,160,225))
            if fire:
                bx=cx+math.cos(a)*(8+t*2); by=cy+math.sin(a)*(8+t*2)
                d.ellipse((bx-1,by-1,bx+1,by+1),fill=(30,160,225))
            frames.append(np.asarray(im))
        clips.append(np.stack(frames))
    return torch.tensor(np.stack(clips)).permute(0,1,4,2,3).float()/255
