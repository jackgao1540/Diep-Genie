"""Locate a possible native class bar for visual review, never approval.

The locator can choose other yellow HUD elements. OCR was not reliable in the
local six-case pilot, so no automated class reader is provided.
"""
import cv2
import numpy as np


def class_bar_crop(frame):
    h,w=frame.shape[:2];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    yellow=((hsv[:,:,0]>18)&(hsv[:,:,0]<40)&(hsv[:,:,1]>70)&(hsv[:,:,2]>140)).astype(np.uint8)
    yellow[:int(h*.8)]=0
    # White class letters split the yellow bar into left/right components.
    yellow=cv2.morphologyEx(yellow,cv2.MORPH_CLOSE,np.ones((1,max(1,round(w*.08))),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(yellow,8)
    bars=[r for r in stats[1:] if r[2]>w*.18 and r[3]<h*.07 and abs(r[0]+r[2]/2-w/2)<w*.08]
    if not bars:return None
    x,y,bw,bh,_=max(bars,key=lambda r:r[4]);cx=x+bw/2
    return frame[max(0,y-3):min(h,y+bh+3),max(0,int(cx-bw*.32)):min(w,int(cx+bw*.32))]
