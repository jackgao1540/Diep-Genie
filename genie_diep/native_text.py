"""Conservative native-name triage using font appearance and a nearby body.

Not semantic proof: edited labels may imitate game fonts. Source review and
full-frame audits remain required. Uncertain detections stay rejected.
"""
import cv2
import numpy as np


def native_name(frame,box):
    points=np.asarray(box);height,width=frame.shape[:2];scale=height/360
    x0,y0=np.floor(points.min(0)).astype(int);x1,y1=np.ceil(points.max(0)).astype(int)
    bw,bh=x1-x0,y1-y0
    if not (5*scale<=bh<=24*scale and 6*scale<=bw<=120*scale):return False
    if x0<0 or y0<0 or x1>width or y1>height:return False
    patch=frame[y0:y1,x0:x1].astype(np.int16)
    low=patch.min(2);high=patch.max(2)
    white=((low>210)&(high-low<32)).astype(np.uint8)
    dark=((high<155)&(high-low<45)).astype(np.uint8)
    outline=(cv2.dilate(white,np.ones((3,3),np.uint8))&dark).mean()
    colored=(high-low>65).mean()
    if not (.015<=white.mean()<=.50 and outline>=.01 and colored<.20):return False
    n,_,glyphs,_=cv2.connectedComponentsWithStats(white,8)
    heights=[glyphs[i,cv2.CC_STAT_HEIGHT] for i in range(1,n) if glyphs[i,cv2.CC_STAT_AREA]>=2*scale*scale]
    if not heights or np.median(heights)>10*scale or max(heights)>16*scale:return False
    # Player names sit just above a tank body. HUD labels and isolated captions
    # do not qualify merely because their text is white with a dark outline.
    left=max(0,x0-round(25*scale));right=min(width,x1+round(25*scale))
    top=max(0,y1-round(2*scale));bottom=min(height,y1+round(50*scale))
    roi=frame[top:bottom,left:right]
    if not roi.size:return False
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]>80)&(hsv[:,:,2]>90)).astype(np.uint8)
    count,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    for i in range(1,count):
        _,_,w,h,area=stats[i];cx,cy=centers[i]+[left,top]
        if min(w,h)<8*scale or area<40*scale*scale:continue
        if min(w,h)/max(w,h)<.45 or area/(w*h)<.45:continue
        if abs(cx-(x0+x1)/2)>max(14*scale,bw*.28):continue
        if 3*scale<=cy-y1<=35*scale:return True
    return False
