import numpy as np
import cv2
from genie_diep.object_quality import compare


def scene():
    a=np.full((64,64,3),205,dtype=np.uint8)
    cv2.rectangle(a,(10,10),(20,20),(255,220,80),-1)
    cv2.fillPoly(a,[np.array([[40,8],[32,23],[48,23]])],(240,90,90))
    return a


def test_identity_and_missing_object():
    a=scene();rows,extra=compare(a,a)
    assert len(rows)==2 and extra==0 and all(r['iou']==1 for r in rows)
    assert all(r['boundary_f1']==1 for r in rows)
    blank=np.full_like(a,205);missing,_=compare(a,blank)
    assert len(missing)==2 and all(not r['matched'] and r['iou']==0 for r in missing)


def test_color_preserved_but_shape_wrong_is_penalized():
    a=scene();b=a.copy();b[7:25,30:51]=205
    cv2.circle(b,(40,17),8,(240,90,90),-1)
    rows,_=compare(a,b);red=next(r for r in rows if r['color']=='red')
    assert red['matched'] and red['iou']<.85 and red['boundary_f1']<.95


def test_extra_object_is_reported():
    a=scene();b=a.copy();cv2.rectangle(b,(40,40),(48,48),(255,220,80),-1)
    rows,extra=compare(a,b)
    assert extra==1 and all(r['iou']==1 for r in rows)
