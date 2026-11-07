import math
import cv2
import numpy as np
import pytest
from genie_diep.visual_probes import barrel_probe,background_shift


def tank(angle):
    frame=np.full((96,96,3),210,dtype=np.uint8)
    center=np.array([48.,48.]);direction=np.array([math.cos(angle),math.sin(angle)])
    side=np.array([-direction[1],direction[0]])*3
    vertices=np.array([center+direction*3+side,center+direction*14+side,
                       center+direction*14-side,center+direction*3-side]).astype(np.int32)
    cv2.fillConvexPoly(frame,vertices,(130,130,130))
    cv2.circle(frame,(48,48),7,(25,170,220),-1)
    return frame


@pytest.mark.parametrize('angle',[0,math.pi/2,math.pi,-math.pi/2])
def test_barrel_probe_known_direction(angle):
    result=barrel_probe(tank(angle))
    assert result['angle'] is not None
    error=math.atan2(math.sin(result['angle']-angle),math.cos(result['angle']-angle))
    assert abs(error)<.2


def test_background_motion_ignores_stationary_centered_player():
    rng=np.random.default_rng(7)
    gray=rng.integers(90,230,(96,96),dtype=np.uint8)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    first=np.repeat(gray[:,:,None],3,-1)
    second=cv2.warpAffine(first,np.array([[1,0,2],[0,1,-1]],dtype=np.float32),(96,96),borderMode=cv2.BORDER_REFLECT)
    for frame in (first,second):cv2.circle(frame,(48,48),8,(20,170,220),-1)
    result=background_shift(first,second)
    assert result['dx']==pytest.approx(2,abs=.2)
    assert result['dy']==pytest.approx(-1,abs=.2)


def test_empty_scene_reports_unknown():
    frame=np.full((64,64,3),200,dtype=np.uint8)
    assert barrel_probe(frame)['angle'] is None
    assert background_shift(frame,frame)['dx'] is None
