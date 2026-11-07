"""Create timestamped source-frame contact sheets without playing or uploading media."""
import argparse
from pathlib import Path
import cv2
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser(); p.add_argument("video"); p.add_argument("output")
    p.add_argument("--times",type=float,nargs="+",default=[5,30,60,90,120,180,240,300])
    a=p.parse_args(); cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise ValueError(a.video)
    sheet=Image.new("RGB",(960,270*((len(a.times)+1)//2)),"white"); draw=ImageDraw.Draw(sheet)
    for i,t in enumerate(a.times):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
        if ok:
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); im.thumbnail((480,250))
            sheet.paste(im,((i%2)*480,(i//2)*270+20))
        draw.text(((i%2)*480+3,(i//2)*270+3),f"{Path(a.video).stem}: {t:g}s",fill="black")
    cap.release(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); sheet.save(a.output)


if __name__=="__main__":main()
