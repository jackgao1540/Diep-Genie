"""Capacity diagnostic on one training clip; never interpreted as generalization."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from .data import ClipDataset
from .model import image_loss
from .train import load_checkpoint,save_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("--data",required=True)
    p.add_argument("--out",required=True);p.add_argument("--steps",type=int,default=750)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(5)
    model,_=load_checkpoint(a.checkpoint);model.train()
    x=ClipDataset(a.data,"train",model.config.context)[0][None]
    opt=torch.optim.AdamW(model.tokenizer.parameters(),lr=.001)
    with torch.no_grad():before=model.tokenizer(x)[0];before_loss=image_loss(before,x).item()
    started=time.monotonic()
    for step in range(a.steps):
        opt.zero_grad();rec,_,vq=model.tokenizer(x);loss=10*image_loss(rec,x)+vq
        loss.backward();opt.step()
    with torch.no_grad():after=model.tokenizer(x)[0];after_loss=image_loss(after,x).item()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    save_checkpoint(out/"overfit-only.pt",model,opt,a.steps,"tokenizer",dict(overfit_only=True))
    report=dict(before_loss=before_loss,after_loss=after_loss,seconds=time.monotonic()-started,
                steps=a.steps,warning="One training clip only. Capacity check, not a model for held-out gameplay.")
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    sheet=Image.new("RGB",(128*x.shape[1],3*152),"white");d=ImageDraw.Draw(sheet)
    for r,(arr,label) in enumerate([(x,"Training source"),(before,"Before overfit"),(after,"After overfit (capacity test only)")]):
        d.text((3,r*152+3),label,fill="black")
        for t in range(x.shape[1]):
            rgb=(arr[0,t].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
            sheet.paste(Image.fromarray(rgb).resize((128,128)),(t*128,r*152+24))
    sheet.save(out/"comparison.png");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
