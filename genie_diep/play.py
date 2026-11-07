"""Local experimental viewer for learned latent actions; no invented WASD mapping."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image,ImageTk
import numpy as np
import torch
from .train import load_checkpoint


def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--image")
    a=p.parse_args(); torch.set_num_threads(4)
    model,state=load_checkpoint(a.checkpoint); model.eval()
    if state["stage"]!="world": raise ValueError("Viewer requires a world-stage checkpoint")
    root=tk.Tk(); root.title("Genie research prototype — latent actions")
    frame=ttk.Frame(root,padding=12); frame.pack()
    ttk.Label(frame,text="Experimental model. Codes are not calibrated movement/aim/fire controls.").pack()
    canvas=tk.Label(frame); canvas.pack()
    status=tk.StringVar(value="Load a gameplay image to begin."); ttk.Label(frame,textvariable=status).pack()
    choices=[]
    for index,count in enumerate(model.config.action_codes):
        row=ttk.Frame(frame); row.pack(fill="x")
        ttk.Label(row,text=f"Latent factor {index+1}").pack(side="left")
        value=tk.IntVar(value=0); choices.append(value)
        ttk.Spinbox(row,from_=0,to=count-1,textvariable=value,width=6).pack(side="left")
    state_ui={}

    def show(x):
        rgb=(x[0,-1].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        picture=ImageTk.PhotoImage(Image.fromarray(rgb).resize((512,512)))
        canvas.configure(image=picture); canvas.image=picture

    @torch.no_grad()
    def load(path=None):
        path=path or filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.webp")])
        if not path:return
        image=Image.open(path).convert("RGB"); side=min(image.size); w,h=image.size
        image=image.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2)).resize((model.config.size,model.config.size))
        x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()[None,None]/255
        _,tokens,_=model.tokenizer.encode(x)
        state_ui.update(tokens=tokens,actions=[],steps=0,source=path)
        show(x); status.set(f"Loaded {Path(path).name}; choose codes and step.")

    @torch.no_grad()
    def step():
        if "tokens" not in state_ui:return
        try:
            values=[v.get() for v in choices]
            if any(v<0 or v>=k for v,k in zip(values,model.config.action_codes)): raise ValueError("Code outside range")
            code=torch.tensor(values)[None,None]
            state_ui["actions"].append(model.lam.lookup(code))
            state_ui["actions"]=state_ui["actions"][-(model.config.context-1):]
            history=state_ui["tokens"][:,-(model.config.context-1):]
            actions=torch.cat(state_ui["actions"],1)
            nxt=model.dynamics.next_tokens(history,actions)
            state_ui["tokens"]=torch.cat([history,nxt],1)
            state_ui["steps"]+=1; show(model.tokenizer.decode_ids(nxt))
            status.set(f"Generated step {state_ui['steps']} using latent codes {values}")
        except Exception as error:status.set(str(error))

    ttk.Button(frame,text="Load image",command=load).pack(side="left")
    ttk.Button(frame,text="Generate one step (space)",command=step).pack(side="left")
    ttk.Button(frame,text="Reset image",command=lambda:load(state_ui.get("source"))).pack(side="left")
    root.bind("<space>",lambda event:step())
    if a.image:load(a.image)
    root.mainloop()


if __name__=="__main__":main()
