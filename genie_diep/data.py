"""Local-only acquisition, deterministic source splits and bounded clip extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def source_split(source_id):
    """Assign an entire source, never individual overlapping windows, to a split."""
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 10
    return "val" if bucket < 2 else "train"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(url, out, node=None):
    """One public video, capped resolution/size; no browser cookies or remote scripts."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
               "--no-remote-components", "--socket-timeout", "20", "--retries", "2",
               "--max-filesize", "500M", "--write-info-json", "--no-write-playlist-metafiles",
               "--download-archive", str(out / "archive.txt"),
               "-f", "best[height<=480]/bestvideo[height<=480]",
               "-o", str(out / "%(id)s.%(ext)s")]
    if node:
        command += ["--js-runtimes", "node:" + node]
    subprocess.run(command + [url], check=True)


def extract(video, out, source_id, start=0., seconds=120., fps=10., size=64, length=16, crop=None, require_tank=False):
    """Sample by timestamp, reject cuts/freezes; emit provenance per nonoverlapping clip.

    Quality heuristics are triage only. Contact-sheet review remains required.
    crop is x,y,width,height in source pixels, applied before aspect-preserving resize.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or fps <= 0 or seconds <= 0 or length < 2:
        raise ValueError("Invalid frame rate, duration or clip length")
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames, timestamps, rows, rejected = [], [], [], 0
    sample_at = start
    frame_number = round(start * source_fps)
    end = start + seconds
    checksum = file_hash(video)
    split = source_split(source_id)
    while frame_number / source_fps < end:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_number / source_fps
        frame_number += 1
        if timestamp + 0.5/source_fps < sample_at:
            continue
        sample_at += 1 / fps
        if crop:
            x, y, w, h = crop
            if min(x,y) < 0 or min(w,h) <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                raise ValueError("Crop outside source frame")
            frame = frame[y:y+h, x:x+w]
        h, w = frame.shape[:2]
        # Center square preserves tank geometry; full-frame original retained for audit.
        side = min(h,w)
        frame = frame[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
        frame = cv2.cvtColor(cv2.resize(frame, (size,size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        frames.append(frame)
        timestamps.append(timestamp)
        if len(frames) == length:
            array = np.stack(frames)
            delta = np.abs(np.diff(array.astype(np.float32), axis=0)).mean((1,2,3))/255
            reason = "cut" if delta.max() > .20 else "freeze" if delta.mean() < .0005 else None
            if require_tank and not all(center_tank_present(f) for f in array):
                reason = "no_center_tank"
            if reason:
                rejected += 1
            else:
                name = f"{source_id}_{timestamps[0]:010.3f}.npy"
                np.save(out / name, array, allow_pickle=False)
                rows.append(dict(path=name, source_id=source_id, split=split,
                                 start=timestamps[0], end=timestamps[-1], fps=fps,
                                 shape=list(array.shape), video_sha256=checksum,
                                 mean_delta=float(delta.mean()), crop=crop, source_fps=source_fps,
                                 center_tank_required=require_tank, reviewed=False))
            frames, timestamps = [], []
    cap.release()
    manifest = out / f"{source_id}.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    print(json.dumps(dict(accepted=len(rows), rejected=rejected, split=split, manifest=str(manifest))))
    return rows


def center_tank_present(frame):
    """Conservative cyan-body heuristic for the selected blue-player videos only.

    Reject invisible/dead players; this is not a universal tank/class detector.
    """
    h,w=frame.shape[:2]
    r,g,b=frame.astype(np.int16).transpose(2,0,1)
    mask=((g-r>45)&(b-r>60)&(b>130)&(g>95)).astype(np.uint8)
    count,_,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    return any(3<=stats[i,cv2.CC_STAT_AREA]<=h*w*.06 and
               np.linalg.norm(centroids[i]-np.array([w/2,h/2]))<min(h,w)*.16
               for i in range(1,count))


class ClipDataset(Dataset):
    def __init__(self, root, split="train", length=5):
        self.root, self.length = Path(root), length
        self.rows = []
        all_rows = [json.loads(line) for path in sorted(self.root.glob("*.jsonl"))
                    for line in path.read_text(encoding="utf-8").splitlines() if line]
        sources = {}
        hashes = {}
        for row in all_rows:
            previous = sources.setdefault(row["source_id"], row["split"])
            if previous != row["split"]:
                raise ValueError("Source leakage across train/validation")
            if row.get("video_sha256"):
                old=hashes.setdefault(row["video_sha256"],row["split"])
                if old!=row["split"]:
                    raise ValueError("Identical video content leakage across train/validation")
            if row["split"] == split and row["shape"][0] >= length:
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No {split} clips in {root}; never fall back to the training set")

    def __len__(self):
        return len(self.rows)

    def get_clip(self, index, start=None):
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Clip path escapes dataset")
        clip = np.load(path, allow_pickle=False, mmap_mode="r")
        if start is None:
            start = int(torch.randint(len(clip)-self.length+1,(1,))) if row["split"] == "train" else 0
        if start < 0 or start+self.length > len(clip):
            raise ValueError("Requested window is outside clip")
        return torch.from_numpy(clip[start:start+self.length].copy()).permute(0,3,1,2).float()/255

    def __getitem__(self,index):
        return self.get_clip(index)

    def source_balanced_weights(self):
        """Equal total sampling mass per training source; never mix in validation."""
        if any(row['split']!='train' for row in self.rows):
            raise ValueError('Source-balanced sampling is training-only')
        counts={}
        for row in self.rows:counts[row['source_id']]=counts.get(row['source_id'],0)+1
        return torch.tensor([1/counts[row['source_id']] for row in self.rows],dtype=torch.double)

    def representative_indices(self,limit=64):
        """Round-robin across sources, uniformly spaced within each source."""
        groups={}
        for i,row in enumerate(self.rows): groups.setdefault(row["source_id"],[]).append(i)
        queues=[]
        for source,indices in sorted(groups.items()):
            indices=sorted(indices,key=lambda i:self.rows[i].get("start",0))
            n=min(len(indices),max(1,int(np.ceil(limit/len(groups)))))
            queues.append([indices[int(j)] for j in np.linspace(0,len(indices)-1,n)])
        result=[]
        while queues and len(result)<limit:
            for queue in queues:
                if queue and len(result)<limit: result.append(queue.pop(0))
            queues=[q for q in queues if q]
        return result


def contact_sheet(root, output, limit=12):
    from PIL import Image, ImageDraw
    paths = sorted(Path(root).glob("*.npy"))[:limit]
    sheet = Image.new("RGB", (640, max(1,len(paths))*150), "white")
    draw = ImageDraw.Draw(sheet)
    for i,path in enumerate(paths):
        clip = np.load(path, allow_pickle=False)
        draw.text((4,i*150),path.stem,fill="black")
        for j,t in enumerate(np.linspace(0,len(clip)-1,5).astype(int)):
            sheet.paste(Image.fromarray(clip[t]).resize((128,128)),(j*128,i*150+20))
    sheet.save(output)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("download"); a.add_argument("url"); a.add_argument("--out",default="data/raw"); a.add_argument("--node")
    e=sub.add_parser("extract"); e.add_argument("video"); e.add_argument("--out",default="data/clips"); e.add_argument("--source-id",required=True)
    e.add_argument("--start",type=float,default=0); e.add_argument("--seconds",type=float,default=120); e.add_argument("--size",type=int,default=64)
    e.add_argument("--crop",type=int,nargs=4)
    e.add_argument("--require-tank",action="store_true")
    c=sub.add_parser("sheet"); c.add_argument("root"); c.add_argument("output")
    args=vars(p.parse_args()); command=args.pop("command")
    {"download":acquire,"extract":extract,"sheet":contact_sheet}[command](**args)


if __name__ == "__main__":
    main()
