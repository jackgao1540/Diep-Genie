"""Export only the explicitly selected, user-authorized public showcase samples."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.serve_footage_review import Library, clip_id, read


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def webp(image, path):
    image.save(path,format='WEBP',lossless=True,method=6)
    # Lossless frame showcases must decode to the exact exported RGB pixels.
    with Image.open(path) as decoded:
        if not np.array_equal(np.array(image.convert('RGB')),np.array(decoded.convert('RGB'))):
            raise ValueError('Image export changed pixels')
    return {'path':'assets/'+path.name,'sha256':digest(path),'bytes':path.stat().st_size}


def run(root, destination):
    root=Path(root).resolve();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    recipe=read(root/'configs/public-showcase-v1.json')
    library=Library(root,feedback='runs/public-showcase-v1/unused-feedback.jsonl')
    focused=read(root/'configs/longform-focused-review-v1.json')
    batches={b['name']:read(root/b['index'])['rows'] for b in focused['batches']}
    clips=[];published=set()
    for item in recipe['selection']:
        r=next(r for r in batches[item['batch']] if r['review_index']==item['review_index'])
        cid=clip_id(r);p=library.public[cid]
        if p['split'] not in {'train','val'} or cid in published:
            raise ValueError('Reserved test or duplicate publication')
        published.add(cid)
        original=Image.open(io.BytesIO(library.picture(cid,'original'))).convert('RGB')
        if p['has_proxy']:
            processed=Image.open(io.BytesIO(library.picture(cid,'proxy'))).convert('RGB')
            processed_kind='retained_proxy' if p['status']=='accepted' else 'withheld_proxy'
        else:
            # Rejected records have no accepted training proxy. Label this preview explicitly.
            w,h=p['crop'][2:];array=np.asarray(original);processed=Image.new('RGB',(1024,640))
            for i in range(40):
                frame=array[i//8*h:(i//8+1)*h,i%8*w:(i%8+1)*w]
                frame=cv2.resize(frame,(128,128),interpolation=cv2.INTER_AREA)
                processed.paste(Image.fromarray(frame),(i%8*128,i//8*128))
            processed_kind='illustrative_resize_not_training_data'
        thumb=processed.crop((4*128,2*128,5*128,3*128))
        media={name:webp(image,assets/f"{item['id']}-{name}.webp") for name,image in [('original',original),('processed',processed),('thumbnail',thumb)]}
        clips.append(dict(id=item['id'],title=item['title'],description=item['description'],status=p['status'],split=p['split'],
            source_id=p['source_id'],source_title=p['title'],creator=p['channel'],
            source_url=f"https://www.youtube.com/watch?v={p['source_id']}&t={int(p['start'])}s",
            start=p['start'],duration=4,fps=10,frames=40,source_crop_size=p['crop'][2:],processed_size=[128,128],
            processed_kind=processed_kind,tags=p['tags'],media=media,
            provenance=dict(video_sha256=r['video_sha256'],proxy_sha256=r.get('proxy_sha256'),crop=r['crop'],source_frame_indices=r['source_frame_indices'])))
    media_bytes=sum(m['bytes'] for c in clips for m in c['media'].values())
    if media_bytes>recipe['max_media_bytes']:raise ValueError('Public sample size cap exceeded')
    # Deliberate field whitelist: no local paths, user feedback, credentials or test samples.
    data=dict(schema_version=1,updated='2026-11-08',project='Diep-Genie',dataset_version='v6',
        retained_hours=library.summary['curation_accepted_hours'],retained_sources=library.summary['source_count'],
        retained_blocks=library.summary['clips'],training_approved=False,model_status='Research prototype; no playable trained world model.',
        cohort_manifest_sha256=library.manifest_hash,selection_sha256=digest(root/'configs/public-showcase-v1.json'),
        sample_count=len(clips),sample_seconds=4*len(clips),media_bytes=media_bytes,
        sample_policy='Hand-selected train/validation examples; not a random or complete representation. No reserved-test footage. No rating or feedback collection.',clips=clips)
    (out/'showcase.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'.nojekyll').write_text('',encoding='utf-8')
    print(json.dumps(dict(samples=len(clips),media_bytes=media_bytes,files=len(clips)*3,output=str(out))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--out',default='docs')
    args=parser.parse_args();run(args.root,args.out)
