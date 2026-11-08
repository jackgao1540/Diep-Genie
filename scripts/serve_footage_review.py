"""Loopback-only footage review. Feedback is durable evidence, never training approval."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image

from genie_diep.data import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def identity(row):
    return row['source_id'], row['block_id']


def clip_id(row):
    return hashlib.sha256('|'.join(identity(row)).encode()).hexdigest()[:24]


class Library:
    def __init__(self, root, cohort='data/curated-longform-v6', feedback='runs/footage-review-v1/feedback.jsonl'):
        self.root = Path(root).resolve()
        self.cohort = self.root / cohort
        self.manifest_hash = file_hash(self.cohort/'manifest.jsonl')
        checkpoint = read(self.root/'configs/longform-sixth-checkpoint-v1.json')
        if self.cohort != self.root/checkpoint['cohort']:
            raise ValueError('Unsupported review cohort')
        for name, expected in checkpoint['hashes'].items():
            if file_hash(self.cohort/name) != expected:
                raise ValueError('Frozen cohort evidence changed: '+name)
        self.feedback_path = self.root / feedback
        self.lock = threading.RLock()
        self.render_lock = threading.Lock()
        self.cache = OrderedDict()
        self.verified_sources = set()
        self.token = secrets.token_urlsafe(32)
        self.sources = {}
        recipe = read(self.root/'configs/longform-cohort-v6.json')
        for path in recipe['source_recipes']:
            for source in read(self.root/path)['sources']:
                source = dict(source)
                info = (self.root/source['media_path']).with_suffix('.info.json')
                metadata = read(info) if info.exists() else {}
                source['title'] = metadata.get('title') or source['id']
                source['channel'] = metadata.get('channel') or metadata.get('uploader') or 'Unknown channel'
                self.sources[source['id']] = source
        active = {identity(r): r for r in rows(self.cohort/'manifest.jsonl')}
        self.records = {}
        self.public = {}
        previous = self.root/'data/curated-longform-v5'
        for r in rows(previous/'manifest.jsonl'):
            if r['split'] == 'test':
                continue
            accepted = identity(r) in active
            current = active.get(identity(r), r)
            self.add(current, 'accepted' if accepted else 'held',
                     (self.cohort if accepted else previous)/current['proxy_path'])
        focused = read(self.root/'configs/longform-focused-review-v1.json')
        for batch in focused['batches']:
            index_path = self.root/batch['index']
            if file_hash(index_path) != batch['index_sha256']:
                raise ValueError('Focused evidence changed')
            index = read(index_path)
            if file_hash(self.root/batch['request']) != index['request_sha256']:
                raise ValueError('Focused request changed')
            request = read(self.root/batch['request'])
            if len(index['rows']) != len(request['rows']):
                raise ValueError('Incomplete focused request')
            for row, requested in zip(index['rows'], request['rows']):
                if identity(row) != identity(requested):
                    raise ValueError('Review request mismatch')
                if not row['accepted']:
                    candidates = [r for r in rows(self.root/requested['screen']/'blocks.jsonl') if identity(r) == identity(row)]
                    if len(candidates) != 1:
                        raise ValueError('Ambiguous rejected clip')
                    self.add(candidates[0], 'rejected', None)
        sparse = read(self.root/'configs/longform-semantic-review-v1.json')
        sparse_index = self.root/sparse['review']/'index.json'
        if file_hash(sparse_index) != sparse['index_sha256']:
            raise ValueError('Sparse evidence changed')
        index = {r['review_index']: r for r in read(sparse_index)['rows']}
        for observation in sparse['annotations']:
            public = self.public.get(clip_id(index[observation['review_index']]))
            if public is None:
                continue
            public['reviewed'] = True
            public['evidence'].append('Sparse review: eight retained frames; observations are not input labels.')
            mapping = {'maze_wall': 'Maze walls', 'blue_pentagon': 'Pentagons', 'large_blue_pentagon': 'Large pentagons', 'aim_change': 'Aim changes', 'player_firing_evidence': 'Visible firing'}
            public['tags'].extend(label for field, label in mapping.items() if observation.get(field) == 'seen')
            public['palette'] = observation['player_palette']
        for batch in focused['batches']:
            index = {r['review_index']: r for r in read(self.root/batch['index'])['rows']}
            for observation in batch['annotations']:
                row = index[observation['review_index']]
                public = self.public[clip_id(row)]
                public['reviewed'] = True
                public['evidence'].append('Dense review: all 40 retained frames. '+observation['notes'])
                # Dense observations supersede sparse firing/aim attribution.
                public['tags'] = [t for t in public['tags'] if t not in {'Visible firing', 'No emission seen', 'Aim changes'}]
                if observation['firing'] == 'seen': public['tags'].append('Visible firing')
                if observation['firing'] == 'none_visible': public['tags'].append('No emission seen')
                if observation['aim_change'] == 'seen': public['tags'].append('Aim changes')
                if observation['scene_motion'] == 'visually_stable': public['tags'].append('Stable scenery')
                for field, label in [('maze_walls','Maze walls'), ('pentagons','Pentagons'), ('large_pentagons','Large pentagons')]:
                    if observation.get(field) == 'seen': public['tags'].append(label)
                if observation.get('quality_hold'): public['tags'].append('Suspected freeze')
        self.sequences = {}
        for sequence in read(self.cohort/'sequences.json'):
            if sequence['split'] == 'test': continue
            ids = [clip_id({'source_id':sequence['source_id'], 'block_id':b}) for b in sequence['blocks']]
            for i, cid in enumerate(ids):
                self.public[cid]['sequence_seconds'] = sequence['seconds']
                self.public[cid]['previous'] = ids[i-1] if i else None
                self.public[cid]['next'] = ids[i+1] if i+1 < len(ids) else None
        for public in self.public.values():
            public['tags'] = sorted(set(public['tags']))
        self.feedback = {}
        if self.feedback_path.exists():
            for entry in rows(self.feedback_path):
                record = self.records.get(entry.get('id'))
                if record and entry.get('content_sha256') == self.content_hash(record):
                    self.feedback[entry['id']] = entry
        self.summary = read(self.cohort/'summary.json')

    def add(self, row, status, proxy):
        if row['split'] not in {'train','val'}:
            raise ValueError('Reserved test is unavailable in review UI')
        source = self.sources[row['source_id']]
        if row['video_sha256'] != source['video_sha256'] or row['crop'] != source['crop']:
            raise ValueError('Source provenance mismatch')
        cid = clip_id(row)
        if cid in self.records:
            raise ValueError('Duplicate review clip')
        self.records[cid] = dict(row, _proxy=proxy)
        reason = ('Retained after existing screening and class/quality review. Sample-backed; not a guarantee for every frame.' if status == 'accepted' else
                  'Entire source withheld from v6 after suspected frozen gameplay at 70–74 seconds. No training use pending source audit.' if status == 'held' else
                  'Screen rejected this clip: '+', '.join(row['reasons'])+'. Retained for comparison; not reinstated.')
        self.public[cid] = dict(id=cid, source_id=row['source_id'], block_id=row['block_id'], title=source['title'], channel=source['channel'],
            split=row['split'], start=row['start'], seconds=row['frames']/row['fps'], fps=row['fps'], frames=row['frames'],
            crop=row['crop'], has_proxy=proxy is not None, status=status, reason=reason, tags=[], evidence=[], reviewed=False,
            sequence_seconds=4, previous=None, next=None, palette=None)

    @staticmethod
    def content_hash(record):
        return hashlib.sha256(json.dumps({k:record.get(k) for k in ('video_sha256','proxy_sha256','source_frame_indices','crop')},sort_keys=True).encode()).hexdigest()

    def catalog(self):
        with self.lock:
            return dict(clips=list(self.public.values()), feedback=dict(self.feedback), token=self.token,
                meta=dict(cohort=self.cohort.name, accepted_hours=self.summary['curation_accepted_hours'],
                    accepted_sources=self.summary['source_count'], reserved_test_seconds=self.summary['seconds_by_split'].get('test',0),
                    manifest_sha256=self.manifest_hash, training_approved=False))

    def save(self, payload):
        cid, decision, note = payload.get('id'), payload.get('decision'), payload.get('note','')
        if cid not in self.records or decision not in {'good','flag','unsure','clear'}:
            raise ValueError('Invalid review')
        if not isinstance(note,str) or len(note)>4000:
            raise ValueError('Note must be at most 4000 characters')
        entry = dict(id=cid,source_id=self.records[cid]['source_id'],block_id=self.records[cid]['block_id'],
            content_sha256=self.content_hash(self.records[cid]),cohort_manifest_sha256=self.manifest_hash,
            decision=decision,note=note,updated_at=datetime.now(timezone.utc).isoformat(),training_approved=False)
        with self.lock:
            self.feedback_path.parent.mkdir(parents=True,exist_ok=True)
            with self.feedback_path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
            self.feedback[cid] = entry
        return entry

    def picture(self, cid, mode):
        if cid not in self.records or mode not in {'thumb','proxy','original','context'}:
            raise KeyError('Unknown clip or view')
        key = (cid,mode)
        with self.render_lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            record = self.records[cid]
            if mode in {'thumb','proxy'} and record['_proxy']:
                if file_hash(record['_proxy']) != record['proxy_sha256']:
                    raise ValueError('Proxy changed; review stopped')
                with np.load(record['_proxy'],allow_pickle=False) as data:frames=data['frames']
                if frames.shape != (40,128,128,3):raise ValueError('Unexpected proxy shape')
            else:
                source = self.sources[record['source_id']]
                media = self.root/source['media_path']
                if source['id'] not in self.verified_sources:
                    if file_hash(media) != source['video_sha256']:raise ValueError('Original changed; review stopped')
                    self.verified_sources.add(source['id'])
                cap = cv2.VideoCapture(str(media))
                targets = record['source_frame_indices']
                if mode in {'thumb','context'}:targets=targets[20:21]
                cap.set(cv2.CAP_PROP_POS_FRAMES,targets[0]);current=targets[0];frames=[]
                try:
                    for target in targets:
                        while current<target:
                            if not cap.grab():raise ValueError('Original decode failed')
                            current+=1
                        ok,frame=cap.read();current+=1
                        if not ok:raise ValueError('Original decode failed')
                        if mode!='context':
                            x,y,w,h=record['crop'];frame=frame[y:y+h,x:x+w]
                        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                finally:cap.release()
                frames=np.stack(frames)
            if mode in {'thumb','context'}:
                result=Image.fromarray(frames[min(20,len(frames)-1)])
            else:
                _,h,w,_=frames.shape
                result=Image.new('RGB',(w*8,h*5))
                for i,frame in enumerate(frames):result.paste(Image.fromarray(frame),(i%8*w,i//8*h))
            buffer=io.BytesIO();result.save(buffer,format='PNG');value=buffer.getvalue()
            self.cache[key]=value
            while len(self.cache)>48:self.cache.popitem(last=False)
            return value


def make_handler(library, assets):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):pass

        def send(self, status, body, kind='application/json'):
            if isinstance(body,(dict,list)):body=json.dumps(body).encode()
            if isinstance(body,str):body=body.encode()
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def local_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.local_host():return self.send(403,{'error':'Local access only'})
            path=urlsplit(self.path).path
            try:
                if path=='/api/catalog':return self.send(200,library.catalog())
                if path=='/api/reviews':
                    with library.lock: reviews=list(library.feedback.values())
                    return self.send(200,dict(cohort_manifest_sha256=library.manifest_hash,reviews=reviews,training_approved=False))
                if path.startswith('/api/image/'):
                    parts=path.split('/')
                    if len(parts)!=5:raise KeyError('Unknown path')
                    return self.send(200,library.picture(parts[3],parts[4]),'image/png')
                files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
                if path not in files:return self.send(404,{'error':'Not found'})
                name,kind=files[path];return self.send(200,(assets/name).read_bytes(),kind)
            except KeyError:return self.send(404,{'error':'Unknown clip'})
            except (ValueError,OSError) as error:return self.send(409,{'error':str(error)})

        def do_POST(self):
            if not self.local_host():return self.send(403,{'error':'Local access only'})
            origin=self.headers.get('Origin')
            allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if origin not in allowed or not secrets.compare_digest(self.headers.get('X-Review-Token',''),library.token):
                return self.send(403,{'error':'Review request rejected'})
            if self.path!='/api/review':return self.send(404,{'error':'Not found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=20000:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('Invalid request')
                return self.send(200,library.save(payload))
            except (ValueError,OSError) as error:return self.send(400,{'error':str(error)})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));args=p.parse_args()
    cv2.setNumThreads(1)
    library=Library(args.root)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(library,Path(args.root)/'web/footage-review'))
    print(f'Footage review ready: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
