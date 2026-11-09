"""Hash-bound long-form windows; checked joins only, no reserved-test decoding."""
from collections import OrderedDict, defaultdict
import bisect
import json
from pathlib import Path
import numpy as np
import torch
from .data import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Cohort:
    def __init__(self, root, binding, split, length=16, cache_size=24):
        if split not in {'train','val'}:raise ValueError('Reserved test is unavailable for development')
        if length<2:raise ValueError('Window needs at least two frames')
        self.root=Path(root).resolve();self.binding_path=self.root/binding
        self.binding=read(self.binding_path);self.binding_sha256=file_hash(self.binding_path)
        self.path=(self.root/self.binding['cohort']).resolve()
        if not self.path.is_relative_to(self.root):raise ValueError('Cohort escapes project')
        for name,sha in self.binding['hashes'].items():
            if name not in {'summary.json','manifest.jsonl','sequences.json'}:raise ValueError('Unexpected binding file')
            if file_hash(self.path/name)!=sha:raise ValueError('Frozen cohort changed')
        self.length=length;self.split=split;self.cache_size=cache_size;self.cache=OrderedDict()
        all_rows=[json.loads(l) for l in (self.path/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l]
        self.rows={};sources={};videos={}
        for r in all_rows:
            k=(r['source_id'],r['block_id'])
            if k in self.rows:raise ValueError('Duplicate block identity')
            self.rows[k]=r
            for values,key in [(sources,r['source_id']),(videos,r['video_sha256'])]:
                if values.setdefault(key,r['split'])!=r['split']:raise ValueError('Cross-split source leakage')
            if r['fps']!=10 or r['frames']!=len(r['source_frame_indices']):raise ValueError('Invalid frame contract')
            if not r.get('curation_accepted') or r.get('reasons'):raise ValueError('Unaccepted block')
        self.sequences=[];seen=set();by_source=defaultdict(list)
        for sequence in read(self.path/'sequences.json'):
            keys=[(sequence['source_id'],b) for b in sequence['blocks']]
            if not keys or any(k in seen or k not in self.rows for k in keys):raise ValueError('Invalid sequence membership')
            seen.update(keys);rr=[self.rows[k] for k in keys]
            if any(r['split']!=sequence['split'] for r in rr):raise ValueError('Sequence split mismatch')
            for a,b in zip(rr,rr[1:]):
                if any(a[f]!=b[f] for f in ['source_id','video_sha256','crop','fps','source_fps','split']):raise ValueError('Incompatible join')
                if abs(a['end']-b['start'])>=.6/a['source_fps']:raise ValueError('Sequence bridges a gap')
            count=sum(r['frames'] for r in rr)-length+1
            if sequence['split']==split and count>0:
                i=len(self.sequences);self.sequences.append(dict(source_id=sequence['source_id'],keys=keys,windows=count))
                by_source[sequence['source_id']].append(i)
        if seen!=set(self.rows):raise ValueError('Incomplete sequence coverage')
        if not self.sequences:raise ValueError('No eligible windows')
        self.by_source=dict(sorted(by_source.items()));self.sources=list(self.by_source)
        self.cumulative=np.cumsum([s['windows'] for s in self.sequences]).tolist()
        self.source_cumulative={s:np.cumsum([self.sequences[i]['windows'] for i in ids]).tolist() for s,ids in self.by_source.items()}

    def _frames(self, key):
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        r=self.rows[key]
        if r['split']!=self.split:raise ValueError('Split decoding denied')
        path=(self.path/r['proxy_path']).resolve()
        if not path.is_relative_to(self.path):raise ValueError('Proxy escapes cohort')
        if file_hash(path)!=r['proxy_sha256']:raise ValueError('Proxy changed')
        with np.load(path,allow_pickle=False) as data:frames=data['frames']
        if frames.dtype!=np.uint8 or frames.shape!=(r['frames'],128,128,3):raise ValueError('Invalid proxy tensor')
        self.cache[key]=frames
        while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return frames

    def window(self, sequence, start):
        seq=self.sequences[sequence]
        if not 0<=start<seq['windows']:raise ValueError('Window outside checked sequence')
        remaining=self.length;offset=start;parts=[]
        for k in seq['keys']:
            count=self.rows[k]['frames']
            if offset>=count:offset-=count;continue
            part=self._frames(k)[offset:offset+remaining];parts.append(part);remaining-=len(part);offset=0
            if remaining==0:break
        if remaining:raise ValueError('Incomplete window')
        return torch.from_numpy(np.concatenate(parts)).permute(0,3,1,2).float()/255

    def descriptor(self, sequence, start):
        s=self.sequences[sequence]
        return dict(source_id=s['source_id'],first_block=s['keys'][0][1],start_frame=start,length=self.length)

    def locate(self, descriptor):
        if descriptor['length']!=self.length:raise ValueError('Evaluation window length changed')
        for i,s in enumerate(self.sequences):
            if s['source_id']==descriptor['source_id'] and s['keys'][0][1]==descriptor['first_block']:
                if not 0<=descriptor['start_frame']<s['windows']:raise ValueError('Invalid evaluation offset')
                return i,descriptor['start_frame']
        raise ValueError('Evaluation sequence absent')


class MixedSampler:
    """Half source-uniform, half window-uniform, with independently saved RNG."""
    def __init__(self, cohort, seed=42, source_fraction=.5):
        if cohort.split!='train' or not 0<=source_fraction<=1:raise ValueError('Invalid training sampler')
        self.cohort=cohort;self.fraction=source_fraction;self.rng=torch.Generator().manual_seed(seed);self.draws=0

    def draw(self):
        c=self.cohort
        if torch.rand((),generator=self.rng).item()<self.fraction:
            source=c.sources[int(torch.randint(len(c.sources),(1,),generator=self.rng))]
            cumulative=c.source_cumulative[source];value=int(torch.randint(cumulative[-1],(1,),generator=self.rng))
            local=bisect.bisect_right(cumulative,value);sequence=c.by_source[source][local]
            start=value-(cumulative[local-1] if local else 0)
        else:
            value=int(torch.randint(c.cumulative[-1],(1,),generator=self.rng));sequence=bisect.bisect_right(c.cumulative,value)
            start=value-(c.cumulative[sequence-1] if sequence else 0)
        self.draws+=1
        return sequence,start

    def batch(self, count):return torch.stack([self.cohort.window(*self.draw()) for _ in range(count)])
    def state_dict(self):return dict(rng=self.rng.get_state(),draws=self.draws,fraction=self.fraction)
    def load_state_dict(self,state):
        if state['fraction']!=self.fraction:raise ValueError('Sampling mixture changed')
        self.rng.set_state(state['rng'].cpu());self.draws=state['draws']
