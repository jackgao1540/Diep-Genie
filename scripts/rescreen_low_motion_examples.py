"""Re-screen the four discussed clips with preservation and complete low-motion OCR."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from genie_diep.data import file_hash
from scripts.screen_longform import screen_source


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def run(destination):
    from rapidocr import RapidOCR
    out=Path(destination)
    if out.exists():raise FileExistsError('Use a new review checkpoint')
    request_path=Path('configs/longform-focused-request-v2.json')
    request=read(request_path)
    focused=read('configs/longform-focused-review-v1.json')
    batch=next(b for b in focused['batches'] if b['name']=='targeted')
    if file_hash(batch['index'])!=batch['index_sha256']:raise ValueError('Focused evidence changed')
    index=read(batch['index'])
    if index['request_sha256']!=file_hash(request_path):raise ValueError('Request changed')
    engine=RapidOCR(params={'Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':2,
                           'EngineConfig.onnxruntime.inter_op_num_threads':1,'Det.limit_side_len':384})
    args=SimpleNamespace(out=str(out),fps=10,block_seconds=4,padding=24,proxy_size=128,
                         text_policy='native-names',caption_policy='score-captions',motion_policy='review',previous_screen=[])
    evidence=[]
    for item in request['rows']:
        if 'screen' not in item:continue
        screen=Path(item['screen']);config=read(screen/'configuration.json')
        matches=[json.loads(s) for s in (screen/'blocks.jsonl').read_text().splitlines() if s.strip()]
        old=next(r for r in matches if r['block_id']==item['block_id'])
        if old['split'] not in {'train','val'}:raise ValueError('Reserved test forbidden')
        source=dict(config['source']);interval,block=map(int,old['block_id'].split('_'))
        start=source['intervals'][interval][0]+block*config['block_seconds']
        source['intervals']=[[start,start+config['block_seconds']]]
        source['preserved_origin']=dict(screen=str(screen),block_id=old['block_id'])
        screen_source(source,args,engine,file_hash(request_path))
        path=out/source['id']/'blocks.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
        if len(rows)!=1:raise ValueError('Expected exactly one block')
        new=rows[0]
        for field in ['source_frame_indices','video_sha256','crop','split','fps','frames']:
            if old[field]!=new[field]:raise ValueError('Rescreen changed '+field)
        original=next(r for r in index['rows'] if r['source_id']==old['source_id'] and r['block_id']==old['block_id'])
        for name in ['sheet','context']:
            if file_hash(Path(batch['index']).parent/original[name])!=original[name+'_sha256']:raise ValueError('Visual evidence changed')
        evidence.append(dict(source_id=old['source_id'],original_block_id=old['block_id'],split=old['split'],
                             original_screen=str(screen),rescreen=str(path),rescreen_sha256=file_hash(path),
                             video_sha256=new['video_sha256'],proxy_sha256=new['proxy_sha256'],
                             source_frame_indices=new['source_frame_indices'],crop=new['crop'],
                             reasons=new['reasons'],ocr_frames=new['ocr_frames'],detections=new['detections'],
                             metrics=new['metrics'],visual_evidence=original,training_approved=False))
    report=dict(rows=evidence,training_approved=False,test_pixels_reviewed=False,
                warning='Rescreening is not reinstatement. Preserve original decisions; full recovery integration remains separate.')
    (out/'review-material.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps([dict(source_id=r['source_id'],ocr_frames=r['ocr_frames'],reasons=r['reasons']) for r in evidence]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
