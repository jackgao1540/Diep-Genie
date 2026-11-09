"""Native evidence and fresh full-OCR screens for triaged low-motion candidates."""
import json
from pathlib import Path
from types import SimpleNamespace
from genie_diep.data import file_hash
from scripts.prepare_stationary_recovery import native_sheet, read, rows, write
from scripts.screen_longform import screen_source


def run():
    from rapidocr import RapidOCR
    out=Path('runs/remaining-low-motion-v1');triage=read('configs/remaining-low-motion-triage-v1.json')
    index=read(out/'index.json');assert file_hash(out/'index.json')==triage['index_sha256']
    engine=RapidOCR(params={'Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':2,
                           'EngineConfig.onnxruntime.inter_op_num_threads':1,'Det.limit_side_len':384})
    evidence=[]
    for decision in triage['decisions']:
        if decision['decision']!='candidate':continue
        q=next(r for r in index['remaining'] if r['queue_index']==decision['queue_index'])
        assert not q['reserved_test'] and q['split'] in {'train','val'}
        screen=Path(q['screen']);config=read(screen/'configuration.json')
        old=next(r for r in rows(screen/'blocks.jsonl') if r['block_id']==q['block_id'])
        assert file_hash(q['media_path'])==q['video_sha256']==old['video_sha256']
        folder=out/f"candidate-{q['queue_index']:02}";folder.mkdir(exist_ok=True)
        native=native_sheet(q['media_path'],q['crop'],old['source_frame_indices'][0],round(old['end']*old['source_fps']),folder,'native')
        source=dict(config['source']);interval,block=map(int,old['block_id'].split('_'))
        start=source['intervals'][interval][0]+block*config['block_seconds']
        source['intervals']=[[start,start+4]];source['preserved_origin']=dict(screen=q['screen'],block_id=q['block_id'])
        args=SimpleNamespace(out=str(folder/'rescreen'),fps=10,block_seconds=4,padding=24,proxy_size=128,
                             text_policy='native-names',caption_policy='score-captions',motion_policy='review',previous_screen=[])
        screen_source(source,args,engine,file_hash('configs/remaining-low-motion-triage-v1.json'))
        path=folder/'rescreen'/source['id']/'blocks.jsonl';new=rows(path)[0]
        for field in ['source_frame_indices','video_sha256','crop','split','fps','frames']:
            assert old[field]==new[field],field
        r=dict(old,proxy_path=(path.parent/new['proxy']).as_posix(),proxy_sha256=new['proxy_sha256'],
               recovery_screen=path.as_posix(),media_path=q['media_path'])
        evidence.append(dict(queue_index=q['queue_index'],row=r,native_review=native,boundaries=[],rescreen_sha256=file_hash(path)))
        write(out/'candidates-native.json',dict(base_manifest_sha256=index['parent_manifest_sha256'],rows=evidence))
        print(json.dumps(dict(queue_index=q['queue_index'],native_frames=native['native_frames'],reasons=new['reasons'],ocr_frames=new['ocr_frames'])),flush=True)


if __name__=='__main__':run()
