"""Freeze deterministic train/validation selections before baseline training."""
import bisect
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from genie_diep.cohort import Cohort
from genie_diep.data import file_hash
from scripts.build_stationary_recovery import read,write


def selection(cohort, per_source):
    selected=[]
    for source in cohort.sources:
        cumulative=cohort.source_cumulative[source]
        for j in range(per_source):
            value=min(cumulative[-1]-1,int((j+.5)*cumulative[-1]/per_source))
            local=bisect.bisect_right(cumulative,value);sequence=cohort.by_source[source][local]
            start=value-(cumulative[local-1] if local else 0)
            selected.append(cohort.descriptor(sequence,start))
    return selected


def run():
    checkpoint=read('configs/longform-ninth-checkpoint-v1.json');binding=read('configs/baseline-data-v8.json')
    binding.update(cohort=checkpoint['cohort'],hashes=checkpoint['hashes'],cohort_verification='configs/longform-ninth-checkpoint-v1.json',
        note='V9 resolves the inherited Sprayer class question and adds five reviewed stationary blocks. Frozen cohort remains immutable; execution approval is stage-specific.',
        class_resolution='configs/sprayer-class-resolution-v1.json',class_resolution_sha256=file_hash('configs/sprayer-class-resolution-v1.json'))
    binding.pop('required_class_followup',None);write('configs/baseline-data-v9.json',binding)
    train=Cohort('.', 'configs/baseline-data-v9.json','train',16);val=Cohort('.', 'configs/baseline-data-v9.json','val',16)
    items=selection(val,4)
    evaluation=dict(schema_version=1,binding_sha256=train.binding_sha256,method='Four valid windows per validation source at deterministic window-count quantiles; fixed before model comparison.',
        validation=items,training_calibration=selection(train,1),
        stress_review='configs/longform-focused-review-v1.json',stress_policy='Report targeted coverage cases separately from the fixed source-stratified evaluation.',
        manual_scoring=dict(player_barrel='per-frame visible retention',food_bullets='annotate >=3px instances and TP/FP/FN; minimum 30 per category from three sources',
            angles='single-barrel only; minimum 30 from three sources; missing values do not pass',
            behavior='train-only code calibration, held-out intervention scoring; unknown code meanings remain unmeasured'),reserved_test_access=False)
    write('configs/baseline-evaluation-v1.json',evaluation)
    spec=read('configs/baseline-v1.json');spec.update(status='executable_stage_gated_baseline',local_execution_ready=False,
        data_binding='configs/baseline-data-v9.json',evaluation_selection='configs/baseline-evaluation-v1.json',
        cohort=checkpoint['cohort'],manifest_sha256=checkpoint['hashes']['manifest.jsonl'])
    spec['data']['training_inputs']='Verified 128px NPZ clips from the bound v9 cohort, using checked sequences only'
    spec['required_before_execution']=['Local real-data preflight must pass for this code/configuration.',
        'Before GPU allocation: explicit authorization, live rate/balance and measured FP16/memory/throughput pilot.',
        'Before later stages: scored G1/G2 promotion evidence bound to the exact prior checkpoint; missing measurements are not passes.']
    write('configs/training-baseline-v1.json',spec)
    out=Path('runs/training-readiness-v1/fidelity');out.mkdir(exist_ok=True)
    sources={s['id']:s for recipe in read('configs/longform-cohort-v6.json')['source_recipes'] for s in read(recipe)['sources']}
    caps={};evidence=[]
    for page in range((len(items)+7)//8):
        sheet=Image.new('RGB',(1600,460),'white');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(items[page*8:(page+1)*8]):
            seq,start=val.locate(item);offset=start+15
            for k in val.sequences[seq]['keys']:
                row=val.rows[k]
                if offset<row['frames']:break
                offset-=row['frames']
            sid=row['source_id'];source=sources[sid]
            if sid not in caps:
                assert file_hash(source['media_path'])==row['video_sha256'];caps[sid]=cv2.VideoCapture(source['media_path'])
            cap=caps[sid];cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_indices'][offset]);ok,frame=cap.read();assert ok
            x,y,w,h=row['crop'];native=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
            proxy=val._frames(k)[offset];expected=cv2.resize(native,(128,128),interpolation=cv2.INTER_AREA)
            assert np.array_equal(proxy,expected),'Native/proxy mismatch'
            col=j%4*400;line=j//4*230
            draw.text((col+2,line+2),f'{page*8+j}: {sid} | native / 128px',fill='black')
            sheet.paste(Image.fromarray(native).resize((200,200)),(col,line+25))
            sheet.paste(Image.fromarray(proxy).resize((200,200),Image.Resampling.NEAREST),(col+200,line+25))
        path=out/f'comparison-{page:02}.png';sheet.save(path);evidence.append(dict(path=path.as_posix(),sha256=file_hash(path)))
    for cap in caps.values():cap.release()
    write(out/'index.json',dict(evaluation_sha256=file_hash('configs/baseline-evaluation-v1.json'),frames=len(items),sources=len(val.sources),
        pixel_exact_resize=True,sheets=evidence,warning='Source-fidelity review, not model reconstruction quality.'))
    print(dict(train_windows=train.cumulative[-1],val_windows=val.cumulative[-1],validation_samples=len(items),validation_sources=len(val.sources),calibration_sources=len(train.sources)))


if __name__=='__main__':run()
