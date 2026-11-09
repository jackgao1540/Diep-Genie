"""Separate oracle prediction diagnostics from future-free commanded rollouts."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .cohort import Cohort, read
from .data import file_hash
from .model import Config, WorldModel, translate_frames


def alternative_codes(codes, capacities):
    result=[('reference',codes.clone())]
    for group,count in enumerate(capacities):
        for value in range(count):
            if value==int(codes[0,-1,group]):continue
            changed=codes.clone();changed[...,group]=value
            result.append((f'group-{group}-code-{value}',changed))
    return result


def constant_velocity(prefix):
    """One-step scene extrapolation estimated solely from the last two past frames."""
    if prefix.shape[1]<2:raise ValueError('Velocity baseline requires two past frames')
    shifts=[]
    for sample in prefix:
        a=sample[-2].mean(0).detach().float().cpu().numpy();b=sample[-1].mean(0).detach().float().cpu().numpy()
        shift,response=cv2.phaseCorrelate(a,b)
        if not np.isfinite(shift).all() or response<.05:shift=(0.,0.)
        shifts.append(np.clip(shift,-16,16))
    displacement=torch.as_tensor(np.array(shifts),device=prefix.device,dtype=prefix.dtype)[:,None]
    return translate_frames(prefix[:,-1:],displacement)


@torch.no_grad()
def oracle_interventions(model, clip):
    """Infer a target label, vary every group independently, report only oracle use."""
    _,chosen,_=model.lam.infer(clip[:,-2:])
    candidates=alternative_codes(chosen,model.config.action_codes)
    target=clip[:,-1:];past=clip[:,-2:-1];motion=(target-past).abs().mean(2,keepdim=True)>.03
    weight=.05+motion.float()
    def error(pred):return float(((pred-target).square()*weight).sum()/weight.expand_as(target).sum())
    reference=model.lam.reconstruct(past,model.lam.lookup(chosen))
    result=[]
    for label,code in candidates:
        pred=model.lam.reconstruct(past,model.lam.lookup(code))
        result.append(dict(command=label,codes=code.flatten().tolist(),motion_mse=error(pred),effect_mse=float((pred-reference).square().mean())))
    return dict(oracle_only=True,copy_mse=error(past),velocity_mse=error(constant_velocity(clip[:,:-1])),alternatives=result)


@torch.no_grad()
def commanded_rollout(model,prefix,codes,frames,stage='dynamics'):
    """The API takes no future frames. Codes are supplied, not inferred from targets."""
    if frames<1 or prefix.shape[1]<2:raise ValueError('Invalid rollout request')
    if codes.shape!=(prefix.shape[0],1,len(model.config.action_codes)):raise ValueError('Command shape mismatch')
    for i,count in enumerate(model.config.action_codes):
        if codes[...,i].min()<0 or codes[...,i].max()>=count:raise ValueError('Command code outside book')
    action=model.lam.lookup(codes)
    if stage=='actions':
        current=prefix[:,-1:];output=[]
        for _ in range(frames):
            current=model.lam.reconstruct(current,action);output.append(current)
        return torch.cat(output,1)
    if stage!='dynamics':raise ValueError('Rollout needs actions or dynamics stage')
    _,history,_=model.tokenizer.encode(prefix)
    prior,_,_=model.lam.infer(prefix);output=[]
    for _ in range(frames):
        history=history[:,-(model.config.context-1):]
        prior=torch.cat([prior,action],1)[:,-history.shape[1]:]
        nxt=model.dynamics.next_tokens(history,prior)
        output.append(model.tokenizer.decode_ids(nxt));history=torch.cat([history,nxt],1)
    return torch.cat(output,1)


def gate_g1(observations, requirements):
    """Missing human object/angle measurements cannot become a passing gate."""
    if not observations:return dict(status='unmeasured',reason='No scored reconstruction anchors')
    required={'player_retained','barrel_retained','source_id','food_tp','food_fp','food_fn','bullet_tp','bullet_fp','bullet_fn','angle_error_degrees'}
    if any(not required<=set(r) for r in observations):return dict(status='unmeasured',reason='Incomplete object or angle measurements')
    result={};sufficient=True
    for category in ['food','bullet']:
        tp=sum(r[category+'_tp'] for r in observations);fp=sum(r[category+'_fp'] for r in observations);fn=sum(r[category+'_fn'] for r in observations)
        sources={r['source_id'] for r in observations if r[category+'_tp']+r[category+'_fn']>0}
        enough=tp+fn>=requirements['minimum_instances_per_object_category'] and len(sources)>=requirements['minimum_sources_per_object_category']
        sufficient &= enough;result[category]=dict(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,instances=tp+fn,sources=len(sources))
    angles=[r for r in observations if r['angle_error_degrees'] is not None]
    sufficient &= len(angles)>=30 and len({r['source_id'] for r in angles})>=3 and len(observations)>=30
    result['player_and_barrel_retention']=sum(bool(r['player_retained'] and r['barrel_retained']) for r in observations)/len(observations)
    result['angle_pass_fraction']=sum(r['angle_error_degrees']<=requirements['single_barrel_angle_error_max_degrees'] for r in angles)/len(angles) if angles else 0
    passed=result['player_and_barrel_retention']>=requirements['player_and_barrel_retention_min_fraction'] and result['angle_pass_fraction']>=requirements['single_barrel_angle_pass_min_fraction']
    passed &= all(result[c][m]>=requirements['food_and_bullet_precision_recall_min_each'] for c in ['food','bullet'] for m in ['precision','recall'])
    return dict(status=('pass' if passed else 'fail') if sufficient else 'unmeasured',measurements=result)


def run(args):
    torch.set_num_threads(2);root=Path(args.root);spec=read(root/args.spec)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=WorldModel(Config(**state['config']));model.load_state_dict(state['model']);model.to(args.device).eval()
    dataset=Cohort(root,spec['data_binding'],'val',model.config.context);selection=read(root/spec['evaluation_selection'])
    if dataset.binding_sha256!=state['contract']['binding_sha256'] or selection['binding_sha256']!=dataset.binding_sha256:raise ValueError('Evaluation binding mismatch')
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new evaluation output')
    out.mkdir(parents=True);stage=state['contract']['stage'];records=[]
    for i,item in enumerate(selection['validation'][:args.limit]):
        clip=dataset.window(*dataset.locate(item))[None].to(args.device)
        record=dict(index=i,selection=item)
        if stage=='tokenizer':
            with torch.no_grad():pred,_,_=model.tokenizer(clip)
            record.update(mse=float((pred-clip).square().mean()))
            np.savez_compressed(out/f'reconstruction-{i:03}.npz',source=clip.cpu().numpy(),prediction=pred.cpu().numpy())
        else:
            record['oracle_diagnostic']=oracle_interventions(model,clip)
            # Last observed transition supplies an arbitrary reference code; no semantic claim.
            with torch.no_grad():_,code,_=model.lam.infer(clip[:,-2:])
            record['commands']=[]
            for label,command in alternative_codes(code,model.config.action_codes):
                generated=commanded_rollout(model,clip,command,args.frames,stage)
                path=out/f'rollout-{i:03}-{label}.npz';np.savez_compressed(path,frames=generated.cpu().numpy())
                record['commands'].append(dict(label=label,codes=command.flatten().tolist(),path=path.name,sha256=file_hash(path)))
        records.append(record)
    report=dict(checkpoint_sha256=file_hash(args.checkpoint),binding_sha256=dataset.binding_sha256,
        selection_sha256=file_hash(root/spec['evaluation_selection']),stage=stage,preflight=state['contract']['preflight'],
        quality_gate='unmeasured',semantic_mapping='unmeasured; calibrate on training footage only',records=records)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(records),stage=stage,quality_gate='unmeasured')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--root',default='.');p.add_argument('--spec',default='configs/training-baseline-v1.json')
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,default=64);p.add_argument('--frames',type=int,default=10)
    run(p.parse_args())
