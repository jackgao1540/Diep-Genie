import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from genie_diep.cohort import Cohort, MixedSampler
from genie_diep.data import file_hash
from genie_diep.baseline import Trainer, Budget, dynamics_mask
from genie_diep.baseline_evaluation import alternative_codes, commanded_rollout, constant_velocity, gate_g1
from genie_diep.model import Config, WorldModel

torch.set_num_threads(2)


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def project(tmp_path):
    data=tmp_path/'data';data.mkdir();records=[];sequences=[]
    for source,split,count in [('a','train',2),('b','train',1),('v','val',1),('t','test',1)]:
        blocks=[]
        for i in range(count):
            name=f'{source}-{i}.npz';frames=np.full((40,128,128,3),20+i*40,dtype=np.uint8)
            frames[:,:,40:60,1]=120;np.savez_compressed(data/name,frames=frames)
            row=dict(source_id=source,block_id=str(i),split=split,video_sha256=source,crop=[0,0,128,128],fps=10,source_fps=30,
                start=i*4,end=(i+1)*4,frames=40,source_frame_indices=list(range(i*120,(i+1)*120,3)),proxy_path=name,
                proxy_sha256=file_hash(data/name),curation_accepted=True,reasons=[],training_approved=False)
            records.append(row);blocks.append(str(i))
        sequences.append(dict(source_id=source,split=split,blocks=blocks))
    (data/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));write(data/'sequences.json',sequences);write(data/'summary.json',{})
    binding=dict(cohort='data',hashes={n:file_hash(data/n) for n in ['manifest.jsonl','summary.json','sequences.json']});write(tmp_path/'binding.json',binding)
    val=Cohort(tmp_path,'binding.json','val',3)
    write(tmp_path/'evaluation.json',dict(binding_sha256=file_hash(tmp_path/'binding.json'),validation=[val.descriptor(0,0)]))
    model=WorldModel(Config(size=128,width=8,token_dim=4,vocab=8,action_dim=2,action_codes=(3,2),layers=1,heads=2,context=3,downsample=8))
    spec=dict(model=model.config_dict(),data_binding='binding.json',evaluation_selection='evaluation.json',local_execution_ready=False,
        data=dict(source_balanced_sampling_fraction=.5),optimization=dict(seed=42,weight_decay=.01,gradient_clip_norm=1),
        stages=[dict(name=s,max_updates=20,max_allocated_hours=1,max_compute_units=2,learning_rate=.001,microbatch_sequences=1,gradient_accumulation=2,color_balance=True,fully_mask_final_frame_fraction=.5) for s in ['tokenizer','actions','dynamics']])
    write(tmp_path/'spec.json',spec)
    return tmp_path


def test_checked_windows_cross_only_real_joins_and_never_test(project):
    c=Cohort(project,'binding.json','train',16);x=c.window(0,35)
    assert x.shape==(16,3,128,128) and float(x[0,0,0,0])==pytest.approx(20/255) and float(x[5,0,0,0])==pytest.approx(60/255)
    with pytest.raises(ValueError,match='Reserved test'):Cohort(project,'binding.json','test')
    with pytest.raises(ValueError,match='Split decoding'):c._frames(('t','0'))
    path=project/'data/sequences.json';sequences=json.loads(path.read_text());sequences[0]['blocks']=['1','0'];write(path,sequences)
    binding=json.loads((project/'binding.json').read_text());binding['hashes']['sequences.json']=file_hash(path);write(project/'binding.json',binding)
    with pytest.raises(ValueError,match='bridges a gap'):Cohort(project,'binding.json','train')


def test_sampler_distribution_and_state_roundtrip(project):
    c=Cohort(project,'binding.json','train',16);s=MixedSampler(c,7);a=sum(s.draw()[0]==0 for _ in range(6000))/6000
    # source a: half * 1/2 + half * 65/(65+25)
    assert a==pytest.approx(.25+.5*65/90,abs=.025)
    state=s.state_dict();expected=[s.draw() for _ in range(12)];s.load_state_dict(state);assert [s.draw() for _ in range(12)]==expected


@pytest.mark.parametrize('stage',['tokenizer','actions','dynamics'])
def test_full_resume_reproduces_next_update_and_freezes_other_stages(project,stage):
    a=Trainer(project,'spec.json',stage,preflight=True);before={n:p.detach().clone() for n,p in a.model.named_parameters()}
    assert np.isfinite(a.update());a.save(project/'checkpoint.pt');loss=a.update();expected={n:p.detach().clone() for n,p in a.model.named_parameters()}
    b=Trainer(project,'spec.json',stage,preflight=True,resume=project/'checkpoint.pt');assert b.update()==loss
    for n,p in b.model.named_parameters():
        torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
        prefix={'tokenizer':'tokenizer.','actions':'lam.','dynamics':'dynamics.'}[stage]
        if not n.startswith(prefix):torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    assert a.scheduler.state_dict()==b.scheduler.state_dict() and a.sampler.draws==b.sampler.draws
    with pytest.raises(ValueError,match='contract changed'):Trainer(project,'spec.json',stage,preflight=True,resume=project/'checkpoint.pt',accumulation=1)


def test_production_requires_readiness_and_stage_promotion(project):
    with pytest.raises(ValueError,match='readiness'):Trainer(project,'spec.json','tokenizer')
    spec=json.loads((project/'spec.json').read_text());spec['local_execution_ready']=True;write(project/'spec.json',spec)
    from genie_diep.baseline import implementation_hash
    spec['readiness']='readiness.json';write(project/'spec.json',spec)
    write(project/'readiness.json',dict(ready_for_gpu_pilot=True,implementation_sha256=implementation_hash(),spec_sha256=file_hash(project/'spec.json'),binding_sha256=file_hash(project/'binding.json')))
    with pytest.raises(ValueError,match='Prior-stage'):Trainer(project,'spec.json','actions')


def test_masks_and_all_codebooks_are_exercised():
    ids=torch.zeros((1,3,4,4),dtype=torch.long)
    final=dynamics_mask(ids,1);assert final[:,-1].all() and not final[:,:-1].any()
    random=dynamics_mask(ids,0);assert not random[:,0].any() and random[:,-1,0,0].all()
    code=torch.tensor([[[2,1,0]]]);alternatives=alternative_codes(code,[9,8,2]);assert len(alternatives)==17
    assert all((c!=code).sum()==1 for _,c in alternatives[1:])


def test_budget_accounts_for_setup_prior_usage_and_monthly_reserve(monkeypatch):
    monkeypatch.setattr('genie_diep.baseline.time.monotonic',lambda:100)
    stage=dict(max_allocated_hours=1,max_compute_units=2)
    budget=Budget(stage,prior_seconds=1000,prior_units=1,rate=2,balance=100,setup_seconds=600,reserve_seconds=120)
    assert budget.remaining()==pytest.approx(1080)
    assert not Budget(stage,rate=2,balance=50.01,setup_seconds=60,reserve_seconds=120).permits()


def test_rollout_uses_only_prefix_and_explicit_commands():
    model=WorldModel(Config(size=32,width=8,token_dim=4,vocab=8,action_dim=2,action_codes=(3,2),layers=1,heads=2,context=3,downsample=8)).eval()
    prefix=torch.rand(1,3,3,32,32);code=torch.tensor([[[0,1]]])
    a=commanded_rollout(model,prefix,code,2);b=commanded_rollout(model,prefix,code,2)
    torch.testing.assert_close(a,b,rtol=0,atol=0);assert a.shape==(1,2,3,32,32)
    assert torch.isfinite(constant_velocity(prefix)).all()
    assert gate_g1([], {})['status']=='unmeasured'
    assert gate_g1([dict(player_retained=True)],{})['status']=='unmeasured'
