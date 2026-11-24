import copy

import pytest
import torch

from genie_diep.baseline import Budget
from scripts import continue_staged_tokenizer_fp32 as extension
from scripts import recover_staged_tokenizer_fp32 as proven
from scripts.run_tokenizer_comparison import compare_exact,source_hash
from test_staged_tokenizer_runner import staged
from test_baseline_pipeline import project


def setup(staged):
    path=staged/'extension.json';path.write_text('{}')
    manifest=dict(parent_checkpoint_sha256='parent',historical_manifest_sha256='old',
        historical_proof_sha256='proof',proven_source_sha256=source_hash(proven.__file__),failed_proof_sha256='failed')
    context=extension.context_for(path,manifest,staged/'extension-events')
    return path,manifest,context


def allocation():return Budget(dict(max_allocated_hours=1,max_compute_units=1),reserve_seconds=0)


def test_only_review_allowance_changes():
    assert extension.POLICY['maximum_joint_events']==12 and proven.POLICY['maximum_joint_events']==4
    assert {k:v for k,v in extension.POLICY.items() if k!='maximum_joint_events'}=={
        k:v for k,v in proven.POLICY.items() if k!='maximum_joint_events'}


def test_adapter_preserves_all_numeric_state_events_and_initial_contract(staged):
    path,manifest,context=setup(staged)
    trainer=proven.ExceptionTrainer(staged,'plan.json','joint',local_check=True,mode='full',
        context=dict(manifest_sha256='old',failed_proof_sha256='failed',out=str(staged/'old-events')))
    trainer.save(staged/'initial.pt');trainer.update();trainer.fp32_events=[dict(attempted_update=n) for n in [1804,1805,2902,3453]]
    trainer.save(staged/'parent.pt');state=torch.load(staged/'parent.pt',weights_only=True);before=copy.deepcopy(state)
    initial=torch.load(staged/'initial.pt',weights_only=True)
    adapted=extension.adapt_state(state,path,manifest);adapted_initial=extension.adapt_state(initial,path,manifest)
    assert adapted['contract']==adapted_initial['contract']
    for key in state:
        if key!='contract':compare_exact(adapted[key],state[key])
    compare_exact(state,before)
    assert len(adapted['fp32_exception_events'])==4 and adapted_initial['fp32_exception_events']==[]
    with pytest.raises(ValueError,match='cannot migrate again'):extension.adapt_state(adapted,path,manifest)


def test_new_loader_accepts_four_preserves_history_and_replays_positionally(staged):
    path,manifest,context=setup(staged)
    trainer=extension.ExtensionTrainer(staged,'plan.json','joint',local_check=True,mode='full',context=context)
    trainer.fp32_events=[dict(old=i) for i in range(4)];trainer.update();trainer.save(staged/'before.pt')
    expected_loss=trainer.update();expected=trainer._update_boundary()
    resumed=extension.ExtensionTrainer(staged,'plan.json','joint','cpu',4,None,staged/'before.pt',True,'full',context=context)
    assert resumed.update()==expected_loss;compare_exact(resumed._update_boundary(),expected)
    assert resumed.fp32_events==[dict(old=i) for i in range(4)]
    assert proven.POLICY['maximum_joint_events']==4


def test_twelfth_event_is_retained_then_next_update_pauses_before_sampling(staged):
    _,_,context=setup(staged)
    trainer=extension.ExtensionTrainer(staged,'plan.json','joint',local_check=True,context=context)
    trainer.fp32_events=[dict(old=i) for i in range(11)]
    assert trainer.fp32_update(allocation(),dict(kind='local_test')) is not None
    assert len(trainer.fp32_events)==12;before=trainer._update_boundary()
    with pytest.raises(ValueError,match='cap reached'):trainer.update(allocation())
    compare_exact(trainer._update_boundary(),before)
    assert proven.POLICY['maximum_joint_events']==4


def test_policy_scope_restores_sealed_policy_even_after_failure():
    original=proven.POLICY
    with pytest.raises(RuntimeError):
        with extension.policy_scope():
            assert proven.POLICY['maximum_joint_events']==12
            raise RuntimeError('test')
    assert proven.POLICY is original


def test_stageB_precision_and_frozen_handoff_unchanged(staged):
    _,_,context=setup(staged)
    joint=extension.ExtensionTrainer(staged,'plan.json','joint',local_check=True,context=context)
    joint.update();joint.save(staged/'joint.pt')
    decoder=extension.ExtensionTrainer(staged,'plan.json','decoder_only',handoff=staged/'joint.pt',local_check=True,context=context)
    assert decoder.scaler_initial_scale==decoder.scaler_max_scale==1024
    decoder.update();decoder.verify_inactive();assert decoder.fp32_events==[]
    with pytest.raises(ValueError,match='Only joint'):decoder.fp32_update(allocation(),{})


@pytest.mark.parametrize('change', ['unrelated_step','wrong_parent_allowance','changed_cap','changed_source'])
def test_manifest_rejects_unrelated_or_reset_policies(staged,change):
    from genie_diep.data import file_hash
    manifest=dict(schema_version=1,version=extension.VERSION,policy=copy.deepcopy(extension.POLICY),
        source_sha256=source_hash(extension.__file__),proven_source_sha256=source_hash(proven.__file__),
        support_source_sha256=source_hash(proven.prior.__file__),accepted_step=3453,
        inherited_event_updates=[1804,1805,2902,3453],old_maximum_joint_events=4,new_maximum_joint_events=12,
        plan='plan.json',original_plan_sha256=file_hash(staged/'plan.json'))
    path=staged/'strict-manifest.json';extension.write_json(path,manifest)
    assert extension.validate_manifest(staged,path)==manifest
    if change=='unrelated_step':manifest['accepted_step']=1803
    if change=='wrong_parent_allowance':manifest['old_maximum_joint_events']=12
    if change=='changed_cap':manifest['policy']['maximum_joint_events']=20
    if change=='changed_source':manifest['source_sha256']='other'
    extension.write_json(path,manifest)
    with pytest.raises(ValueError):extension.validate_manifest(staged,path)
