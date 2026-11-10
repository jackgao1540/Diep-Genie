import pytest
import torch

from scripts.validate_gpu_recovery_artifacts import finite_tensors, validate_report


@pytest.fixture
def evidence():
    members={name:name for name in ['scripts/gpu_recovery_verification.py','genie_diep/gpu_precision.py',
        'genie_diep/model.py','scripts/diagnose_tokenizer_precision.py','recovery-inputs/original-spec.json']}
    manifest=dict(member_hashes=members,**{'expected_'+name+'_sha256':name for name in ['checkpoint','implementation','spec','model']})
    cases=[]
    for dtype,scale in [('fp32',1),('fp16',65536),('fp16',1024)]:
        cases.append(dict(precision=dtype,scale=scale,model_state_unchanged=True,optimizer_updates=0,status='finite',
                          microbatches=[dict(finite_loss=True,loss=1.,nonfinite_gradient_parameters=[],gradients=[dict(nonfinite=0)]) for _ in range(4)]))
    windows=[dict(index=i) for i in range(16)]
    report=dict(status='passed',preflight=True,promotion_approved=False,quality_gates='unmeasured',
        **{name+'_sha256':name for name in ['checkpoint','implementation','spec','model']},
        harness_sha256=members['scripts/gpu_recovery_verification.py'],source_step=219,final_step=518,
        maximum_optimizer_calls=300,optimizer_calls=300,maximum_compute_units=2,maximum_allocated_hours=1,
        minimum_remaining_units=50,spent_seconds=300,estimated_units=.25,
        sustained=dict(unique_additional_steps=299,final_step=518,measured_updates=298),
        recovery=dict(status='passed',rollback_exact=True,max_absolute_state_difference=0.,reference_loss=1.,replayed_loss=1.,kind='exact'),
        evaluation=dict(complete=True,samples=64,sources=16,step=500,stage='tokenizer',preflight=True,quality_gate='unmeasured',
                        pooled=.1,equal_source=.1,by_source={str(i):.1 for i in range(16)}),
        precision=dict(status='passed',complete=True,optimizer_updates=0,checkpoint_file_unchanged=True,
            failed_samples_recovered_exactly=True,checkpoint_sha256='checkpoint',diagnostic_source_sha256=members['genie_diep/gpu_precision.py'],
            model_source_sha256=members['genie_diep/model.py'],recovery_helper_sha256=members['scripts/diagnose_tokenizer_precision.py'],
            specification_sha256=members['recovery-inputs/original-spec.json'],cases=cases,fp32_finite=True,conservative_fp16_finite=True,samples=windows),
        migration=dict(kind='explicit_diagnostic_fork',preflight=True,promotion_approved=False,source_step=219,failed_update=220,
                       source_sampler_draws=3521,recovered_sampler_draws=3505,recovered_windows=windows,initial_scale=1024,maximum_scale=1024,scaler_reset=True),
        injected_failure=dict(restored_pre_update_boundary=True,attempted_step=220,phase='gradient_check',exception='FloatingPointError',
                              sampler_draws_before=3505,sampler_draws_at_failure=3521,sampled_windows=windows))
    return report,manifest


def test_high_scale_may_be_finite_without_invalidating_lower_scale_evidence(evidence):
    report,manifest=evidence
    validate_report(report,manifest)
    report['precision']['cases'][1]['status']='nonfinite'
    report['precision']['cases'][1]['microbatches'][0]['gradients'][0]['nonfinite']=1
    validate_report(report,manifest)


@pytest.mark.parametrize('change,match',[
    (lambda r:r.update(optimizer_calls=301),'Optimizer-call'),
    (lambda r:r.update(estimated_units=2.1),'exceeds budget'),
    (lambda r:r['evaluation'].update(step=518),'evaluation'),
    (lambda r:r['recovery'].update(max_absolute_state_difference=1e-8),'precision claim'),
    (lambda r:r['precision']['cases'][2].update(status='nonfinite'),'conservative'),
    (lambda r:r.update(preflight=False),'Diagnostic-only'),
    (lambda r:r.update(harness_sha256='different'),'Harness identity'),
])
def test_inconsistent_pass_reports_are_rejected(evidence,change,match):
    report,manifest=evidence;change(report)
    with pytest.raises(ValueError,match=match):validate_report(report,manifest)


def test_optimizer_nonfinite_is_detected_independently_of_model():
    state=dict(model=dict(weight=torch.ones(2)),optimizer=dict(state={0:dict(exp_avg=torch.tensor([float('nan')]))}))
    with pytest.raises(ValueError,match='optimizer.state.0.exp_avg'):finite_tensors(state)
