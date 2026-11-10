import copy
import pytest
import torch

from genie_diep.model import Config, WorldModel
from scripts.evaluate_tokenizer_quality import aggregate, metric_means, metric_totals, reconstruct, validate_state


def test_temporal_chunks_match_full_causal_reconstruction_without_mutation():
    torch.set_num_threads(2); torch.manual_seed(19)
    model=WorldModel(Config(size=16,width=8,token_dim=4,vocab=8,action_dim=2,action_codes=(3,2),layers=1,heads=2,context=16,downsample=4)).eval()
    clip=torch.rand(16,3,16,16)
    before=copy.deepcopy(model.state_dict())
    with torch.inference_mode():
        full,expected_ids,_=model.tokenizer(clip[None])
    chunked,ids=reconstruct(model.tokenizer,clip,4)
    torch.testing.assert_close(chunked,full[0],rtol=1e-6,atol=1e-7)
    assert torch.equal(ids,expected_ids[0])
    for key,value in model.state_dict().items():
        assert torch.equal(value,before[key])
    model.train()
    with pytest.raises(ValueError,match='eval mode'):
        reconstruct(model.tokenizer,clip)


def test_foreground_and_temporal_metrics_do_not_hide_missing_colored_objects():
    source=torch.full((3,3,4,4),.5)
    source[:,0,1,1]=1; source[:,1:,1,1]=0
    prediction=torch.full_like(source,.5)
    means=metric_means(metric_totals(source,prediction))
    assert means['foreground_mse']==pytest.approx(.25)
    assert means['mse']==pytest.approx(.25/16)
    assert means['temporal_difference_mse']==0
    source[1,0,1,1]=.75
    assert metric_means(metric_totals(source,prediction))['temporal_difference_mse']>0
    empty=metric_means(metric_totals(prediction,prediction))
    assert empty['foreground_mse'] is None


def test_equal_source_aggregation_does_not_weight_longer_sources():
    def row(source,value):
        totals={key+suffix: (value if suffix=='_sum' else 1) for key in ('mse','foreground_mse','temporal_difference_mse') for suffix in ('_sum','_count')}
        return dict(selection=dict(source_id=source),after=totals)
    result=aggregate([row('a',1),row('a',1),row('b',4)],'after')
    assert result['pooled']['mse']==2
    assert result['equal_source']['mse']==2.5


@pytest.mark.parametrize('field,value,error',[
    ('stage','actions','tokenizer'),('binding_sha256','wrong','binding'),('evaluation_sha256','wrong','selection')])
def test_checkpoint_contract_rejects_wrong_source_or_stage(field,value,error):
    state=dict(contract=dict(stage='tokenizer',binding_sha256='b',evaluation_sha256='e'),config=dict(action_codes=(9,8,2)),model={'x':torch.ones(2)})
    spec=dict(model=dict(action_codes=[9,8,2]))
    validate_state(state,spec,'b','e')
    state['contract'][field]=value
    with pytest.raises(ValueError,match=error):
        validate_state(state,spec,'b','e')


def test_checkpoint_rejects_architecture_and_nonfinite_weights():
    state=dict(contract=dict(stage='tokenizer',binding_sha256='b',evaluation_sha256='e'),config={'width':8},model={'x':torch.ones(2)})
    with pytest.raises(ValueError,match='architecture'):
        validate_state(state,dict(model={'width':16}),'b','e')
    state['model']['x'][0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        validate_state(state,dict(model={'width':8}),'b','e')
