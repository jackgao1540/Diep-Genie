import torch
import pytest
from genie_diep.model import Config, WorldModel, image_loss, masked_dynamics_loss
from genie_diep.synthetic import batch
from genie_diep.train import initialize, save_checkpoint, load_checkpoint

torch.set_num_threads(2)

@pytest.fixture(params=[4,8])
def setup(request):
    torch.manual_seed(7)
    model=WorldModel(Config(size=32,width=32,context=4,action_codes=(9,8,2),lam_variant="delta",downsample=request.param))
    x=batch(2,4,32)
    initialize(model,x)
    return model,x


def test_tokenizer_future_does_not_change_past(setup):
    m,x=setup; y=x.clone(); y[:,2:]=torch.rand_like(y[:,2:])
    torch.testing.assert_close(m.tokenizer.features(x)[:,:2],m.tokenizer.features(y)[:,:2],rtol=0,atol=0)


def test_dynamics_future_tokens_and_actions_do_not_change_past(setup):
    m,x=setup
    _,ids,_=m.tokenizer.encode(x); actions,_,_=m.lam.infer(x)
    altered=ids.clone(); altered[:,2:]=(altered[:,2:]+1)%m.config.vocab
    action2=actions.clone(); action2[:,1:]+=10
    a=m.dynamics(ids,actions)[:,:2]; b=m.dynamics(altered,action2)[:,:2]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_backward_all_components_and_discrete_action_bottleneck(setup):
    m,x=setup
    rec,ids,tvq=m.tokenizer(x)
    pred,act,aid,avq=m.lam(x)
    loss=image_loss(rec,x)+tvq+image_loss(pred,x[:,1:],x[:,:-1])+avq+masked_dynamics_loss(m.dynamics,ids,act)
    loss.backward()
    for module in [m.tokenizer,m.lam,m.dynamics]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads)
    torch.testing.assert_close(act,m.lam.lookup(aid),atol=1e-6,rtol=1e-5)


def test_dynamics_cannot_backpropagate_into_action_encoder(setup):
    m,x=setup
    _,ids,_=m.tokenizer.encode(x); act,_,_=m.lam.infer(x)
    masked_dynamics_loss(m.dynamics,ids,act).backward()
    assert all(p.grad is None for p in m.lam.parameters())


def test_checkpoint_roundtrip_and_rollout(setup,tmp_path):
    m,x=setup; opt=torch.optim.AdamW(m.parameters())
    rec,_,loss=m.tokenizer(x); (image_loss(rec,x)+loss).backward(); opt.step()
    save_checkpoint(tmp_path/"model.pt",m,opt,1,"tokenizer")
    restored,state=load_checkpoint(tmp_path/"model.pt")
    torch.testing.assert_close(m.tokenizer(x)[0],restored.tokenizer(x)[0],rtol=0,atol=0)
    assert state["step"]==1 and state["optimizer"]["state"]
    _,ids,_=m.tokenizer.encode(x)
    acts,_,_=m.lam.infer(x)
    nxt=restored.dynamics.next_tokens(ids[:,:-1],acts)
    grid=m.config.size//m.config.downsample
    assert nxt.shape==(2,1,grid,grid) and nxt.min()>=0 and nxt.max()<m.config.vocab


def test_resume_optimizer_matches_next_update(setup,tmp_path):
    m,x=setup; opt=torch.optim.AdamW(m.parameters(),lr=.001)
    def update(model,optimizer):
        optimizer.zero_grad(); rec,_,vq=model.tokenizer(x)
        (image_loss(rec,x)+vq).backward(); optimizer.step()
    update(m,opt)
    save_checkpoint(tmp_path/"resume.pt",m,opt,1,"tokenizer")
    restored,state=load_checkpoint(tmp_path/"resume.pt")
    opt2=torch.optim.AdamW(restored.parameters(),lr=.001); opt2.load_state_dict(state["optimizer"])
    update(m,opt); update(restored,opt2)
    for p,q in zip(m.parameters(),restored.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
