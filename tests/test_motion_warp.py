import torch
from genie_diep.model import Config,LatentActionModel,WorldModel,translate_frames
from genie_diep.train import load_checkpoint,save_checkpoint


def test_translation_sign_identity_and_gradient():
    image=torch.full((1,1,3,16,16),.1)
    image[:,:,:,5:9,4:8]=.9
    assert torch.allclose(translate_frames(image,torch.zeros(1,1,2)),image,atol=1e-6)
    shift=torch.tensor([[[3.,-2.]]],requires_grad=True)
    moved=translate_frames(image,shift)
    assert torch.allclose(moved[:,:,:,3:7,7:11],torch.full((1,1,3,4,4),.9),atol=1e-6)
    loss=(moved-image).square().mean();loss.backward()
    assert torch.isfinite(shift.grad).all() and shift.grad.abs().sum()>0


def test_disabled_branch_preserves_old_state_and_enabling_starts_at_identity():
    torch.manual_seed(3)
    model=LatentActionModel(Config(width=8,action_dim=4,action_codes=(4,)))
    current=torch.rand(2,1,3,16,16)*.8+.1;actions=torch.randn(2,1,4)
    original=model.reconstruct(current,actions)
    assert not any('motion_head' in k for k in model.state_dict())
    rng=torch.get_rng_state().clone()
    model.enable_motion_warp()
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.allclose(original,model.reconstruct(current,actions),atol=2e-6)
    with torch.no_grad():model.motion_head.weight[0,0]=1
    actions.zero_();actions[0,0,0]=1;actions[1,0,0]=-1
    shifts=model.motion_shifts(actions,16,16)
    assert shifts[0,0,0]>0 and shifts[1,0,0]<0
    assert shifts.abs().max()<2


def test_motion_checkpoint_roundtrip(tmp_path):
    model=WorldModel(Config(size=16,width=8,heads=2,layers=1,action_dim=4,
                            action_codes=(4,),lam_motion_warp=True))
    opt=torch.optim.AdamW(model.parameters())
    with torch.no_grad():model.lam.motion_head.weight.fill_(.2)
    path=tmp_path/'motion.pt';save_checkpoint(path,model,opt,4,'actions')
    restored,state=load_checkpoint(path)
    assert restored.config.lam_motion_warp and state['step']==4
    assert torch.equal(restored.lam.motion_head.weight,model.lam.motion_head.weight)
