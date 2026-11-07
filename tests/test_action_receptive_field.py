import torch
from genie_diep.model import Config,LatentActionModel


def test_wider_decoder_can_access_an_object_eight_pixels_away():
    torch.manual_seed(4)
    local=LatentActionModel(Config(size=32,width=16,action_dim=4,action_codes=(4,)))
    wide=LatentActionModel(Config(size=32,width=16,action_dim=4,action_codes=(4,),lam_dilations=(1,3,6)))
    wide.load_state_dict(local.state_dict())
    current=torch.full((1,1,3,32,32),.5,requires_grad=True)
    action=torch.zeros(1,1,4)
    local_output=local.reconstruct(current,action)[0,0,:,16,16].sum()
    wide_output=wide.reconstruct(current,action)[0,0,:,16,16].sum()
    local_grad=torch.autograd.grad(local_output,current,retain_graph=True)[0]
    wide_grad=torch.autograd.grad(wide_output,current)[0]
    assert local_grad[0,0,:,16,8].abs().sum()==0
    assert wide_grad[0,0,:,16,8].abs().sum()>0
