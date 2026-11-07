import torch
from genie_diep.model import image_loss


def test_balanced_color_loss_increases_signal_on_small_blue_object():
    target=torch.full((1,1,3,32,32),.8)
    target[:,:,0,3:20,3:20]=.1
    target[:,:,1,3:20,3:20]=.65
    target[:,:,2,3:20,3:20]=.85
    target[:,:,:,24:28,24:28]=torch.tensor([.25,.3,.9]).view(1,1,3,1,1)
    predicted=target.clone();predicted[:,:,1,24:28,24:28]=.7
    predicted.requires_grad_()
    regular=image_loss(predicted,target)
    regular_grad=torch.autograd.grad(regular,predicted,retain_graph=True)[0]
    balanced=image_loss(predicted,target,color_balance=True)
    balanced_grad=torch.autograd.grad(balanced,predicted)[0]
    assert balanced>regular
    assert balanced_grad[:,:,1,24:28,24:28].abs().mean()>2*regular_grad[:,:,1,24:28,24:28].abs().mean()


def test_balanced_loss_empty_foreground_is_finite():
    target=torch.full((1,2,3,16,16),.7);prediction=torch.full_like(target,.6,requires_grad=True)
    loss=image_loss(prediction,target,color_balance=True);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
