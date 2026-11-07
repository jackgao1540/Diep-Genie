import torch
from genie_diep.model import VQ


def test_cosine_regularization_cannot_be_evaded_by_feature_magnitude():
    torch.manual_seed(21)
    book=VQ(8,4)
    # All samples have the same direction but different radii.
    x=torch.tensor([[1.,2.,3.,4.]])*torch.linspace(.01,.5,12)[:,None]
    q,ids,loss=book(x)
    scaled_q,scaled_ids,scaled_loss=book(x*100)
    torch.testing.assert_close(q,scaled_q)
    torch.testing.assert_close(ids,scaled_ids)
    torch.testing.assert_close(loss,scaled_loss)
