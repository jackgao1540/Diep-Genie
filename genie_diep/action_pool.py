"""Representative-pool action code management; training clips only, no action labels."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def collect_features(lam,dataset,device,limit=64,batch_size=4):
    indices=dataset.representative_indices(limit)
    features=[]
    for offset in range(0,len(indices),batch_size):
        x=torch.stack([dataset.get_clip(i,start=0) for i in indices[offset:offset+batch_size]]).to(device)
        features.append(lam.features(x).flatten(0,1))
    return torch.cat(features),indices


@torch.no_grad()
def initialize_book(book,features,iterations=10):
    """Deterministic farthest-point seeds followed by spherical k-means."""
    z=F.normalize(features.reshape(-1,features.shape[-1]),dim=-1)
    centers=[z[0]]
    nearest=z@centers[0]
    for _ in range(1,len(book.embedding.weight)):
        centers.append(z[nearest.argmin()])
        nearest=torch.maximum(nearest,z@centers[-1])
    centers=torch.stack(centers)
    for _ in range(iterations):
        assignments=(z@centers.T).argmax(-1)
        for i in range(len(centers)):
            members=z[assignments==i]
            if len(members):centers[i]=F.normalize(members.mean(0),dim=-1)
    book.embedding.weight.copy_(centers)


@torch.no_grad()
def update_books(lam,features,optimizer=None,initialize=False):
    reports=[]
    for i,book in enumerate(lam.books):
        z=features[:,i]
        if initialize:
            initialize_book(book,z)
            changed=list(range(len(book.embedding.weight)))
        else:
            changed=book.revive_unused(z)
        if optimizer is not None and changed:
            state=optimizer.state.get(book.embedding.weight,{})
            for key in ("exp_avg","exp_avg_sq","max_exp_avg_sq"):
                if key in state:state[key][changed]=0
        ids=(F.normalize(z,dim=-1)@F.normalize(book.embedding.weight,dim=-1).T).argmax(-1)
        counts=torch.bincount(ids,minlength=len(book.embedding.weight))
        reports.append(dict(factor=i,changed=changed,counts=counts.tolist()))
    return reports
