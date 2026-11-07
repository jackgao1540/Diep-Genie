"""Small, explicit deviations from Genie: conv tokenizer/LAM, ST masked dynamics.

All time-dependent paths are causal. The LAM may see the target only while inferring
a discrete training label. The dynamics receives its detached quantized code only.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    size: int = 64
    width: int = 64
    token_dim: int = 32
    vocab: int = 128
    action_dim: int = 16
    action_codes: tuple = (16,)
    layers: int = 2
    heads: int = 4
    context: int = 5
    lam_variant: str = "pair"
    downsample: int = 8


class VQ(nn.Module):
    """Cosine codebook with straight-through gradients and optional data initialization."""
    def __init__(self, count, dim):
        super().__init__()
        self.embedding = nn.Embedding(count, dim)
        nn.init.normal_(self.embedding.weight)

    @torch.no_grad()
    def initialize(self, x):
        flat = x.reshape(-1,x.shape[-1])
        picks = torch.arange(len(self.embedding.weight),device=x.device) % len(flat)
        self.embedding.weight.copy_(F.normalize(flat[picks] + torch.randn_like(flat[picks])*.01,dim=-1))

    def forward(self, x):
        z = F.normalize(x, dim=-1)
        code = F.normalize(self.embedding.weight, dim=-1)
        scores = z @ code.T
        ids = scores.argmax(-1)
        q = F.embedding(ids,code)
        loss = F.mse_loss(q,z.detach()) + .25*F.mse_loss(z,q.detach())
        # Mild aggregate diversity pressure; does not prove semantic action diversity.
        p = (scores/.15).softmax(-1).reshape(-1,len(code)).mean(0)
        balance = (p * (p.clamp_min(1e-8).log()+math.log(len(code)))).sum()
        # Discourage both identical features and unused codebooks. Quantization error
        # is a regularizer, not the main task: over-weighting it creates constant codes.
        entropy=-(scores/.15).softmax(-1).mul((scores/.15).log_softmax(-1)).sum(-1).mean()
        # Quantization uses direction, so magnitude variation must not satisfy
        # the anti-collapse penalty. Otherwise radial features evade the check.
        flat=z.reshape(-1,z.shape[-1])
        spread=F.relu(.1-flat.std(0,unbiased=False)).mean()
        return z + (q-z).detach(), ids, .1*loss + .1*balance + .01*entropy + spread

    def lookup(self, ids):
        return F.embedding(ids,F.normalize(self.embedding.weight,dim=-1))

    @torch.no_grad()
    def revive_unused(self,x):
        """For a representative pool only: seed unused codes at uncovered features.

        Do not apply to arbitrary small minibatches where absence is not deadness.
        Returns replaced indices so an optimizer's row state can be cleared.
        """
        z=F.normalize(x.reshape(-1,x.shape[-1]),dim=-1)
        code=F.normalize(self.embedding.weight,dim=-1)
        ids=(z@code.T).argmax(-1)
        used=ids.unique(); missing=sorted(set(range(len(code)))-set(used.tolist()))
        centers=code[used]
        for index in missing:
            candidate=(z@centers.T).max(-1).values.argmin()
            self.embedding.weight[index].copy_(z[candidate])
            centers=torch.cat([centers,z[candidate:candidate+1]],0)
        return missing


class CausalConv(nn.Module):
    def __init__(self, cin, cout, spatial_stride=1):
        super().__init__()
        self.conv = nn.Conv3d(cin,cout,(3,3,3),stride=(1,spatial_stride,spatial_stride),padding=(0,1,1))

    def forward(self,x):
        return self.conv(F.pad(x,(0,0,0,0,2,0)))


class VideoTokenizer(nn.Module):
    def __init__(self,c):
        super().__init__()
        if c.downsample not in (4,8): raise ValueError("Tokenizer downsample must be 4 or 8")
        self.encoder=nn.Sequential(CausalConv(3,c.width//2,2),nn.SiLU(),
                                   CausalConv(c.width//2,c.width,2),nn.SiLU(),
                                   CausalConv(c.width,c.token_dim,c.downsample//4))
        self.vq=VQ(c.vocab,c.token_dim)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(c.token_dim,c.width,4,2,1),nn.SiLU(),
                                  nn.ConvTranspose2d(c.width,c.width//2,4,2,1),nn.SiLU(),
                                  nn.ConvTranspose2d(c.width//2,3,4,2,1),nn.Sigmoid())
        if c.downsample==4:
            self.decoder=nn.Sequential(nn.ConvTranspose2d(c.token_dim,c.width,4,2,1),nn.SiLU(),
                                      nn.ConvTranspose2d(c.width,c.width//2,4,2,1),nn.SiLU(),
                                      nn.Conv2d(c.width//2,3,3,1,1),nn.Sigmoid())

    def features(self,x):
        return self.encoder(x.transpose(1,2)).permute(0,2,3,4,1)

    def encode(self,x):
        return self.vq(self.features(x))

    def decode(self,q):
        b,t,h,w,d=q.shape
        result=self.decoder(q.reshape(b*t,h,w,d).permute(0,3,1,2))
        return result.reshape(b,t,3,result.shape[-2],result.shape[-1])

    def decode_ids(self,ids):
        return self.decode(self.vq.lookup(ids))

    def forward(self,x):
        q,ids,loss=self.encode(x)
        return self.decode(q),ids,loss


class LatentActionModel(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        d=c.action_dim*len(c.action_codes)
        self.encoder=nn.Sequential(nn.Conv2d(6,c.width,5,2,2),nn.SiLU(),
                                   nn.Conv2d(c.width,c.width,3,2,1),nn.SiLU(),
                                   nn.Conv2d(c.width,c.width,3,2,1),nn.SiLU(),
                                   nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(c.width*4,d))
        if c.lam_variant=="delta":
            self.encoder=nn.Sequential(nn.Conv2d(9,c.width,5,1,2),nn.SiLU(),
                                       nn.Conv2d(c.width,c.width,5,1,2),nn.SiLU(),
                                       nn.Conv2d(c.width,c.width,3,1,1),nn.SiLU(),
                                       nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(c.width,d))
        elif c.lam_variant!="pair":
            raise ValueError("Unknown LAM encoder variant")
        self.books=nn.ModuleList([VQ(k,c.action_dim) for k in c.action_codes])
        self.decoder=nn.Sequential(nn.Conv2d(3+d,c.width,3,1,1),nn.SiLU(),
                                   nn.Conv2d(c.width,c.width,3,1,1),nn.SiLU(),nn.Conv2d(c.width,3,3,1,1))

    def features(self,x):
        b,t,ch,h,w=x.shape
        parts=[x[:,:-1],x[:,1:]]
        if self.c.lam_variant=="delta": parts.append(x[:,1:]-x[:,:-1])
        return self.encoder(torch.cat(parts,2).reshape(b*(t-1),-1,h,w)).reshape(b,t-1,len(self.books),-1)

    def infer(self,x):
        features=self.features(x)
        parts=[book(features[:,:,i]) for i,book in enumerate(self.books)]
        return torch.cat([p[0] for p in parts],-1),torch.stack([p[1] for p in parts],-1),sum(p[2] for p in parts)

    def lookup(self,ids):
        return torch.cat([book.lookup(ids[...,i]) for i,book in enumerate(self.books)],-1)

    def reconstruct(self,current,actions):
        b,t,ch,h,w=current.shape
        spatial=actions[:,:, :,None,None].expand(-1,-1,-1,h,w)
        residual=self.decoder(torch.cat([current,spatial],2).reshape(b*t,-1,h,w)).reshape_as(current)
        return torch.sigmoid(torch.logit(current.clamp(.01,.99))+residual)

    def forward(self,x):
        actions,ids,loss=self.infer(x)
        return self.reconstruct(x[:,:-1],actions),actions,ids,loss


class STBlock(nn.Module):
    def __init__(self,d,heads):
        super().__init__()
        self.spatial=nn.MultiheadAttention(d,heads,batch_first=True)
        self.temporal=nn.MultiheadAttention(d,heads,batch_first=True)
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))

    def forward(self,x):
        b,t,n,d=x.shape
        a=self.norms[0](x).reshape(b*t,n,d)
        x=x+self.spatial(a,a,a,need_weights=False)[0].reshape(b,t,n,d)
        a=self.norms[1](x).transpose(1,2).reshape(b*n,t,d)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        x=x+self.temporal(a,a,a,attn_mask=mask,need_weights=False)[0].reshape(b,n,t,d).transpose(1,2)
        return x+self.ff(self.norms[2](x))


class Dynamics(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        self.embed=nn.Embedding(c.vocab+1,c.width)
        self.action=nn.Linear(c.action_dim*len(c.action_codes),c.width)
        self.spatial=nn.Parameter(torch.randn(1,1,(c.size//c.downsample)**2,c.width)*.02)
        self.time=nn.Parameter(torch.randn(1,c.context,1,c.width)*.02)
        self.blocks=nn.Sequential(*[STBlock(c.width,c.heads) for _ in range(c.layers)])
        self.head=nn.Sequential(nn.LayerNorm(c.width),nn.Linear(c.width,c.vocab))

    def forward(self,tokens,actions):
        b,t,h,w=tokens.shape
        if t > self.c.context or actions.shape[:2] != (b,t-1):
            raise ValueError("History/actions misaligned")
        # action a[t-1] controls frame t. Frame zero has no preceding action.
        cond=F.pad(self.action(actions),(0,0,1,0))
        x=self.embed(tokens.flatten(2))+self.spatial+self.time[:,:t]+cond[:,:,None]
        return self.head(self.blocks(x)).reshape(b,t,h,w,-1)

    @torch.no_grad()
    def next_tokens(self,history,actions,iterations=4):
        b,_,h,w=history.shape
        if actions.shape[1] != history.shape[1]:
            raise ValueError("One action required per transition including requested next frame")
        history=history[:,-(self.c.context-1):]
        actions=actions[:,-history.shape[1]:]
        current=torch.full((b,1,h,w),self.c.vocab,dtype=torch.long,device=history.device)
        for step in range(iterations):
            probs=self(torch.cat([history,current],1),actions)[:,-1].softmax(-1)
            confidence,pred=probs.max(-1)
            unknown=current[:,0].eq(self.c.vocab)
            confidence=confidence.masked_fill(~unknown,-1)
            remaining=unknown.flatten(1).sum(1)
            for i in range(b):
                count=math.ceil(int(remaining[i])/(iterations-step))
                if count:
                    indices=confidence[i].flatten().topk(count).indices
                    current[i,0].view(-1)[indices]=pred[i].flatten()[indices]
        return current


class WorldModel(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.config=c
        self.tokenizer=VideoTokenizer(c)
        self.lam=LatentActionModel(c)
        self.dynamics=Dynamics(c)

    def config_dict(self):
        return asdict(self.config)


def image_loss(pred,target,previous=None,color_balance=False):
    chroma=target.max(2,keepdim=True).values-target.min(2,keepdim=True).values
    weights=1+10*(chroma>.12).float().detach()
    if previous is not None:
        weights=.05+((target-previous).abs().mean(2,keepdim=True)>.03).float().detach()
    weights=weights/weights.mean((-3,-2,-1),keepdim=True).clamp_min(.01)
    edges=(torch.diff(pred,dim=-1)-torch.diff(target,dim=-1)).abs().mean()
    edges=edges+(torch.diff(pred,dim=-2)-torch.diff(target,dim=-2)).abs().mean()
    loss=((pred-target).square()*weights).mean() + .1*(pred-target).abs().mean()+.2*edges
    if color_balance:
        # Self-supervised color groups from target pixels; no external labels.
        # Average within groups before across groups, so a large cyan tank cannot
        # dominate the color error of a small blue/yellow object. Require 4 pixels
        # to avoid making a single compression artifact a full color category.
        r,g,b=target.unbind(2)
        masks=torch.stack(((r-b>.12)&(g-b>.12),
                           (r-g>.12)&(r-b>.08),
                           (b-r>.12)&(g-r>.12)&(b-g<.12),
                           (b-r>.12)&(b-g>=.12)),2).float().detach()
        counts=masks.sum((-2,-1));valid=counts>=4
        error=(pred-target).square().mean(2,keepdim=True)
        per_group=(error*masks).sum((-2,-1))/counts.clamp_min(1)
        loss=loss+(per_group*valid).sum()/valid.sum().clamp_min(1)
    return loss


def masked_dynamics_loss(dynamics,ids,actions):
    mask=torch.rand(ids.shape,device=ids.device) < float(torch.empty((),device=ids.device).uniform_(.5,1.))
    mask[:,0]=False
    # Guarantee at least one supervised token even in tiny test batches.
    mask[:,-1,0,0]=True
    logits=dynamics(ids.masked_fill(mask,dynamics.c.vocab),actions.detach())
    return F.cross_entropy(logits[mask],ids[mask])
