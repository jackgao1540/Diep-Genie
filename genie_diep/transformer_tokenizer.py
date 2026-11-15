"""Small ST-transformer tokenizer components following Genie (2402.15391v1).

Sections 2/2.1: frame-local spatial attention, causal same-site temporal
attention, then one feedforward. Width/depth, learned positions and pre-LN are
local choices, not the paper's full-size architecture or training recipe.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TokenizerSTBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.spatial = nn.MultiheadAttention(width, heads, batch_first=True)
        self.temporal = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x):
        b, t, n, d = x.shape
        a = self.norms[0](x).reshape(b*t, n, d)
        x = x+self.spatial(a, a, a, need_weights=False)[0].reshape(b, t, n, d)
        a = self.norms[1](x).transpose(1, 2).reshape(b*n, t, d)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        x = x+self.temporal(a, a, a, attn_mask=mask, need_weights=False)[0].reshape(b, n, t, d).transpose(1, 2)
        return x+self.ff(self.norms[2](x))


class TokenizerStack(nn.Module):
    def __init__(self, width, depth, heads, grid, context):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, depth, heads, grid, context)) or width % heads:
            raise ValueError('Positive integer dimensions and width divisible by heads required')
        self.grid, self.context = grid, context
        self.spatial = nn.Parameter(torch.randn(1, 1, grid*grid, width)*.02)
        self.time = nn.Parameter(torch.randn(1, context, 1, width)*.02)
        self.blocks = nn.Sequential(*[TokenizerSTBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] != self.grid**2 or not 1 <= x.shape[1] <= self.context:
            raise ValueError('Tokenizer requires its configured grid and 1..context frames')
        return self.norm(self.blocks(x+self.spatial+self.time[:, :x.shape[1]]))


class TransformerEncoder(nn.Module):
    def __init__(self, c, width, depth, heads):
        super().__init__()
        self.size, self.dim = c.size, c.token_dim
        self.patch = nn.Linear(48, width)
        self.stack = TokenizerStack(width, depth, heads, c.size//4, c.context)
        self.output = nn.Linear(width, c.token_dim)

    def forward(self, x):
        if x.ndim != 5 or x.shape[2:] != (3, self.size, self.size) or not 1 <= x.shape[1] <= self.stack.context:
            raise ValueError('Encoder expects B,T,3,size,size RGB input')
        b, t = x.shape[:2]
        patches = F.pixel_unshuffle(x.reshape(b*t, 3, self.size, self.size), 4)
        patches = patches.flatten(2).transpose(1, 2).reshape(b, t, -1, 48)
        z = self.output(self.stack(self.patch(patches)))
        return z.reshape(b, t, self.size//4, self.size//4, self.dim)


class TransformerDecoder(nn.Module):
    def __init__(self, c, width, depth, heads):
        super().__init__()
        self.grid, self.dim = c.size//4, c.token_dim
        self.input = nn.Linear(c.token_dim, width)
        self.stack = TokenizerStack(width, depth, heads, self.grid, c.context)
        self.output = nn.Linear(width, 48)

    def forward(self, q):
        if q.ndim != 5 or q.shape[2:] != (self.grid, self.grid, self.dim):
            raise ValueError('Decoder expects B,T,grid,grid,token_dim code embeddings')
        b, t = q.shape[:2]
        x = self.output(self.stack(self.input(q.flatten(2, 3)))).sigmoid()
        x = x.reshape(b*t, self.grid*self.grid, 48).transpose(1, 2).reshape(b*t, 48, self.grid, self.grid)
        return F.pixel_shuffle(x, 4).reshape(b, t, 3, self.grid*4, self.grid*4)
