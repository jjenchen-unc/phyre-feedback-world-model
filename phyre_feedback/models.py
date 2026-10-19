"""Action-conditioned axial transformer and frozen geometry probe."""
import torch
from torch import nn
from torch.nn import functional as F
W = 256

class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sp, self.tm = Attn(d, h), Attn(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T_, S, D = x.shape
        x = x + self.sp(self.n1(x).reshape(B * T_, S, D)).reshape(B, T_, S, D)
        y = self.n2(x).permute(0, 2, 1, 3).reshape(B * S, T_, D)
        x = x + self.tm(y).reshape(B, S, T_, D).permute(0, 2, 1, 3)
        return x + self.mlp(self.n3(x))


class Predictor(nn.Module):
    def __init__(self, Din, T_, S, d, L, h):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(Din), nn.Linear(Din, d))
        self.act = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Parameter(torch.zeros(1, T_, 1, d))
        self.space = nn.Parameter(torch.zeros(1, 1, S, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, Din))

    def forward(self, z0, a):
        x = self.inp(z0)[:, None] + self.time + self.space + self.act(a)[:, None, None]
        for b in self.blocks:
            x = b(x)
        out = z0[:, None] + self.out(x)
        return torch.cat([z0[:, None], out[:, 1:]], 1)


class AttnProbe(nn.Module):
    def __init__(self, D, S, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.space = nn.Parameter(torch.zeros(1, S, d))
        self.qin = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.ModuleList([nn.MultiheadAttention(d, 4, batch_first=True) for _ in range(2)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(2)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.out = nn.Linear(d, 4)

    def read(self, kv, q):
        h = self.qin(q)[:, None]
        for att, mlp, ln in zip(self.att, self.mlp, self.ln):
            h = h + att(ln(h), kv, kv)[0]
            h = h + mlp(h)
        o = self.out(h[:, 0])
        return torch.sigmoid(o[:, :2]) * W, torch.tanh(o[:, 2:])
