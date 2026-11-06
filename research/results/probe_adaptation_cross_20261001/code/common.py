import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

W, T = 256, 32
GREEN, BLUE, PURPLE, BLACK = 9, 10, 11, 13


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


def surface_dist(f):
    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
    goals = np.flatnonzero((f[0, :, BLUE] > .5) | (f[0, :, PURPLE] > .5))
    cg, rg = f[:, g, :2] * W, f[0, g, 3] * W / 2
    ds = []
    for o in goals:
        co, L = f[:, o, :2] * W, f[0, o, 3] * W
        if np.argmax(f[0, o, 4:8]) == 1:
            th = f[:, o, 2] * 2 * np.pi
            u = np.stack([np.cos(th), np.sin(th)], -1)
            t = np.clip(((cg - co) * u).sum(-1), -L / 2, L / 2)
            ds.append(np.linalg.norm(cg - (co + t[:, None] * u), axis=-1) - rg)
        else:
            ds.append(np.linalg.norm(cg - co, axis=-1) - rg - L / 2)
    return np.min(ds, 0)


def longest(m):
    b = r = 0
    for v in m:
        r = r + 1 if v else 0
        b = max(b, r)
    return b


def rule_score(d, rule):
    kind, tau, s = rule
    return longest(d < tau) if kind == 'hard' else float((0.5 * (1 + np.tanh((tau - d) / (2 * s)))).sum())


class MemoryPredictor(nn.Module):
    def __init__(self, base, Din, d_mem=256, mode='full', mem_steps=8, mem_grid=7, per_attempt=False, freeze_base=True, gate_empty=False):
        super().__init__()
        self.base, self.mode, self.mem_steps, self.mem_grid, self.per_attempt = base, mode, mem_steps, mem_grid, per_attempt
        self.gate_empty = gate_empty
        self.out_scale = 1.0
        for p_ in self.base.parameters():
            p_.requires_grad_(not freeze_base)
        d = base.inp[1].out_features
        self.zin = nn.Sequential(nn.LayerNorm(Din * (2 if mode == 'residual' else 1)), nn.Linear(Din * (2 if mode == 'residual' else 1), d_mem))
        self.act = nn.Sequential(nn.Linear(3, d_mem), nn.GELU(), nn.Linear(d_mem, d_mem))
        self.outcome = nn.Embedding(2, d_mem)
        self.mtime = nn.Parameter(torch.zeros(1, mem_steps, 1, d_mem))
        self.mspace = nn.Parameter(torch.zeros(1, 1, mem_grid * mem_grid, d_mem))
        self.null = nn.Parameter(torch.zeros(1, 1, d_mem))
        self.menc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_mem, 4, 4 * d_mem, batch_first=True, norm_first=True), 2)
        L = len(base.blocks)
        self.qn = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.kv = nn.ModuleList([nn.Linear(d_mem, 2 * d) for _ in range(L)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        for o in self.o:
            nn.init.zeros_(o.weight); nn.init.zeros_(o.bias)
        self.h = base.blocks[0].sp.h

    @torch.no_grad()
    def context_residual(self, ctx_full, ctx_act):
        was = self.base.training
        self.base.eval()
        B, K = ctx_act.shape[:2]
        z = ctx_full.reshape(B * K, *ctx_full.shape[2:])
        pr = self.base(z[:, 0], ctx_act.reshape(B * K, 3))
        self.base.train(was)
        return (z - pr).reshape(ctx_full.shape)

    def encode_memory(self, ctx_lat, ctx_act, ctx_out, ctx_mask, ctx_res=None):
        B, K = ctx_act.shape[:2]
        a = self.act(ctx_act) + (self.outcome(ctx_out.long()) if self.mode != 'action' else 0)
        if self.mode in ('action', 'outcome'):
            tok, tmask = a, ctx_mask
        else:
            T_ = ctx_lat.shape[2]
            z = ctx_lat[:, :, ::T_ // self.mem_steps][:, :, :self.mem_steps]
            if self.mode == 'residual':
                z = torch.cat([z, ctx_res[:, :, ::ctx_res.shape[2] // self.mem_steps][:, :, :self.mem_steps]], -1)
            G_ = int(round(z.shape[3] ** .5))
            z = z.reshape(B * K * self.mem_steps, G_, G_, -1).permute(0, 3, 1, 2)
            z = F.adaptive_avg_pool2d(z, self.mem_grid).permute(0, 2, 3, 1).reshape(B, K, self.mem_steps, -1, z.shape[1])
            tok = self.zin(z) + self.mtime + self.mspace + a[:, :, None, None]
            n = tok.shape[2] * tok.shape[3]
            if self.per_attempt:
                tok = self.menc(tok.reshape(B * K, n, -1)).reshape(B, K * n, -1)
                tmask = ctx_mask[:, :, None].expand(B, K, n).reshape(B, K * n)
                tok = torch.cat([self.null.expand(B, -1, -1), tok], 1)
                tmask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=tok.device), tmask], 1)
                return tok, tmask
            tok = tok.reshape(B, K * n, -1)
            tmask = ctx_mask[:, :, None].expand(B, K, n).reshape(B, K * n)
        tok = torch.cat([self.null.expand(B, -1, -1), tok], 1)
        tmask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=tok.device), tmask], 1)
        return self.menc(tok, src_key_padding_mask=~tmask), tmask

    def forward(self, z0, a, mem, mmask):
        b = self.base
        x = b.inp(z0)[:, None] + b.time + b.space + b.act(a)[:, None, None]
        B, T_, S, D = x.shape
        g = mmask[:, 1:].any(1).to(x.dtype)[:, None, None, None] if self.gate_empty else 1.0
        for l, blk in enumerate(b.blocks):
            x = blk(x)
            q = self.q[l](self.qn[l](x)).reshape(B, T_ * S, self.h, D // self.h).transpose(1, 2)
            k, v = self.kv[l](mem).reshape(B, mem.shape[1], 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mmask[:, None, None, :])
            x = x + self.out_scale * g * self.o[l](att.transpose(1, 2).reshape(B, T_, S, D))
        out = z0[:, None] + b.out(x)
        return torch.cat([z0[:, None], out[:, 1:]], 1)


class SceneClassifier(nn.Module):
    def __init__(self, D=1024, S=196, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.space = nn.Parameter(torch.zeros(1, S, d))
        self.act = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), 2)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, tok, a):
        h = self.inp(tok.float()) + self.space + self.act(a)[:, None]
        h = self.body(h)
        return self.out(self.pool(self.q.expand(len(h), -1, -1), h, h)[0][:, 0]).squeeze(-1)
