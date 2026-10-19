"""Geometry-based score inputs; these are proxies, not simulator success labels."""
import numpy as np
W = 256
GREEN, BLUE, PURPLE, BLACK = 9, 10, 11, 13

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
