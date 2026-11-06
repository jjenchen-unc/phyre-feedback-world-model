import sys
import numpy as np
from PIL import Image, ImageDraw
sys.argv = ['x', '--preds', 'x', '--out', 'x']
W = 256
GF = list(range(0, 128, 4))
GREEN, BLUE, PURPLE = 9, 10, 11
WAD = np.array([[255, 255, 255], [0xf3, 0x4f, 0x46], [0x6b, 0xce, 0xbb], [0x18, 0x77, 0xf2],
                [0x4b, 0x4a, 0xa4], [0xb9, 0xca, 0xd2], [0, 0, 0], [0xfc, 0xdf, 0xe3]], np.uint8)


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


z = np.load('results/preds_levjepa_full_div16/ball_cross_template_full.npz', allow_pickle=True)
y, s = z['solved'], z['e2e_scores']
fail_med = np.median(s[y == 0])
fn = [k for k in range(len(y)) if y[k] and s[k] <= fail_med]
print('FN rollouts:', [(str(z['tasks'][k]), round(float(s[k]), 2)) for k in fn])
tiles = []
for k in fn[:6]:
    d = np.load('data/' + str(z['files'][k]))
    f, fr, p = d['features'][GF], d['frames'][GF], z['pred'][k]
    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
    t = int(np.argmin(surface_dist(f)))
    err = np.linalg.norm((p[:, g, :2] - f[:, g, :2]) * W, axis=-1)
    img = Image.fromarray(WAD[fr[t]][::-1].copy()).resize((512, 512), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    for xy, col in [(f[t, g, :2], (0, 160, 0)), (p[t, g, :2], (255, 0, 255))]:
        x, yy = xy[0] * 512, (1 - xy[1]) * 512
        dr.ellipse([x - 7, yy - 7, x + 7, yy + 7], outline=col, width=3)
    dr.text((6, 6), '%s t=%d err@t=%.0fpx maxerr=%.0fpx' % (z['tasks'][k], t, err[t], err.max()), fill=(0, 0, 0))
    print(z['tasks'][k], 'err per step:', np.round(err).astype(int).tolist())
    tiles.append(img)
canvas = Image.new('RGB', (512 * 3, 512 * ((len(tiles) + 2) // 3)), (255, 255, 255))
for i, im in enumerate(tiles):
    canvas.paste(im, ((i % 3) * 512, (i // 3) * 512))
canvas.save('results/check2_fn_examples.png')
print('saved', len(tiles))
