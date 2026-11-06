import argparse, json, os, time
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModel

p = argparse.ArgumentParser()
p.add_argument('--encoder', choices=['levjepa', 'vjepa2'], required=True)
p.add_argument('--chunk', choices=['overlap', 'indep'], default='overlap')
p.add_argument('--manifests', nargs='+', required=True)
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--out', default='/path/to/phyre-unc/latents')
p.add_argument('--batch', type=int, default=8)
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
p.add_argument('--pca', default='')
args = p.parse_args()

dev = torch.device('cuda')
WAD = torch.tensor([[255, 255, 255], [0xf3, 0x4f, 0x46], [0x6b, 0xce, 0xbb], [0x18, 0x77, 0xf2],
                    [0x4b, 0x4a, 0xa4], [0xb9, 0xca, 0xd2], [0, 0, 0], [0xfc, 0xdf, 0xe3]], dtype=torch.float32, device=dev) / 255
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
GRID_FRAMES = list(range(0, 128, 4))
REV = {'levjepa': 'e831a0347737fcaa660b39c57d41c109de399845'}

if args.encoder == 'levjepa':
    model = AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large', trust_remote_code=True, revision=REV['levjepa'])
    RES, TUB, G = 224, 1, 14
else:
    model = AutoModel.from_pretrained('facebook/vjepa2-vitl-fpc64-256')
    RES, TUB, G = 256, 2, 16
model = model.eval().to(dev).half()


def to_pixels(codes):
    x = WAD[torch.from_numpy(codes.astype(np.int64)).to(dev)].flip(1).permute(0, 3, 1, 2)
    if RES != 256:
        x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False, antialias=True)
    return ((x - MEAN) / STD).half()


@torch.no_grad()
def encode_clips(clips):
    if args.encoder == 'levjepa':
        h = model(pixel_values=clips.permute(0, 2, 1, 3, 4)).last_hidden_state[:, 1:]
    else:
        h = model.get_vision_features(clips)
    B = clips.shape[0]
    return h.reshape(B, 16 // TUB, G, G, -1)


@torch.no_grad()
def encode_rollout(frames):
    x = to_pixels(frames)
    starts = list(range(0, 128, 16)) if args.chunk == 'indep' else list(range(0, 113, 8))
    clips = torch.stack([x[s:s + 16] for s in starts])
    h = torch.cat([encode_clips(clips[i:i + args.batch]) for i in range(0, len(clips), args.batch)])
    if args.chunk == 'indep':
        steps = h.reshape(-1, G, G, h.shape[-1])
    else:
        keep = 8 // TUB
        steps = torch.cat([h[0]] + [h[i, -keep:] for i in range(1, len(h))])
    assert steps.shape[0] == 128 // TUB, steps.shape
    return steps


def check_token_order():
    torch.manual_seed(0)
    base = torch.randn(1, 16, 3, RES, RES, device=dev).half()
    ref = encode_clips(base).float()
    pert = base.clone()
    pert[:, :, :, :16, :16] += 3
    d = (encode_clips(pert).float() - ref).norm(dim=-1)[0].mean(0)
    top = divmod(int(d.argmax()), G)
    out = {'spatial_argmax_hw': top}
    if args.encoder == 'levjepa':
        pert2 = base.clone(); pert2[:, 10] += 3
        dt = (encode_clips(pert2).float() - ref).norm(dim=-1)[0].mean((1, 2))
        out['frames_changed_by_frame10'] = [int(i) for i in torch.nonzero(dt > 1e-3).flatten()]
        assert out['frames_changed_by_frame10'][0] == 10, out
    assert top == (0, 0), out
    return out


PCA = None
if args.pca:
    _p = torch.load(args.pca, map_location='cpu')
    PCA = (_p['mean'].to(dev), _p['components'].to(dev))


if __name__ == '__main__':
    t0 = time.time()
    order = check_token_order()
    print('token order check', order, flush=True)
    odir = os.path.join(args.out, f'{args.encoder}_{args.chunk}_g32' + (f'_pca{PCA[1].shape[1]}' if PCA else ''))
    os.makedirs(odir, exist_ok=True)
    rows = []
    for m in args.manifests:
        rows += [json.loads(l) for l in open(os.path.join(args.data, 'manifests', m + '.jsonl'))]
    files = sorted({r['file'] for r in rows})[args.shard::args.nshards]
    done = 0
    for f in files:
        path = os.path.join(odir, os.path.basename(f).replace('.npz', '.npy'))
        if os.path.exists(path):
            continue
        frames = np.load(os.path.join(args.data, f))['frames']
        steps = encode_rollout(frames)
        grid = steps[[fr // TUB for fr in GRID_FRAMES]].reshape(len(GRID_FRAMES), G * G, -1)
        if PCA:
            grid = (grid.float() - PCA[0]) @ PCA[1]
        np.save(path + '.tmp.npy', grid.half().cpu().numpy())
        os.replace(path + '.tmp.npy', path)
        done += 1
        if done % 200 == 0:
            print(f'{done}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    meta = {'encoder': args.encoder, 'chunk': args.chunk, 'res': RES, 'tubelet': TUB, 'grid': G,
            'grid_frames': GRID_FRAMES, 'files': len(files), 'shard': args.shard, 'nshards': args.nshards,
            'encoded_now': done, 'token_order_check': order, 'seconds': round(time.time() - t0, 1),
            'revision': REV.get(args.encoder), 'manifests': args.manifests,
            'preprocess': 'WAD palette, rows flipped upright, ImageNet norm; levjepa bilinear 256->224; vjepa2 native 256, no crop'}
    json.dump(meta, open(os.path.join(odir, f'meta_shard{args.shard}of{args.nshards}.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))
