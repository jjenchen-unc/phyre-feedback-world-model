import json, os
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModel
dev = torch.device('cuda')
WAD = torch.tensor([[255, 255, 255], [0xf3, 0x4f, 0x46], [0x6b, 0xce, 0xbb], [0x18, 0x77, 0xf2],
                    [0x4b, 0x4a, 0xa4], [0xb9, 0xca, 0xd2], [0, 0, 0], [0xfc, 0xdf, 0xe3]], dtype=torch.float32, device=dev) / 255
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
m = AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large', trust_remote_code=True, revision='e831a0347737fcaa660b39c57d41c109de399845').eval().to(dev).half()
rows = [json.loads(l) for l in open('/path/to/phyre-unc/data/manifests/ball_within_template_f0_dev_n100.jsonl')][:4]
for r in rows:
    codes = np.load('/path/to/phyre-unc/data/' + r['file'])['frames']
    ref = np.load('/mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32/' + os.path.basename(r['file']).replace('.npz', '.npy'))[0].astype(np.float32)
    x = WAD[torch.from_numpy(codes.astype(np.int64)).to(dev)].flip(1).permute(0, 3, 1, 2)
    x = ((F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True) - MEAN) / STD).half()
    out = {}
    with torch.no_grad():
        for T in [1, 2, 'dup2', 16]:
            try:
                clip = (x[[0, 0]] if T == 'dup2' else x[:T])[None].permute(0, 2, 1, 3, 4)
                h = m(pixel_values=clip).last_hidden_state[0, 1:1 + 196].float().cpu().numpy()
                out[T] = round(float(np.abs(h - ref).max()), 4), round(float(np.abs(h - ref).mean()), 5)
            except Exception as e:
                out[T] = type(e).__name__ + ': ' + str(e)[:120]
    print(r['task'], 'maxdiff vs stored (T frames):', out)
