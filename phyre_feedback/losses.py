"""The geometry loss used by the matched pretraining and feedback pilots."""
import numpy as np
import torch

def geometry_loss(probe, prediction, features, times):
    """Equal query weight, then equal dynamic-object/time weight; initial queries only."""
    per_query = []
    for j, (f, ts) in enumerate(zip(features, times)):
        tok = prediction[j, ts].float()
        kv = probe.inp(tok) + probe.space
        objects = []
        for o in range(f.shape[1]):
            if f[0, o, 13] >= .5 or f[0, o, 11] >= .5:
                continue
            q = torch.as_tensor(np.repeat(f[:1, o], len(ts), axis=0), device=tok.device)
            xy, angle = probe.read(kv, q)
            target = torch.as_tensor(f[ts, o, :2] * 256, device=tok.device)
            loss = (xy - target).abs().mean() / 256 * 10
            if f[0, o, 4:8].argmax() == 1:
                theta = torch.as_tensor(f[ts, o, 2] * (4 * np.pi), device=tok.device)
                loss = loss + (angle - torch.stack([theta.cos(), theta.sin()], -1)).abs().mean()
            objects.append(loss)
        assert objects
        per_query.append(torch.stack(objects).mean())
    return torch.stack(per_query).mean()
