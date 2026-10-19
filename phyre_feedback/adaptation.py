"""AdaJEPA-style predictor adaptation; not an official AdaJEPA reproduction."""
import torch

class PredictorAdapter:
    """Same predictor-only adaptation for a bare base or a Q0 wrapper.

    Initial tokens remain frozen. Each scene resets weights/replay; each call
    recreates Adam. No online access to untried actions' true trajectories.
    """
    def __init__(self, model, lr=3e-4, steps=1, replay=5):
        self.model, self.lr, self.steps, self.replay = model, lr, steps, replay
        # A caller may reuse a model after updater training. Freezing a parameter
        # does not clear its existing .grad; adaptation must start clean.
        model.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
        backbone = model.base if hasattr(model, 'base') else model
        for layer in (backbone.blocks[-1], backbone.out[0]):
            layer.requires_grad_(True)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.snapshot = [p.detach().clone() for p in self.params]
        self.buffer = []

    @torch.no_grad()
    def reset(self):
        for p, initial in zip(self.params, self.snapshot):
            p.copy_(initial); p.grad = None
        self.buffer = []
        self.model.eval()

    def update(self, index, z0, action, truth):
        self.buffer.append((int(index), z0.detach(), action.detach(), truth.detach()))
        self.buffer = self.buffer[-self.replay:]
        z = torch.cat([x[1] for x in self.buffer])
        a = torch.cat([x[2] for x in self.buffer])
        target = torch.cat([x[3] for x in self.buffer])
        optimizer = torch.optim.Adam(self.params, lr=self.lr)
        losses = []
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = self.model(z, a).float()
                loss = (pred[:, 1:] - target[:, 1:]).abs().mean()
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None for p in self.model.parameters() if not p.requires_grad)
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.params)
                assert all(torch.isfinite(p.grad).all() for p in self.params if p.grad is not None)
                optimizer.step(); losses.append(float(loss.detach()))
        return {'replay_indices': [x[0] for x in self.buffer], 'losses': losses}
