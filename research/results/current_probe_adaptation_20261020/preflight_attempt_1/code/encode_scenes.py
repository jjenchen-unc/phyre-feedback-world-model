"""Canonical A6000 scene-classifier encoding, verified against existing tokens."""
import ast
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

P = Path(__file__).resolve().parents[3]
R = P/'results/current_probe_adaptation_20261020'
S = P/'results/q0_first_feedback_20261018'
C = P/'results/candidate_coverage_100k_20261016'
torch.set_num_threads(2)
assert 'A6000' in torch.cuda.get_device_name(0)
tree = ast.parse((S/'code/scene_classifier.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'encode']
assert len(nodes) == 1
dev = torch.device('cuda')
env = dict(torch=torch, F=F, np=np, dev=dev,
           WAD=torch.tensor([[255,255,255],[243,79,70],[107,206,187],[24,119,242],[75,74,164],[185,202,210],[0,0,0],[252,223,227]], device=dev)/255,
           MEAN=torch.tensor([.485,.456,.406], device=dev).view(1,3,1,1),
           STD=torch.tensor([.229,.224,.225], device=dev).view(1,3,1,1))
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<canonical-scene-encode>', 'exec'), env)
model = AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large', trust_remote_code=True,
                                 revision='e831a0347737fcaa660b39c57d41c109de399845',
                                 local_files_only=True).eval().cuda().half()
# Verify original batch shape and values against a canonical 32-scene block.
reference_frames = np.load(R/'inputs/reference_frames.npy')
with np.load(C/'inputs/cross_tokens.npz') as z:
    expected = z['tokens'][:32].copy()
actual = env['encode'](reference_frames, model).numpy()
np.testing.assert_array_equal(actual, expected)
with np.load(R/'inputs/scenes.npz') as z:
    tokens = env['encode'](z['frames'], model).numpy()
    np.savez(R/'inputs/tokens.npz', tasks=z['tasks'], tokens=tokens)
(R/'SCENE_ENCODING.json').write_text(json.dumps(dict(passed=True, reference_exact=True, n=96))+'\n')
print('SCENE_ENCODING_EXACT', flush=True)
