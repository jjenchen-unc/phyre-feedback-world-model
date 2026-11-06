"""Read-only hashes for experiment artifacts, never scans outside this study."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checkpoints={}
for folder in [R/'models',R/'calibration_update/models',R/'contact_training/models']:
    for p in folder.rglob('complete.json'):
        receipt=json.loads(p.read_text());ck=p.parent/'updater.pt'
        assert receipt['complete'] and receipt['checkpoint_sha']==sha(ck),str(ck)
        checkpoints[str(ck.relative_to(R))]=sha(ck)
assert len(checkpoints)==16,len(checkpoints)
scores={str(p.relative_to(R)):sha(p) for p in R.rglob('scores.npz')}
sources={str(p.relative_to(R)):sha(p) for p in R.rglob('*.py')}
print(json.dumps(dict(checkpoints=checkpoints,scores=scores,sources=sources),indent=2))
