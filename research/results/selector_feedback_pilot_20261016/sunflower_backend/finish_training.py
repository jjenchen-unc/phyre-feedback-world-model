"""After-all-jobs completion receipt; never reports an absent checkpoint as trained."""
from pathlib import Path
import hashlib,json,os,time
B=Path(__file__).resolve().parent;R=B.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected=read(B/'VALIDATION_SUMMARY.json')['provenance'];arms=[]
for panel in ['cross','within']:
    for kind in ['residual','zero']:
        p=R/'models'/panel/kind/'complete.json';entry={'panel':panel,'kind':kind,'complete':False}
        if p.exists():
            x=read(p)
            assert x['complete'] and x['provenance']==expected and x['slots']==256
            assert sha(Path(x['checkpoint']))==x['checkpoint_sha256']
            entry.update(x)
        arms.append(entry)
value={'stage':'trained' if all(x['complete'] for x in arms) else 'training_incomplete',
       'training_complete':all(x['complete'] for x in arms),'evaluation_complete':False,
       'time':time.time(),'provenance':expected,'arms':arms}
tmp=R/f'STATUS.{os.getpid()}.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(R/'STATUS.json')
(B/'TRAINING_SUMMARY.json').write_text(json.dumps(value,indent=2)+'\n')
# This service belongs to these training jobs; free it once all array jobs exit.
(B/'rpc/STOP').write_text('Training array ended; inspect TRAINING_SUMMARY.json before evaluation.\n')
print(json.dumps(value,indent=2))
if not value['training_complete']:raise SystemExit(1)
