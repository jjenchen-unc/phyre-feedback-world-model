import json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for i in range(2):
 p=R/f'encoder{i}/rpc/STOP';p.parent.mkdir(parents=True,exist_ok=True);p.touch()
(R/'ENCODER_STOP_REQUESTED.json').write_text(json.dumps({'requested':True,'time':time.time(),'scope':'Only this experiment encoder0/1; bounded lifetime fallback8h'},indent=2)+'\n')
for attempt in range(30):
 statuses=[]
 for i in range(2):
  p=R/f'encoder{i}/rpc/STATUS.json'
  statuses.append(json.loads(p.read_text()) if p.exists() else {})
 if all(s.get('stage')=='stopped' for s in statuses):
  (R/'ENCODER_CLEANUP.json').write_text(json.dumps({'complete':True,'statuses':statuses,'time':time.time()},indent=2)+'\n')
  print('ENCODERS_STOPPED');break
 time.sleep(2)
else:raise RuntimeError('Stop requested but not confirmed; inspect owned encoder services')
