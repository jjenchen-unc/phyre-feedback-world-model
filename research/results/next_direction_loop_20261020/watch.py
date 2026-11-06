"""Local monitor. Emits one terminal result for same-thread follow-up."""
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = '/path/to/phyre-unc/results/next_direction_loop_20261020'
REMOTE = r'''
import json,subprocess
from pathlib import Path
r=Path(ROOT)
jobs=json.loads((r/'JOBS.json').read_text())
states=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'--format=JobID,State,ExitCode'],text=True)
failure=[x for x in states.splitlines() if any(s in x for s in ['FAILED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL'])]
external=[]
counts={k:len(list((r/d).glob(pattern))) for k,d,pattern in [
 ('heads','heads','*/complete.json'),('discovery','discovery','*/result.json'),
 ('confirmation','confirmation','*/result.json')]}
final=r/'FINAL_SUMMARY.json'
complete=final.exists() and json.loads(final.read_text())['complete']
print(json.dumps(dict(complete=complete,failure=failure,external_failure=external,counts=counts)))
'''.replace('ROOT', repr(ROOT))
cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'GSSAPIAuthentication=no', '-o', 'ControlPath=none',
       '-o', 'ConnectTimeout=8', 'sunflower',
       '/path/to/phyre-unc/.venv-ml/bin/python -c '+shlex.quote(REMOTE)]
start = time.time()
out = Path(__file__).parent/'WATCH_STATUS.json'
while time.time()-start < 172800:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        p.check_returncode()
        status = json.loads(p.stdout)
        status['checked_at'] = time.time()
        out.write_text(json.dumps(status, indent=2)+'\n')
        if status['complete'] or status['failure'] or status['external_failure']:
            print(json.dumps(status), flush=True)
            break
    except Exception as e:
        out.write_text(json.dumps(dict(last_monitor_error=str(e), checked_at=time.time()))+'\n')
    time.sleep(120)
else:
    print(json.dumps(dict(monitor_expired=True)), flush=True)
