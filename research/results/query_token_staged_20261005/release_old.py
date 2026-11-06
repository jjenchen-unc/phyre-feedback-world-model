"""Restore user-held older evaluations after priority experiment audited completion."""
import datetime
import json
from pathlib import Path
import subprocess
ROOT=Path('/path/to/phyre-unc/results/query_token_staged_20261005')
for name in ['SUMMARY.json','WITHIN_SUMMARY.json']:
    result=json.loads((ROOT/name).read_text());assert result['complete'] and result['raw_audit_passed']
a=json.loads((ROOT/'PRIORITY_ACTIONS.json').read_text());released=[]
for job in a['held_pending_jobs']:
    result=subprocess.run(['scontrol','release',job],capture_output=True,text=True)
    released.append({'job':job,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
(ROOT/'OLD_JOBS_RELEASE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':released},indent=2)+'\n')
assert all(r['returncode']==0 for r in released)
