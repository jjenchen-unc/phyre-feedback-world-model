"""CPU data scheduling and raw-policy audit checks for the staged runner."""
import tempfile
from pathlib import Path
import numpy as np
import prefix_staged as p

def main():
    rows=[{'task':'test','candidates':[0,1],'query':list(range(2,14)),'times':[[1,2]]*12}]
    for i in range(100):
        a=p.episode(rows,1,i);b=p.episode(rows,1,i)
        assert a==b and len(a[1])==len(set(a[1]))==8
        assert not set(a[1])&set(a[0]['candidates']) and a[3] in [1,2,4,8]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        p.f.save_npz(root/'scores.npz',{'indices':np.arange(3),'labels':np.array([False,True,False]),
            'scores':np.array([[3.,2.,1.],[7.,4.,3.]]),'selected':np.array([0,1])})
        p.f.write_json(root/'result.json',{'stage':'residual','first_success':2,'attempts':2,'updates':1,'budget':9})
        p.audit_case(root)
        p.f.write_json(root/'result.json',{'stage':'residual','first_success':None,'attempts':2,'updates':1,'budget':9})
        try:p.audit_case(root)
        except AssertionError:pass
        else:raise AssertionError('Audit accepted wrong success outcome')
    print('{"passed":true,"deterministic_paired_episode_schedule":true,"raw_trace_audit_rejects_corruption":true}')
if __name__=='__main__':main()
