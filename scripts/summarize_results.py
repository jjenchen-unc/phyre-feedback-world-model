"""Recompute the published S@9 table and paired gains/losses from task records."""
import json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
d=json.loads((r/'results/milestone.json').read_text())
for panel,p in d['classifier_first_feedback'].items():
    print('\n'+panel.upper())
    base=p['methods']['q0']['first_success']
    solved=lambda x:x is not None and x<=9
    for name,m in p['methods'].items():
        assert len(m['first_success'])==m['n']==100
        for k in [1,3,5,9]:
            assert sum(v is not None and v<=k for v in m['first_success'].values())==m['counts'][f's{k}']
        assert set(base)==set(m['first_success'])
        gain=sum(solved(v) and not solved(base[t]) for t,v in m['first_success'].items())
        loss=sum(solved(base[t]) and not solved(v) for t,v in m['first_success'].items())
        print(f"{name:18} S@9={m['counts']['s9']:3}%  gained/lost vs Q0={gain}/{loss}")
print('\nAll per-task counts verified. No Q0-first effectiveness results included.')
