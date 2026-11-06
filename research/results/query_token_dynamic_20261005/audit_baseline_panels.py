"""Independently recount old 20-scene baseline vs staged 100-scene baseline."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
STAGED=R.parent/'query_token_staged_20261005'
read=lambda p:json.loads(p.read_text())
m=read(R/'manifest.json');sm=read(STAGED/'manifest.json')
assert m['base_sha256']==sm['base_sha256'] and m['heads']['predicted_mix_s0']==sm['heads']['predicted_mix_s0']
assert m['cross_meta_sha256']==sm['cross_meta_sha256']
w=read(R/'within_inputs.json');sw=read(STAGED/'within_inputs.json')
assert w['pack']==sw['pack'] and w['meta_sha256']==sw['meta_sha256']
def rank(y,s):
    hits=np.flatnonzero(y[np.argsort(-s,kind='stable')])
    return int(hits[0])+1 if len(hits) else None
def hit(r,k):return r is not None and r<=k
def stable(x):return int(hashlib.sha256(str(x).encode()).hexdigest()[:16],16)
out={'base_sha256':m['base_sha256'],'head_sha256':m['heads']['predicted_mix_s0'],'splits':{}}
for split,section in [('cross','evaluation'),('within','within_evaluation')]:
    specs=w['specs'] if split=='within' else [s for s in m['final_panels'] if s['panel']=='cross_natural']
    per=m['evaluation'][split+'_per_template']
    selected=[s for t in sorted({s['task'][:5] for s in specs}) for s in sorted([s for s in specs if s['task'][:5]==t],key=lambda s:stable((20261005,'ablation_closed',s['task'])))[:per]]
    chosen={s['task'] for s in selected};assert len(chosen)==20
    rows=[];n_checked=0
    for p in sorted((STAGED/section/'q0_s0'/'closed').glob('*/scores.npz')):
        with np.load(p) as d:
            ids=d['indices'];y=d['labels'];scores=d['original']
        task=p.parent.name.replace('_',':');first=rank(y,scores)
        assert first==read(p.parent/'result.json')['original_first_rank']
        rows.append(dict(task=task,template=task[:5],first_success_rank=first,in_old_panel=task in chosen))
        if task not in chosen:continue
        for folder in sorted((R/section).glob('*_s*')):
            path=folder/'closed'/'full'/p.parent.name
            with np.load(path/'snapshots.npz') as d:
                np.testing.assert_array_equal(d['indices'],ids)
                np.testing.assert_array_equal(d['labels'],y)
                np.testing.assert_array_equal(d['base_scores'],scores)
            r=read(path/'trace.json')['result'];expected=first if hit(first,20) else 0
            assert r['task']==task and r['base_first_success']==expected
            assert r['base_success9']==hit(first,9) and r['base_success20']==hit(first,20)
            n_checked+=1
    assert len(rows)==100 and n_checked==300
    old=[r for r in rows if r['in_old_panel']];assert {r['task'] for r in old}==chosen
    rest=[r for r in rows if not r['in_old_panel']]
    result={'old_panel_n':len(old),'old_s9':sum(hit(r['first_success_rank'],9) for r in old),'old_s20':sum(hit(r['first_success_rank'],20) for r in old),
            'full_n':len(rows),'full_s9':sum(hit(r['first_success_rank'],9) for r in rows),
            'remaining_n':len(rest),'remaining_s9':sum(hit(r['first_success_rank'],9) for r in rest),'raw_model_scene_comparisons':n_checked,
            'per_template':{},'old_panel_rows':old}
    for t in sorted({r['template'] for r in old}):
        rr=[r for r in old if r['template']==t];result['per_template'][t]={'n':len(rr),'s9':sum(hit(r['first_success_rank'],9) for r in rr)}
    summary=read(R/('WITHIN_SUMMARY.json' if split=='within' else 'SUMMARY.json'))
    for v in summary['per_model'].values():
        assert v['closed']['full']['base_success9']==result['old_s9']/20
        assert v['closed']['full']['base_success20']==result['old_s20']/20
    out['splits'][split]=result
c={r['task'] for r in out['splits']['cross']['old_panel_rows']};w={r['task'] for r in out['splits']['within']['old_panel_rows']}
assert not c&w and not {t[:5] for t in c}&{t[:5] for t in w}
out.update(passed=True,scene_and_template_overlap=False,selection='Deterministic task hash per template, rederived from registered code; no outcome-based selection')
(R/'BASELINE_PANEL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({**out,'splits':{s:{k:v for k,v in d.items() if k!='old_panel_rows'} for s,d in out['splits'].items()}},indent=2))
