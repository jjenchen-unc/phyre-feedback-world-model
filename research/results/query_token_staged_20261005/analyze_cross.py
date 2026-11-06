"""Post-hoc CPU attribution using saved scores; does not run/change models."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'evaluation'
HEAD = 'predicted_mix_s0'
read = lambda p: json.loads(p.read_text())

def rank(s, y, mask=None):
    order = np.argsort(-s, kind='stable')
    if mask is not None: order = order[mask[order]]
    hits = np.flatnonzero(y[order])
    return int(hits[0]) + 1 if len(hits) else None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

rows = []
for seed in range(3):
    for p in sorted((BASE / f'q0_s{seed}' / 'closed').glob('*/scores.npz')):
        task = p.parent.name.replace('_', ':')
        with np.load(p) as z:
            y=z['labels'].astype(bool); ids=z['indices']; q0=z['q0']; orig=z['original']
        rr=read(p.parent/'result.json')
        qr=rank(q0,y); br=rank(orig,y)
        assert qr==rr['q0_first_rank'] and br==rr['original_first_rank']
        row=dict(task=task,template=task[:5],seed=seed,n_positive=int(y.sum()),q0_rank=qr,original_rank=br)
        for arm in ['residual','zero']:
            folder=BASE/f'{arm}_s{seed}'/'closed'/p.parent.name
            d=read(folder/'trace.json')
            with np.load(folder/'snapshots.npz') as z:
                ss=z['scores']; np.testing.assert_array_equal(z['indices'],ids)
                np.testing.assert_array_equal(z['labels'],y)
                np.testing.assert_array_equal(z['base_scores'],q0)
                np.testing.assert_array_equal(ss[0],q0)
            tried=np.zeros(len(y),bool); steps=[]; first=0
            for j,a in enumerate(d['attempts']):
                local=int(np.argmax(np.where(tried,-np.inf,ss[j])))
                assert local==a['local'] and int(ids[local])==a['index'] and bool(y[local])==a['solved']
                steps.append(dict(attempt=j+1,rank=rank(ss[j],y,~tried),q0_remaining_rank=rank(q0,y,~tried),
                                  all_rank=rank(ss[j],y),selected_q0_rank=int(np.flatnonzero(np.argsort(-q0,kind='stable')==local)[0])+1,
                                  mean_score_shift=float((ss[j]-q0).mean()),std_score_shift=float((ss[j]-q0).std())))
                tried[local]=True
                if y[local]: first=j+1; assert j==len(ss)-1
            assert first==d['result']['first_success'] and (first or len(ss)==9)
            assert all(d[k] for k in ['state_replay_exact','restored_state_score_replay_exact','network_weights_unchanged'])
            row[arm]=dict(first_success=first,steps=steps)
        rows.append(row)
assert len(rows)==300 and len({r['task'] for r in rows})==100

def counts(rr):
    c=Counter(n=len(rr))
    for r in rr:
        q=bool(r['q0_rank'] and r['q0_rank']<=9); b=bool(r['original_rank'] and r['original_rank']<=9)
        m=bool(r['residual']['first_success']); z=bool(r['zero']['first_success'])
        for key,val in dict(original=b,q0=q,residual=m,zero=z,no_positive=r['n_positive']==0,
                            rescued=(not q and m),harmed=(q and not m),both_fail=(not q and not m),
                            missed_available=(not m and r['n_positive']>0),q0_missed_available=(not q and r['n_positive']>0),
                            original_lost=(b and not m),original_gained=(not b and m)).items(): c[key]+=int(val)
    return dict(c)

out=dict(scope='Post-hoc reused cross panel, 100 scenes x 3 paired seeds, no new GPU inference',raw_closed_audit_passed=True,
         total=counts(rows),per_seed={str(s):counts([r for r in rows if r['seed']==s]) for s in range(3)},
         per_template={t:counts([r for r in rows if r['template']==t]) for t in sorted({r['template'] for r in rows})})
fail=[r for r in rows if not r['residual']['first_success'] and r['n_positive']]
out['available_but_failed']={'n':len(fail),'positive_count_median':mean([]) if not fail else float(np.median([r['n_positive'] for r in fail])),
    'q0_rank_quantiles':np.quantile([r['q0_rank'] for r in fail],[0,.25,.5,.75,1]).tolist(),
    'final_all_rank_quantiles':np.quantile([r['residual']['steps'][-1]['all_rank'] for r in fail],[0,.25,.5,.75,1]).tolist(),
    'final_vs_q0_same_remaining':dict(Counter('better' if r['residual']['steps'][-1]['rank']<r['residual']['steps'][-1]['q0_remaining_rank'] else 'worse' if r['residual']['steps'][-1]['rank']>r['residual']['steps'][-1]['q0_remaining_rank'] else 'same' for r in fail))}
out['cases']={k:[r for r in rows if pred(r)] for k,pred in {
    'harmed':lambda r:bool(r['q0_rank'] and r['q0_rank']<=9 and not r['residual']['first_success']),
    'rescued':lambda r:bool(r['q0_rank'] and r['q0_rank']>9 and r['residual']['first_success'])}.items()}
out['initial_rank_bins']={}
for lo,hi in [(1,9),(10,20),(21,50),(51,1024)]:
    rr=[r for r in rows if r['q0_rank'] and lo<=r['q0_rank']<=hi]
    out['initial_rank_bins'][f'{lo}-{hi}']={'n':len(rr),'residual_solved':sum(bool(r['residual']['first_success']) for r in rr)}
out['failure_partition']={'no_positive':sum(r['n_positive']==0 for r in rows),
    'q0_already_failed_with_positive':sum(r['n_positive']>0 and r['q0_rank']>9 and not r['residual']['first_success'] for r in rows),
    'q0_solved_but_update_failed':len(out['cases']['harmed'])}
out['rows']=rows
# Fixed one original-policy failed support: same queries, distinct from actual closed-loop paths.
fixed=[]
for r in rows:
    d=read(BASE/f"residual_s{r['seed']}"/'fixed'/r['task'].replace(':','_')/'summary.json')
    if not d['context']: continue
    a=d['conditions'][HEAD+'__base']; b=d['conditions'][HEAD+'__memory']
    if not a['n_query']: continue
    fixed.append(dict(task=r['task'],seed=r['seed'],closed_success=bool(r['residual']['first_success']),
        auc0=a['auc'],auc1=b['auc'],l10=a['all']['latent_l1'],l11=b['all']['latent_l1'],
        geom0=a['all']['surface_mae'],geom1=b['all']['surface_mae'],success_geom0=a['success']['surface_mae'],
        success_geom1=b['success']['surface_mae'],rank0=a['first_success_rank'],rank1=b['first_success_rank']))
out['fixed_one_support']={}
for name,rr in [('all',fixed),('closed_failed',[r for r in fixed if not r['closed_success']]),('closed_solved',[r for r in fixed if r['closed_success']])]:
    out['fixed_one_support'][name]={'n':len(rr),'means':{k:mean([r[k] for r in rr]) for k in ['auc0','auc1','l10','l11','geom0','geom1','success_geom0','success_geom1']},
        'auc_up':sum(r['auc0'] is not None and r['auc1']>r['auc0'] for r in rr),
        'auc_defined':sum(r['auc0'] is not None for r in rr),
        'l1_down':sum(r['l11']<r['l10'] for r in rr)}
# Matched scenes for every depth; no changing-cohort cumulative claims.
diag={}
for seed in range(3):
    for p in (BASE/f'residual_s{seed}'/'diagnostics').glob('*/*/full/summary.json'):
        d=read(p);k=int(p.parent.parent.name);task=d['task']
        diag[(seed,task,k)]=d['conditions']
common=sorted({(s,t) for s,t,k in diag if all((s,t,j) in diag for j in [1,2,4,8])})
out['diagnostic_common_k']={'n_seed_scenes':len(common),'n_scenes':len({t for s,t in common}),'by_k':{}}
for k in [1,2,4,8]:
    rr=[diag[s,t,k] for s,t in common]
    out['diagnostic_common_k']['by_k'][str(k)]={}
    for suffix in ['base','memory']:
        cc=[d[HEAD+'__'+suffix] for d in rr]
        out['diagnostic_common_k']['by_k'][str(k)][suffix]=dict(auc=mean([d['auc'] for d in cc]),auc_n=sum(d['auc'] is not None for d in cc),
            l1=mean([d['all']['latent_l1'] for d in cc]),geometry=mean([d['all']['surface_mae'] for d in cc]),success_geometry=mean([d['success']['surface_mae'] for d in cc]))
out['stress']={}
for seed in range(3):
    ds=[read(p)['result'] for p in (BASE/f'residual_s{seed}'/'stress').glob('*/trace.json')]
    out['stress'][str(seed)]={k:sum(bool(d[k]) for d in ds) for k in ['success9','success20','base_success9','base_success20']}
    out['stress'][str(seed)]['n']=len(ds)
(ROOT/'CROSS_FAILURE_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','cases']},indent=2))
