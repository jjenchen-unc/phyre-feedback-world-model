"""Full-panel evaluation of existing ABCDE checkpoints; no retraining.

Import the original frozen numerical implementation. This wrapper only changes
scene coverage, reuses immutable old traces, and derives static plain rankings.
"""
import argparse, json, hashlib, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'query_token_dynamic_20261005'
STAGED=ROOT.parent/'query_token_staged_20261005'
sys.path.insert(0,str(PARENT/'code'))
import query_token_dynamic as q
f=q.f
MEMBERS=[(arm,seed,split) for seed in range(3) for arm in q.ARMS[:3] for split in ['cross','within']]

def read(p):return json.loads(p.read_text())
def digest(p):return f.digest(p)
def write(p,d):f.write_json(p,d)
def section(split):return 'within_evaluation' if split=='within' else 'evaluation'
def rank(y,s):
    hits=np.flatnonzero(y[np.argsort(-s,kind='stable')])
    return int(hits[0])+1 if len(hits) else None
def hit(r,k):return r is not None and 0<r<=k
def checkpoint(arm,seed):return read(PARENT/'models'/f'{arm}_s{seed}'/'complete.json')
def specs_for(m,split):
    return read(PARENT/'within_inputs.json')['specs'] if split=='within' else [s for s in m['final_panels'] if s['panel']=='cross_natural']
def checked():
    m=q.checked(PARENT);reg=read(ROOT/'manifest.json')
    assert digest(PARENT/'manifest.json')==reg['parent_manifest_sha256']
    assert digest(Path(__file__))==reg['wrapper_sha256']
    assert digest(PARENT/'within_inputs.json')==reg['within_inputs_sha256']
    return m,reg
def verify_static(split,arm,seed):
    """The no-memory forward is independent of fixed support; compare raw arrays."""
    parent=PARENT/section(split)/f'{arm}_s{seed}';records=[];replayed=0
    for p in sorted((parent/'fixed').glob('*/truth.npz')):
        with np.load(p) as z:ids=z['indices'];y=z['labels']
        with np.load(p.parent/'predicted_mix_s0__memory.npz') as z:sc=z['log_score']
        with np.load(STAGED/section(split)/'q0_s0'/'closed'/p.parent.name/'scores.npz') as z:
            np.testing.assert_array_equal(ids,z['indices']);np.testing.assert_array_equal(y,z['labels']);base=z['original']
        first=rank(y,sc);bf=rank(y,base)
        old=parent/'closed'/'full'/p.parent.name
        if old.exists():
            trace=q.audit_closed(old)
            with np.load(old/'snapshots.npz') as z:
                np.testing.assert_array_equal(z['indices'],ids);np.testing.assert_array_equal(z['labels'],y)
                np.testing.assert_array_equal(z['base_scores'],base)
                for scores in z['scores']:np.testing.assert_array_equal(scores,sc)
            assert trace['result']['first_success']==(first if hit(first,20) else 0)
            replayed+=1
        records.append({'task':p.parent.name.replace('_',':'),'first_success_rank':first,'base_first_success_rank':bf,
                        'source':str(p.parent/'predicted_mix_s0__memory.npz')})
    assert len(records)==100 and replayed==20
    return {'records':records,'old20_static_score_replay_passed':True,'n':100,'arm':arm,'seed':seed,'split':split}

def prepare():
    assert not (ROOT/'manifest.json').exists(),'Do not replace a registered run'
    m=q.checked(PARENT);wm=read(PARENT/'within_inputs.json');sm=read(STAGED/'manifest.json');sw=read(STAGED/'within_inputs.json')
    assert m['cross_meta_sha256']==sm['cross_meta_sha256'] and wm['meta_sha256']==sw['meta_sha256']
    assert m['base_sha256']==sm['base_sha256'] and m['heads']==sm['heads']
    models={}
    for arm in q.ARMS:
        for seed in range(3):
            d=checkpoint(arm,seed)
            assert d['complete'] and d['optimizer_steps']==10000 and d['provenance']==f.provenance(PARENT)
            assert digest(Path(d['checkpoint']))==d['checkpoint_sha256']
            models[f'{arm}_s{seed}']={'path':d['checkpoint'],'sha256':d['checkpoint_sha256']}
    for split in ['cross','within']:
        assert read(PARENT/('WITHIN_SUMMARY.json' if split=='within' else 'SUMMARY.json'))['audit_passed']
        specs=specs_for(m,split);assert len(specs)==100 and len({s['task'] for s in specs})==100
        assert all(len(s['indices'])==1024 for s in specs)
        for arm in q.ARMS[3:]:
            for seed in range(3):write(ROOT/'static'/split/f'{arm}_s{seed}.json',verify_static(split,arm,seed))
        # Link only already completed immutable full recurrence outputs, provenance checked by closed_case.
        for arm in q.ARMS[:3]:
            for seed in range(3):
                old=PARENT/section(split)/f'{arm}_s{seed}';prov=read(old/'complete.json')['provenance']
                paths=sorted((old/'closed'/'full').glob('*/trace.json'));assert len(paths)==20
                for p in paths:
                    assert read(p)['provenance']==prov and read(p)['budget']==20
                    dst=ROOT/'closed'/split/f'{arm}_s{seed}'/p.parent.name
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    assert not dst.exists();dst.symlink_to(p.parent.resolve(),target_is_directory=True)
    reg={'scope':'Full100-scene closed-loop extension of frozen ABCDE v1; no training or architecture changes',
         'parent':str(PARENT),'parent_manifest_sha256':digest(PARENT/'manifest.json'),
         'within_inputs_sha256':digest(PARENT/'within_inputs.json'),'wrapper_sha256':digest(Path(__file__)),
         'models':models,'primary':'S@9','secondary':['S@1','S@3','S@5','S@20'],'budget':20,
         'tasks':{s:[v['task'] for v in specs_for(m,s)] for s in ['cross','within']},
         'members':[{'arm':a,'seed':s,'split':p} for a,s,p in MEMBERS],
         'limits':m['limits']+['Full100 replaces old20 coverage only; reused development panels',
         'S@20 extrapolates beyond8training updates; not official AUCCESS',
         'Memory arms use cross-attention reader and random-failure replay; not clarified prefix/on-policy method',
         'D/E static scores exactly replay old20; no need for repeated forward passes']}
    # Supersede only the obsolete 20-task coverage limitation in inherited limits.
    reg['limits']=[x for x in reg['limits'] if not x.startswith('Closed-loop20tasks')]
    write(ROOT/'manifest.json',reg);write(ROOT/'CPU_AUDIT.json',{'passed':True,'static_model_split_runs':12,
        'static_scene_scores':1200,'old_static_trace_replays':240,'memory_old_traces_reused':360,
        'parent_checkpoint_hashes':15,'manifest_sha256':digest(ROOT/'manifest.json')})
    print('PREPARE_PASS',flush=True)

def load_case(arm,seed,split):
    m,reg=checked();specs=specs_for(m,split);done=checkpoint(arm,seed);ck=Path(done['checkpoint'])
    assert digest(ck)==reg['models'][f'{arm}_s{seed}']['sha256']
    ref,probe,rule,model=q.models(m,arm,seed,ck);model.eval()
    heads,hashes=f.load_heads(m,f.PROBES,probe)
    assert all(hashes[h]==v for h,v in m['heads'].items())
    pack=read(PARENT/'within_inputs.json')['pack'] if split=='within' else m['cross_pack']
    expected=read(PARENT/'within_inputs.json')['meta_sha256'] if split=='within' else m['cross_meta_sha256']
    assert digest(Path(pack+'_meta.npz'))==expected
    source=f.Pack(pack);prov=read(PARENT/section(split)/f'{arm}_s{seed}'/'complete.json')['provenance']
    assert prov['checkpoint']==digest(ck)
    return specs,source,ref,model,heads[m['primary_head']],rule,prov

def smoke():
    checked();checks=[]
    for arm,split in [(q.ARMS[0],'cross'),(q.ARMS[1],'within'),(q.ARMS[2],'cross')]:
        specs,source,ref,model,head,rule,prov=load_case(arm,0,split)
        old=PARENT/section(split)/f'{arm}_s0'/'closed'/'full'
        p=next(p for p in sorted(old.glob('*/trace.json')) if len(read(p)['attempts'])>=3)
        spec=next(s for s in specs if s['task']==read(p)['result']['task'])
        dest=ROOT/'smoke'/f'{arm}_{split}'
        d=q.closed_case(dest,spec,source,model,ref,head,rule,prov,budget=3)
        q.audit_closed(dest)
        with np.load(dest/'snapshots.npz') as new,np.load(p.parent/'snapshots.npz') as original:
            for k in ['indices','labels','base_scores']:np.testing.assert_array_equal(new[k],original[k])
            for k in ['scores','tokens']:np.testing.assert_array_equal(new[k],original[k][:len(new[k])])
        assert d['attempts']==read(p)['attempts'][:len(d['attempts'])]
        checks.append({'arm':arm,'split':split,'task':spec['task'],'exact_saved_prefix_replay':True})
        print('SMOKE_CASE_PASS',arm,split,flush=True)
        del source,ref,model,head
        q.torch.cuda.empty_cache()
    write(ROOT/'SMOKE.json',{'passed':True,'manifest_sha256':digest(ROOT/'manifest.json'),'cases':checks})
    print('SMOKE_PASS',flush=True)

def evaluate(index):
    assert read(ROOT/'SMOKE.json')['passed']
    arm,seed,split=MEMBERS[index];dest=ROOT/'closed'/split/f'{arm}_s{seed}'
    if (dest/'complete.json').exists():return
    specs,source,ref,model,head,rule,prov=load_case(arm,seed,split)
    records=[];start=time.time();reused=0
    for spec in specs:
        folder=dest/spec['task'].replace(':','_');reused+=int(folder.is_symlink())
        d=q.closed_case(folder,spec,source,model,ref,head,rule,prov,budget=20)
        q.audit_closed(folder)
        with np.load(folder/'snapshots.npz') as raw,np.load(STAGED/section(split)/'q0_s0'/'closed'/folder.name/'scores.npz') as original:
            np.testing.assert_array_equal(raw['indices'],original['indices'])
            np.testing.assert_array_equal(raw['labels'],original['labels'])
            np.testing.assert_array_equal(raw['base_scores'],original['original'])
        records.append(d['result'])
        write(dest/'progress.json',{'n':len(records),'total':100,'seconds':time.time()-start,'last_task':spec['task']})
        print('TASK',arm,seed,split,len(records),spec['task'],round(time.time()-start),flush=True)
    assert len(records)==100 and reused==20
    write(dest/'complete.json',{'complete':True,'manifest_sha256':digest(ROOT/'manifest.json'),
        'arm':arm,'seed':seed,'split':split,'n':100,'old_reused':reused,'results':records,'seconds':time.time()-start})

def summarize():
    checked();out={'complete':True,'raw_audit_passed':True,'manifest_sha256':digest(ROOT/'manifest.json'),'splits':{}}
    for split in ['cross','within']:
        per={};base={}
        for arm in q.ARMS:
            for seed in range(3):
                name=f'{arm}_s{seed}';records={}
                if arm in q.ARMS[3:]:
                    d=read(ROOT/'static'/split/(name+'.json'))
                    assert d==verify_static(split,arm,seed)
                    for r in d['records']:
                        records[r['task']]=r['first_success_rank'];base[r['task']]=r['base_first_success_rank']
                else:
                    folder=ROOT/'closed'/split/name;d=read(folder/'complete.json')
                    assert d['complete'] and d['manifest_sha256']==digest(ROOT/'manifest.json')
                    for p in sorted(folder.glob('*/trace.json')):
                        r=q.audit_closed(p.parent)['result'];records[r['task']]=r['first_success'] or None
                assert len(records)==100
                per[name]={'counts':{f's{k}':sum(hit(r,k) for r in records.values()) for k in [1,3,5,9,20]},'first_success':records}
        assert len(base)==100
        basecounts={f's{k}':sum(hit(r,k) for r in base.values()) for k in [1,3,5,9,20]}
        assert basecounts['s9']==(47 if split=='cross' else 54)
        paired={}
        for left,right in [(q.ARMS[0],'base'),(q.ARMS[1],q.ARMS[3]),(q.ARMS[2],q.ARMS[4])]:
            tasks=sorted(base);delta=[]
            for task in tasks:
                delta.append(np.mean([int(hit(per[f'{left}_s{s}']['first_success'][task],9))-int(hit(base[task] if right=='base' else per[f'{right}_s{s}']['first_success'][task],9)) for s in range(3)]))
            delta=np.array(delta);rng=np.random.default_rng(20261007)
            boots=delta[rng.integers(100,size=(10000,100))].mean(1)
            paired[left+'_minus_'+right]={'s9_mean_delta':float(delta.mean()),'scene_bootstrap95':np.quantile(boots,[.025,.975]).tolist(),
                'scope':'Scene paired, seeds clustered; conditional on these models/templates, not new test'}
        out['splits'][split]={'n':100,'base':basecounts,'per_model':per,'paired':paired}
    write(ROOT/'SUMMARY.json',out);print('FULL_ABCDE_SUMMARY_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','eval','summary']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    q.torch.set_num_threads(4)
    if a.command=='prepare':prepare()
    elif a.command=='smoke':smoke()
    elif a.command=='eval':evaluate(a.index)
    else:summarize()
