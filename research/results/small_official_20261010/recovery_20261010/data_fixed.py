"""Official task selection, initial scenes and action outcomes. CPU PHYRE env."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import phyre
from phyre import simulator as S
R=Path(__file__).resolve().parents[1];PROJECT=R.parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def stable(x):return hashlib.sha256(str(x).encode()).hexdigest()
def select():
 assert not (R/'selection.json').exists()
 selections={}
 for panel,setup,fold in [('cross','ball_cross_template',1),('within','ball_within_template',0)]:
  tr,dv,te=phyre.get_fold(setup,fold);templates=sorted({t[:5] for t in te});per=100//len(templates)
  tasks=sorted(t for tp in templates for t in sorted([t for t in te if t.startswith(tp+':')],key=lambda t:stable(('small_official_20261010',panel,t)))[:per])
  assert len(tasks)==100 and not set(tasks)&set(tr)
  selections[panel]={'setup':setup,'fold':fold,'train':list(tr),'dev':list(dv),'test':list(te),'selected':tasks,'templates':templates}
 write(R/'selection.json',{'panels':selections,'rule':'fixed SHA-stratified task sampling before model scores','seed_tag':'small_official_20261010'})
 print({k:{'train':len(v['train']),'test':len(v['test']),'selected':len(v['selected'])} for k,v in selections.items()},flush=True)

def candidate_one(item):
 panel,task=item;out=R/'assets'/panel/'candidates'/(task.replace(':','_')+'.npz')
 if out.exists():return task,'existing'
 cache=phyre.get_default_100k_cache('ball');acts=cache.action_array[:5000];states=cache.load_simulation_states(task)[:5000]
 sim=phyre.initialize_simulator([task],'ball');frames=[];features=[];ids=[]
 for i in np.flatnonzero(states!=0):
  ui,ok=sim._action_mapper.action_to_user_input(acts[i]);assert ok
  if S.check_for_occlusions(sim._serialized[0],ui,keep_space_around_bodies=sim._keep_spaces):continue
  scene=S.add_user_input_to_scene(sim._tasks[0].scene,ui,sim._keep_spaces);s0=S.simulate_scene(scene,1)[0]
  frames.append(S.scene_to_raster(s0).astype(np.uint8));features.append(S.scene_to_featurized_objects(s0).features[0]);ids.append(int(i))
 assert len(ids)>=1024,(task,len(ids))
 out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
 np.savez_compressed(tmp,frames=np.stack(frames),features=np.stack(features).astype(np.float32),action=acts[ids].astype(np.float32),action_idx=ids)
 tmp.replace(out);return task,len(ids)

def labels_one(item):
 panel,task=item;folder=R/'assets_fixed'/panel/'shortlist';src=folder/(task.replace(':','_')+'.npz');out=folder/(task.replace(':','_')+'_labels.npy')
 if out.exists():return task,'existing'
 with np.load(src) as z:actions=z['action'];aids=z['action_idx']
 assert len(actions)==1024
 sim=phyre.initialize_simulator([task],'ball');status=np.array([int(sim.simulate_action(0,a,need_images=False).status) for a in actions],dtype=np.int8)
 assert (status!=0).all(),task
 temp=out.with_suffix('.tmp.npy');np.save(temp,status);temp.replace(out)
 return task,int((status>0).sum())

def collection(mode):
 sel=read(R/'selection.json');items=[(p,t) for p,v in sel['panels'].items() for t in v['selected']]
 with Pool(12) as pool:
  for i,v in enumerate(pool.imap_unordered(candidate_one if mode=='candidates' else labels_one,items)):
   print(mode,i+1,v,flush=True)
 write(R/(mode.upper()+'_FIXED_COMPLETE.json'),{'complete':True,'tasks':200})

def serve():
 last=None;sim=None
 for line in sys.stdin:
  req=json.loads(line);task=req['task'];action=np.asarray(req['action'],np.float32);path=Path(req['out'])
  if task!=last:sim=phyre.initialize_simulator([task],'ball');last=task
  status=int(sim.simulate_action(0,action,need_images=False).status);assert status!=0
  ui,ok=sim._action_mapper.action_to_user_input(action);assert ok
  scene=S.add_user_input_to_scene(sim._tasks[0].scene,ui,sim._keep_spaces);states=S.simulate_scene(scene,1017)[::8]
  frames=np.stack([S.scene_to_raster(s) for s in states]).astype(np.uint8)
  features=np.concatenate([S.scene_to_featurized_objects(s).features for s in states]).astype(np.float32)
  assert len(frames)==128
  path.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(path,frames=frames,features=features,action=action,status=status,task=task)
  print(json.dumps({'ok':True,'out':str(path),'status':status}),flush=True)

if __name__=='__main__':
 command=sys.argv[1]
 if command=='select':select()
 elif command=='serve':serve()
 else:collection(command)
