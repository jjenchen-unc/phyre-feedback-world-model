"""Second updater initialization/order seed; retain prediction and zero controls."""
import argparse,importlib.util,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import study

def bind(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
tr=bind('train',ROOT/'code/train.py');ev=bind('evaluate',ROOT/'code/evaluate.py');ev.ARMS=['prediction_residual','prediction_zero','pair_residual','pair_zero']
p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','train','discovery','confirmation']);p.add_argument('--index',type=int,default=0);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
if a.mode=='register':
 study.checked();first=study.read(ROOT/'ROUND1_SUMMARY_seed7919.json');assert first['candidates']['pair']['promising_vs_prediction']
 study.write(ROOT/'replication/PLAN.json',dict(parent=study.sha(ROOT/'PLAN.json'),source_sha=study.sha(Path(__file__)),seed=7920,arms=ev.ARMS,original_training_indices=[0,1,4,5],trigger='Pairwise early-rank gain over matched prediction baseline nonnegative across both partitions, but residual utility unresolved. Replicate with both zero and prediction controls; no claim pair beats BCE.',preserved='FrozenQ0, geometry/ranking probes, original scorer, fixeddataset8queries/scene, 128steps, same calibratedlossweight; fresh updater initialization and order seed7920; all4 share draws.',budget='Total12 updaterfits:6round1+2BCEcal+4replication. No additionalfitswithoutnewexplicit boundedplan.'))
else:
 plan=study.read(ROOT/'replication/PLAN.json');assert plan['source_sha']==study.sha(Path(__file__)) and plan['parent']==study.sha(ROOT/'PLAN.json')
 if a.mode=='train':tr.run(plan['original_training_indices'][a.index],plan['seed'])
 else:ev.run(a.mode,a.shard,8,plan['seed'])
