import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];R=P/'results/support_transfer_iteration_20261019';L=P/'results/loss_design_iteration_20261019';B=P/'results/query_balance_training_20261019';C=P/'results/query_balance_centered_eval_20261019';S=P/'results/q0_first_feedback_20261018'
sys.path.insert(0,str(L/'code'));import loss_runtime as lr
read=lr.read;sha=lr.sha;write=lr.write

def team(panel):
 t=lr.make_team(panel);lr.load(t,panel);t.truth.parents.insert(0,L/'observations'/panel);t.truth.root=R/'observations'/panel;return t

def checked():
 p=read(R/'PLAN.json')
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 return p

def metrics(first):return {'n':len(first),'counts':{f's{k}':sum(v is not None and v<=k for v in first.values()) for k in [1,3,5,9]},'first_success':first}

def paired(a,b):
 assert set(a['first_success'])==set(b['first_success']);out={}
 for k in [1,3,5,9]:
  hit=lambda d,t:d['first_success'][t] is not None and d['first_success'][t]<=k
  gains=[t for t in a['first_success'] if hit(a,t) and not hit(b,t)];losses=[t for t in a['first_success'] if hit(b,t) and not hit(a,t)];out[str(k)]={'gains':gains,'losses':losses,'net':len(gains)-len(losses)}
 return out
