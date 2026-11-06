"""Explicit local-module binding after inherited runtime mutates sys.path.

Original training, evaluation code, PLAN and checkpoints stay immutable.
Only resolve this study's train/evaluate modules instead of older same-named files.
"""
import importlib.util,json,sys,runpy,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;C=R/'code'
sys.path.insert(0,str(C))
import study
study.checked()
def bind(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
bind('train',C/'train.py')
bind('evaluate',C/'evaluate.py')
print('EXECUTION_MODULE_BINDING',json.dumps({n:sys.modules[n].__file__ for n in ['study','train','evaluate']}),flush=True)
mode=sys.argv[1];assert mode in ['evaluate','summarize'];sys.argv=[str(C/(mode+'.py'))]+sys.argv[2:]
runpy.run_path(str(C/(mode+'.py')),run_name='__main__')
