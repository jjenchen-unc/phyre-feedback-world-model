"""Expand evaluation only to remaining80 existing cross scenes; no new fitting."""
import argparse,importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cross_transfer_original',HERE/'experiment.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
original_checked=m.checked;original_write=m.write

def register():
    base=original_checked();alltasks=m.st.ar.checked()['test'];remaining=[t for t in alltasks if t not in base['tasks']]
    assert len(alltasks)==100 and len(remaining)==80
    p=dict(parent_plan=m.sha(HERE/'PLAN.json'),source_sha=m.sha(Path(__file__)),tasks=remaining,all_tasks=alltasks,
      trigger='Second updater seed did not reproduce broad improvement across both training-template partitions. Expand fixed-method transfer evaluation to all existing100crossscenes to reduce tiny-panel uncertainty; do not change any trainedmethod.',
      fits_added=0,original_small_panel=base['tasks'],reuse='Reuse20originalpanel records; only80new scenes. Same immutableQ0 pools/cachedtruefirstsupport; exactQ0score replay, all16 scoremethods unchanged.',
      limits='Onefold/fiveunseen-trainingtemplates andtwo updaterseeds, sharedbase. Existingtestpanelhasbeenusedhistorically. Stillonefeedbackupdate andstaticnextk; notfullrepeated-feedbackS9.')
    assert not(HERE/'EXTENSION_PLAN.json').exists();original_write(HERE/'EXTENSION_PLAN.json',p)

def check():
    base=original_checked();p=m.read(HERE/'EXTENSION_PLAN.json');assert p['parent_plan']==m.sha(HERE/'PLAN.json') and p['source_sha']==m.sha(Path(__file__))
    return dict(base,tasks=p['tasks'])

def write(path,value):
    if isinstance(value,dict):value=dict(value,extension_plan=m.sha(HERE/'EXTENSION_PLAN.json'))
    original_write(path,value)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','pack','evaluate']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=32);a=p.parse_args()
    if a.mode=='register':register()
    else:
        m.checked=check;m.write=write
        if a.mode=='pack':m.pack(a.shard,a.nshards)
        else:m.evaluate(a.shard,a.nshards)
