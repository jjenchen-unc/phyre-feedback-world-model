"""Matched validation losses for unchanged old checkpoints and contact-target fits."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('contact_metrics',Path(__file__).with_name('experiment.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
import torch
c.check();team=c.st.team_new();cal=c.read(c.ROOT/'DATA_AND_CALIBRATION.json')
arms={'q0':None}
for arm in ['prediction_residual','prediction_zero','bce_residual','pair_residual']:
    arms[arm]=c.ROOT/'models/seed7919'/arm
for arm in c.ARMS:arms[arm]=c.RR/'models'/arm
records={};reader=c.st.ar.lr.source().old.reader_hash(team.model)
try:
    with torch.no_grad():
        for stage in ['discovery','confirmation']:
            records[stage]={}
            for task in c.st.checked()[stage]:
                row=c.st.data(task)
                if not row['failed']:continue
                records[stage][task]={}
                for arm,folder in arms.items():
                    if folder:
                        ck=c.read(folder/'complete.json');assert ck['checkpoint_sha']==c.sha(folder/'updater.pt')
                        missing,extra=team.model.load_state_dict(torch.load(folder/'updater.pt',map_location='cpu',weights_only=True),strict=False)
                        assert not extra and all(n.startswith('base.') or n=='initial_tokens' for n in missing)
                    team.model.eval().requires_grad_(False);team.model.checkpoint_blocks=False
                    team.input_mode='zero' if arm.endswith('zero') else 'centered'
                    state=team.model.initial_state() if arm=='q0' else c.tr.state_for(team,row)
                    ls,pred,_=c.tr.terms(team,row,state,cal,grad=False)
                    values={k:float(v) for k,v in ls.items()};values['contact']=float(c.contact_loss(team,pred,row))
                    records[stage][task][arm]=values
                print('LOSS_CONTROL',stage,task,flush=True)
    assert reader==c.st.ar.lr.source().old.reader_hash(team.model)
    c.write(c.RR/'LOSS_CONTROLS.json',dict(complete=True,source_sha=c.sha(Path(__file__)),plan=c.sha(c.RR/'PLAN.json'),reader_unchanged=True,records=records))
finally:c.st.ar.lr.close(team)
