from transfer_core import *
a=read(R/'FINAL_SUMMARY.json');b=read(L/'FINAL_SUMMARY.json');assert a['complete'] and b['complete'];out={'complete':True,'support_transfer_sha256':sha(R/'FINAL_SUMMARY.json'),'loss_design_sha256':sha(L/'FINAL_SUMMARY.json'),'panels':{}};lines=['# Joint loss and support-transfer findings','']
for panel in ['cross','within']:
 loss=b['panels'][panel];tr=a['panels'][panel];out['panels'][panel]={'loss_interpretation':loss['interpretation'],'transfer_interpretation':tr['interpretation'],'loss_seed0_counts':{k:v['counts'] for k,v in loss['methods'].items()},'loss_seed1_counts':{k:v['counts'] for k,v in loss['seed1'].items()},'transfer_counts':{k:v['counts'] for k,v in tr['methods'].items()}}
 lines += [f'## {panel}','',loss['interpretation'],*tr['interpretation'],'','| Experiment | S@1 | S@3 | S@5 | S@9 |','|---|---:|---:|---:|---:|']
 for category in ['loss_seed0_counts','loss_seed1_counts','transfer_counts']:
  for k,v in out['panels'][panel][category].items():lines.append('| '+category+'/'+k+' | '+' | '.join(str(v[f's{j}']) for j in [1,3,5,9])+' |')
 lines+=['','First-only querygate policies have a different feedback budget from recurrentpolicies; use their matchedfirst-only controls. Seed1 here replicates updater initialization/sampling, notQ0or fold.','']
write(R/'JOINT_SUMMARY.json',out);(R/'JOINT_REPORT.md').write_text('\n'.join(lines)+'\n');print('JOINT_REPORT_COMPLETE')
