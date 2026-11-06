import json, sys
import numpy as np
W = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))
R = {n: json.load(open(f'/path/to/phyre-unc/results/{n}.json')) for n in sys.argv[1:]}
for n, d in R.items():
    a = d['aggregate']
    print(f"{n:14} AUCCESS {a['auccess']:.4f}  " + '  '.join(f"S@{k} {v:.2f}" for k, v in a['success_at'].items()))


def paired(a, b):
    fa = {r['task']: r for r in R[a]['per_task']}; fb = {r['task']: r for r in R[b]['per_task']}
    ts = sorted(fa)
    da = np.array([fb[t]['auccess'] - fa[t]['auccess'] for t in ts])
    se = da.std(ddof=1) / np.sqrt(len(da))
    first = lambda r: r['first_success'] if r['first_success'] else 101
    better = sum(first(fb[t]) < first(fa[t]) for t in ts); worse = sum(first(fb[t]) > first(fa[t]) for t in ts)
    print(f'{b} vs {a}: dAUCCESS {da.mean():+.4f} (SE {se:.4f}, z={da.mean() / se:+.2f}); tasks solved earlier {better}, later {worse}, same {len(ts) - better - worse}')


for pair in [('clm_base', 'clm_v3_nofb'), ('clm_v3_nofb', 'clm_v3_fb'), ('clm_v1c_nofb', 'clm_v1c_fb'), ('clm_base', 'clm_v1c_fb'), ('clm_base', 'clm_v3_fb'), ('clm_base', 'clm_tta20'), ('clm_dqn_fixed', 'clm_dqn_online'), ('clm_dqnm_fixed', 'clm_dqnm_online'), ('clm_dqnm_online', 'clm_v1c_fb'), ('clm_dqn_online', 'clm_v1c_fb'), ('cx_clm_base', 'cx_clm_nofb'), ('cx_clm_nofb', 'cx_clm_fb'), ('cx_clm_base', 'cx_clm_fb'), ('w400_base', 'w400_v1c_nofb'), ('w400_v1c_nofb', 'w400_v1c_fb'), ('w400_v3_nofb', 'w400_v3_fb'), ('w400_base', 'w400_v1c_fb'), ('w400_base', 'w400_v3_fb'), ('w400_base', 'w400_tta20'), ('w400_dqnm_fixed', 'w400_dqnm_online'), ('w400_dqn_fixed', 'w400_dqn_online'), ('w400_dqnm_online', 'w400_v1c_fb'), ('cx_clm_base', 'cx_clm_fb_gateinf'), ('cx_clm_base', 'cx_clm_fb_gate'), ('cx_clm_base', 'cx_clm_tta20'), ('cx_clm_dqn_fixed', 'cx_clm_dqn_online'), ('cx_clm_dqnm_fixed', 'cx_clm_dqnm_online'), ('cx_clm_base', 'cx_clm_fb_gate_scale0.5'), ('cx_clm_base', 'cx_clm_fb_gate_scale0.25'), ('cx_clm_dqnm_fixed', 'cx_clm_base'), ('cx_clm_dqn_fixed', 'cx_clm_base')]:
    if all(p in R for p in pair):
        paired(*pair)
