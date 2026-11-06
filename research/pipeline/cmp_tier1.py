import json, sys
D = [(f.split('/')[-1].replace('tier1_', '').replace('.json', ''), json.load(open(f))) for f in sys.argv[1:]]
for k in ['k1', 'k2', 'k4', 'scorer_fail']:
    for s in ['all', 'near', 'far']:
        if not D[0][1][k][s].get('n_queries'):
            continue
        print(f"{k:11} {s:4} n={D[0][1][k][s]['n_queries']}")
        for n, d in D:
            x = d[k][s]
            print(f"    {n:26} L1 {x['latent_l1']:.4f}  obj {x['object_latent_l1']:.4f}  green {x['green_err_median_px']:6.2f}px  rank_auc {x['rank_auc_mean']:.3f}")
