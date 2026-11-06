import json
rows = [('base', 'results/eval_pred_direct_within_f0.json', 'results/eval_base_indep.json')] + \
       [(c, f'results/eval_sweep_{c}_main.json', f'results/eval_sweep_{c}_indep.json') for c in ['motion', 'aux0.1', 'aux1', 'big'] ]
print('config   | main probe: auc wt_auc green | indep probe: auc wt_auc green | indep green by time t=4,8,16,31')
for n, a, b in rows:
    try:
        A, B = json.load(open(a)), json.load(open(b))
    except FileNotFoundError:
        continue
    bt = B['green_err_pred_by_time_median_px']
    print(f"{n:8} | {A['auc_pred']:.3f} {A['within_task_auc_pred']:.3f} {A['green_err_pred_median_px']:5.1f} | "
          f"{B['auc_pred']:.3f} {B['within_task_auc_pred']:.3f} {B['green_err_pred_median_px']:5.1f} | {[bt[4], bt[8], bt[16], bt[31]]} (real auc {B['auc_real']:.3f})")
