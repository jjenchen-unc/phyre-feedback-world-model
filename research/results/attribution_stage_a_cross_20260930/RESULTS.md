# Attribution stage A

Reused cross development set; same query and common base contexts. No training.

| Model | AUC | Overall latent L1 | Success surface MAE | Failure surface MAE |
|---|---:|---:|---:|---:|
| base | 0.56540 | 0.11501 | 32.41926 | 72.34720 |
| observed_latent | 0.97689 | 0.00000 | 6.13234 | 4.44273 |
| true_geometry | 0.99416 | NA | 0.00000 | 0.00000 |
| true_geometry_coarse_weighted | 0.99408 | NA | 0.00000 | 0.00000 |
| true_geometry_dense_weighted | 0.99405 | NA | 0.00000 | 0.00000 |
| initial_R0 | 0.50516 | 0.11421 | 41.23192 | 72.56279 |
| A_s0 | 0.44240 | 0.11402 | 49.87902 | 70.27725 |
| B_s0 | 0.44351 | 0.11401 | 48.83023 | 70.44604 |
| A_s1 | 0.45362 | 0.11390 | 48.01468 | 70.79763 |
| B_s1 | 0.45355 | 0.11386 | 47.89049 | 70.90311 |

Scene/class denominators and matched K4 sensitivity: SUMMARY.json. Geometry contact metrics are threshold proxies, not simulator success.
