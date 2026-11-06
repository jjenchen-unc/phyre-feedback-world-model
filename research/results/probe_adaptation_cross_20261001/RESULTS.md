# Probe adaptation results

Reused cross D0; frozen base and R0; common K1/query; fixed final checkpoints.
AUC and success geometry use AUC/success-defined scenes; all denominators in SUMMARY.json.

| Probe | Input | AUC | Success surface MAE (px) |
|---|---|---:|---:|
| original | base | 0.56540 | 32.419 |
| true_control_s0 | base | 0.56063 | 33.224 |
| predicted_mix_s0 | base | 0.68949 | 32.269 |
| initial_only_s0 | base | 0.58094 | 34.218 |
| true_control_s1 | base | 0.56345 | 33.075 |
| predicted_mix_s1 | base | 0.73084 | 31.478 |
| initial_only_s1 | base | 0.57907 | 34.545 |
| true_control_s2 | base | 0.56110 | 32.822 |
| predicted_mix_s2 | base | 0.70987 | 32.646 |
| initial_only_s2 | base | 0.58011 | 34.852 |
| original | memory | 0.50516 | 41.232 |
| true_control_s0 | memory | 0.50201 | 42.579 |
| predicted_mix_s0 | memory | 0.60887 | 38.678 |
| initial_only_s0 | memory | 0.53673 | 39.346 |
| true_control_s1 | memory | 0.50585 | 42.435 |
| predicted_mix_s1 | memory | 0.64902 | 37.998 |
| initial_only_s1 | memory | 0.53735 | 39.296 |
| true_control_s2 | memory | 0.50364 | 41.968 |
| predicted_mix_s2 | memory | 0.62864 | 39.314 |
| initial_only_s2 | memory | 0.51775 | 40.482 |
| original | observed | 0.97689 | 6.132 |
| true_control_s0 | observed | 0.97496 | 6.208 |
| predicted_mix_s0 | observed | 0.97646 | 5.986 |
| initial_only_s0 | observed | 0.98732 | 13.745 |
| true_control_s1 | observed | 0.97849 | 6.076 |
| predicted_mix_s1 | observed | 0.97251 | 6.297 |
| initial_only_s1 | observed | 0.98381 | 14.707 |
| true_control_s2 | observed | 0.97669 | 6.010 |
| predicted_mix_s2 | observed | 0.97432 | 6.462 |
| initial_only_s2 | observed | 0.98301 | 13.450 |
| original | initial | 0.53880 | 136.186 |
| true_control_s0 | initial | 0.56450 | 135.968 |
| predicted_mix_s0 | initial | 0.60963 | 137.432 |
| initial_only_s0 | initial | 0.41877 | 67.155 |
| true_control_s1 | initial | 0.59057 | 135.736 |
| predicted_mix_s1 | initial | 0.53900 | 136.968 |
| initial_only_s1 | initial | 0.43771 | 67.930 |
| true_control_s2 | initial | 0.59958 | 135.644 |
| predicted_mix_s2 | initial | 0.60926 | 136.725 |
| initial_only_s2 | initial | 0.41657 | 66.238 |

Three probe seeds share the same pretrained head/base/memory. No new-template population or closed-loop success claim.
A positive readout intervention supports a remediable readout bottleneck; it does not uniquely prove latent distribution shift or physical correctness.
