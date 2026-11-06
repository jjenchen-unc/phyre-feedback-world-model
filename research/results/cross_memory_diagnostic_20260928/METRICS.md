# Fixed-support diagnostic results

Ranks below exclude the K support actions. Lower ranks/errors are better; these are frozen-ranking diagnostics, not closed-loop success rates.

## 00014:273, K=1

| Condition | First success rank | AUC | All latent L1 | Success latent L1 | All green error px | Success green error px |
|---|---:|---:|---:|---:|---:|---:|
| base | 2 | 0.7827 | 0.11663 | 0.13109 | 63.48 | 77.39 |
| empty_original | 288 | 0.2436 | 0.11510 | 0.13935 | 68.86 | 110.46 |
| original_full | 367 | 0.1965 | 0.11475 | 0.13862 | 77.87 | 104.71 |
| gated_full | 350 | 0.2275 | 0.11468 | 0.13900 | 72.61 | 109.30 |
| gated_scale025 | 1 | 0.6568 | 0.11598 | 0.13363 | 65.12 | 83.05 |
| action_outcome_only | 13 | 0.6043 | 0.11526 | 0.13391 | 67.19 | 87.40 |
| no_residual | 312 | 0.2377 | 0.11514 | 0.13862 | 72.06 | 107.87 |
| no_video | 280 | 0.3047 | 0.11443 | 0.13632 | 76.30 | 93.84 |
| wrong_scene_same_template | 340 | 0.2138 | 0.11486 | 0.13777 | 79.70 | 104.20 |
| wrong_scene_other_template | 317 | 0.3346 | 0.11552 | 0.13490 | 88.47 | 83.16 |
| observed_latents | 1 | 1.0000 | 0.00000 | 0.00000 | 1.76 | 1.30 |

## 00014:273, K=2

| Condition | First success rank | AUC | All latent L1 | Success latent L1 | All green error px | Success green error px |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 0.7835 | 0.11659 | 0.13109 | 63.47 | 77.39 |
| empty_original | 287 | 0.2439 | 0.11506 | 0.13935 | 68.84 | 110.46 |
| original_full | 346 | 0.1949 | 0.11477 | 0.13864 | 78.47 | 103.62 |
| gated_full | 367 | 0.2182 | 0.11462 | 0.13905 | 74.02 | 109.48 |
| gated_scale025 | 2 | 0.6519 | 0.11593 | 0.13363 | 65.88 | 82.80 |
| action_outcome_only | 20 | 0.5724 | 0.11527 | 0.13397 | 68.14 | 88.08 |
| no_residual | 297 | 0.2381 | 0.11514 | 0.13869 | 73.58 | 108.17 |
| no_video | 218 | 0.3301 | 0.11434 | 0.13658 | 74.60 | 93.72 |
| wrong_scene_same_template | 336 | 0.2129 | 0.11479 | 0.13771 | 79.70 | 104.68 |
| wrong_scene_other_template | 305 | 0.3424 | 0.11540 | 0.13493 | 88.38 | 83.43 |
| observed_latents | 1 | 1.0000 | 0.00000 | 0.00000 | 1.76 | 1.30 |

## 00014:407, K=1

| Condition | First success rank | AUC | All latent L1 | Success latent L1 | All green error px | Success green error px |
|---|---:|---:|---:|---:|---:|---:|
| base | 4 | 0.7049 | 0.11902 | 0.13032 | 67.09 | 119.28 |
| empty_original | 311 | 0.2250 | 0.11838 | 0.13645 | 59.65 | 144.35 |
| original_full | 182 | 0.2648 | 0.11744 | 0.13400 | 66.44 | 131.91 |
| gated_full | 141 | 0.2905 | 0.11738 | 0.13388 | 61.03 | 133.92 |
| gated_scale025 | 15 | 0.5810 | 0.11843 | 0.13177 | 64.34 | 122.03 |
| action_outcome_only | 72 | 0.4790 | 0.11814 | 0.13246 | 65.14 | 125.05 |
| no_residual | 147 | 0.2997 | 0.11789 | 0.13405 | 61.47 | 132.66 |
| no_video | 270 | 0.3376 | 0.11777 | 0.13283 | 68.72 | 131.44 |
| wrong_scene_same_template | 133 | 0.3169 | 0.11776 | 0.13323 | 62.88 | 132.61 |
| wrong_scene_other_template | 83 | 0.4870 | 0.11880 | 0.13274 | 74.54 | 125.59 |
| observed_latents | 1 | 1.0000 | 0.00000 | 0.00000 | 1.68 | 0.99 |

## 00014:407, K=4

| Condition | First success rank | AUC | All latent L1 | Success latent L1 | All green error px | Success green error px |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 0.7070 | 0.11897 | 0.13032 | 67.03 | 119.28 |
| empty_original | 309 | 0.2256 | 0.11835 | 0.13645 | 59.59 | 144.35 |
| original_full | 182 | 0.2630 | 0.11750 | 0.13420 | 66.15 | 132.08 |
| gated_full | 152 | 0.2802 | 0.11741 | 0.13407 | 60.60 | 134.49 |
| gated_scale025 | 14 | 0.5785 | 0.11839 | 0.13182 | 64.16 | 122.10 |
| action_outcome_only | 67 | 0.4620 | 0.11813 | 0.13235 | 64.82 | 125.60 |
| no_residual | 149 | 0.2980 | 0.11789 | 0.13404 | 61.24 | 132.81 |
| no_video | 235 | 0.3586 | 0.11778 | 0.13338 | 69.10 | 130.06 |
| wrong_scene_same_template | 143 | 0.3019 | 0.11766 | 0.13325 | 62.40 | 133.25 |
| wrong_scene_other_template | 82 | 0.4947 | 0.11864 | 0.13279 | 73.76 | 125.64 |
| observed_latents | 1 | 1.0000 | 0.00000 | 0.00000 | 1.68 | 0.99 |

Validation: base ranks reproduced; candidate IDs unique; support failures excluded; outputs finite; ranks/AUC/latent means independently recomputed; checkpoint and diagnostic script hashes identical across tasks.
