# Joint loss and support-transfer findings

## cross

No new raw-input loss met the predeclared S@9 gain without S@5 regression; no second training round launched. Use fixed-query and input controls to guide next work.
Matching centered-update magnitude did not recover a clear centered-policy advantage; a magnitude-only explanation is not established.
Action-distance gating did not beat every amplitude/shuffle/ungated first-only control; do not promote a locality architecture based on this proxy.

| Experiment | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| loss_seed0_counts/l1 | 3 | 7 | 10 | 12 |
| loss_seed0_counts/mse | 3 | 7 | 8 | 8 |
| loss_seed0_counts/smooth | 3 | 7 | 10 | 12 |
| loss_seed0_counts/l1_centered | 3 | 6 | 10 | 15 |
| loss_seed0_counts/l1_zero | 3 | 7 | 7 | 9 |
| loss_seed0_counts/mse_centered | 3 | 7 | 9 | 10 |
| loss_seed0_counts/mse_zero | 3 | 7 | 9 | 10 |
| loss_seed0_counts/smooth_centered | 3 | 9 | 12 | 15 |
| loss_seed0_counts/smooth_zero | 3 | 10 | 12 | 14 |
| transfer_counts/raw_first_only | 3 | 7 | 9 | 13 |
| transfer_counts/raw_full | 3 | 7 | 10 | 12 |
| transfer_counts/centered_first_only | 3 | 4 | 8 | 10 |
| transfer_counts/centered_full | 3 | 6 | 10 | 15 |
| transfer_counts/norm_matched_full | 3 | 8 | 10 | 14 |
| transfer_counts/action_gate_first | 3 | 7 | 9 | 9 |
| transfer_counts/shuffled_gate_first | 3 | 9 | 11 | 13 |
| transfer_counts/uniform_gate_first | 3 | 5 | 10 | 11 |

First-only querygate policies have a different feedback budget from recurrentpolicies; use their matchedfirst-only controls. Seed1 here replicates updater initialization/sampling, notQ0or fold.

## within

Round1 improvement did not replicate as a positive S@9 difference; no robust winner established.
Norm-matched raw-direction updates reached or exceeded centered S@9 in this pilot; magnitude is a plausible contributor, not a proved sole cause.
Action-distance association failed the predeclared directional-repeat screen; gate evaluation skipped. This does not rule out other measures of relevance.

| Experiment | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| loss_seed0_counts/l1 | 29 | 45 | 50 | 56 |
| loss_seed0_counts/mse | 29 | 42 | 50 | 56 |
| loss_seed0_counts/smooth | 29 | 46 | 51 | 61 |
| loss_seed0_counts/l1_centered | 29 | 46 | 51 | 57 |
| loss_seed0_counts/l1_zero | 29 | 41 | 49 | 55 |
| loss_seed0_counts/mse_centered | 29 | 42 | 52 | 56 |
| loss_seed0_counts/mse_zero | 29 | 39 | 49 | 54 |
| loss_seed0_counts/smooth_centered | 29 | 48 | 54 | 61 |
| loss_seed0_counts/smooth_zero | 29 | 43 | 53 | 57 |
| loss_seed1_counts/l1 | 29 | 45 | 49 | 58 |
| loss_seed1_counts/smooth | 29 | 40 | 47 | 50 |
| transfer_counts/raw_first_only | 29 | 41 | 48 | 53 |
| transfer_counts/raw_full | 29 | 45 | 50 | 56 |
| transfer_counts/centered_first_only | 29 | 43 | 48 | 55 |
| transfer_counts/centered_full | 29 | 46 | 51 | 57 |
| transfer_counts/norm_matched_full | 29 | 48 | 53 | 59 |

First-only querygate policies have a different feedback budget from recurrentpolicies; use their matchedfirst-only controls. Seed1 here replicates updater initialization/sampling, notQ0or fold.
