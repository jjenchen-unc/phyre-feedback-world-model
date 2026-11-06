# Support-to-query transfer investigation

All interventions use the frozen raw-trained128 updater, common Q0 first action and classifier1024 pool. No new updater training.

## cross

| Policy | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| raw_first_only | 3 | 7 | 9 | 13 |
| raw_full | 3 | 7 | 10 | 12 |
| centered_first_only | 3 | 4 | 8 | 10 |
| centered_full | 3 | 6 | 10 | 15 |
| norm_matched_full | 3 | 8 | 10 | 14 |
| action_gate_first | 3 | 7 | 9 | 9 |
| shuffled_gate_first | 3 | 9 | 11 | 13 |
| uniform_gate_first | 3 | 5 | 10 | 11 |

Matching centered-update magnitude did not recover a clear centered-policy advantage; a magnitude-only explanation is not established.
Action-distance gating did not beat every amplitude/shuffle/ungated first-only control; do not promote a locality architecture based on this proxy.

## within

| Policy | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| raw_first_only | 29 | 41 | 48 | 53 |
| raw_full | 29 | 45 | 50 | 56 |
| centered_first_only | 29 | 43 | 48 | 55 |
| centered_full | 29 | 46 | 51 | 57 |
| norm_matched_full | 29 | 48 | 53 | 59 |

Norm-matched raw-direction updates reached or exceeded centered S@9 in this pilot; magnitude is a plausible contributor, not a proved sole cause.
Action-distance association failed the predeclared directional-repeat screen; gate evaluation skipped. This does not rule out other measures of relevance.

## Limits

Small fixedhistory panel, samefold/seed; no robustdesignwinner fromsinglepilot
Oracle residualalignment andoraclealpha usequerytruth fordiagnosis only; neverdeploymentgates
Forcedhistories are notcounterfactual S9 rollouts
Normmatching separates updateamplitude fromrawdirection locally, notall nonlinear laterconsequences
Querygate isfirst-update-only; compare againstfirst-only controls, notclaimfair recurrentarchitecture superiority
Residualcosine/actiondistance associations do notprove causaltransfer
Lossiteration runs independently; integrate reports afterbothcomplete
