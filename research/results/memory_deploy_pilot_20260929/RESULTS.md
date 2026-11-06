# Deployment memory pilot

Complete: True; A/B eligible: True

Training diagnostics are resubstitution; A/B is fixed-query, memory-only held-template validation.
Two fine-tuning seeds share one pretrained R0 initialization. No full adaptive cross evaluation.

| Model | Train nonempty scenes | Base AUC | Correct AUC | Wrong-scene AUC |
|---|---:|---:|---:|---:|
| R0 | 22 | 0.7425 | 0.8151 | 0.8086 |
| R1 | 22 | 0.7425 | 0.8087 | 0.7983 |
| R2 | 22 | 0.7425 | 0.8144 | 0.8004 |
| R3 | 22 | 0.7425 | 0.8031 | 0.7900 |
| R4 | 22 | 0.7425 | 0.8255 | 0.7927 |

| Model | Pilot policy-context AUC | Random-context AUC |
|---|---:|---:|
| unchanged R0 | 0.7849782147548127 | 0.7883517728737123 |
| A_s0 | 0.782522456416666 | 0.7835761880542486 |
| A_s1 | 0.7763158241611458 | 0.7843057691879571 |
| B_s0 | 0.7772979666207476 | 0.7825074707122393 |
| B_s1 | 0.7815118529592583 | 0.7838530431700137 |
