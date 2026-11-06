# Memory geometry-loss intervention

Fixed original training probe; paired R0 initialization/data/steps. No closed-loop result.

| Model | Readout | AUC | Successful surface MAE (px) |
|---|---|---:|---:|
| latent_only_s0 | original | 0.51761 | 39.518 |
| latent_only_s0 | predicted_mix_s0 | 0.62068 | 37.645 |
| latent_only_s0 | predicted_mix_s1 | 0.65799 | 36.879 |
| latent_only_s0 | predicted_mix_s2 | 0.63979 | 38.121 |
| latent_geometry_s0 | original | 0.55681 | 33.853 |
| latent_geometry_s0 | predicted_mix_s0 | 0.65460 | 34.449 |
| latent_geometry_s0 | predicted_mix_s1 | 0.68816 | 33.774 |
| latent_geometry_s0 | predicted_mix_s2 | 0.67431 | 34.543 |
| latent_only_s1 | original | 0.51315 | 40.048 |
| latent_only_s1 | predicted_mix_s0 | 0.62475 | 37.673 |
| latent_only_s1 | predicted_mix_s1 | 0.66249 | 36.924 |
| latent_only_s1 | predicted_mix_s2 | 0.64256 | 38.157 |
| latent_geometry_s1 | original | 0.55470 | 33.657 |
| latent_geometry_s1 | predicted_mix_s0 | 0.65787 | 34.655 |
| latent_geometry_s1 | predicted_mix_s1 | 0.69530 | 33.953 |
| latent_geometry_s1 | predicted_mix_s2 | 0.67888 | 34.793 |
| latent_only_s2 | original | 0.51224 | 40.931 |
| latent_only_s2 | predicted_mix_s0 | 0.62188 | 38.203 |
| latent_only_s2 | predicted_mix_s1 | 0.65971 | 37.439 |
| latent_only_s2 | predicted_mix_s2 | 0.64150 | 38.744 |
| latent_geometry_s2 | original | 0.55655 | 34.548 |
| latent_geometry_s2 | predicted_mix_s0 | 0.66161 | 35.119 |
| latent_geometry_s2 | predicted_mix_s1 | 0.69669 | 34.474 |
| latent_geometry_s2 | predicted_mix_s2 | 0.68317 | 35.264 |

Task-latent continuation criterion passed: True. See SUMMARY.json for every denominator and paired/template effect.
