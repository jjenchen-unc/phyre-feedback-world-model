> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Deployment-distribution memory pilot results

Completed 2026-09-29 UTC (2026-09-28 evening America/New_York). Registration:
[MEMORY_DEPLOY_PILOT_RUN.md](MEMORY_DEPLOY_PILOT_RUN.md). Run artifacts:
`results/memory_deploy_pilot_20260929/` on local and shared oprime project.

The specified base/scorer collection produced new data, but this pilot does
not show a consistent benefit from training with its failed contexts instead
of random-policy failed contexts. This is not evidence of equivalence or a
general rejection of deployment-distribution training.

## Completed experiment

All four models completed 2,000 memory fine-tuning steps, 16,000 query examples
each. A uses random-policy failed contexts; B uses base/scorer-policy failed
contexts. Both share the same pretrained R0 initialization, query draws, actual
context sizes, 50% legacy replay schedule, plain latent L1, and frozen base/probe.
Two seeds vary fine-tuning, not initial pretraining. Per-seed A/B counts match.
The 4,000-draw pairing audit passed. Collection yielded 3,555 trajectories from
76 scenes; 51 train and 12 validation scenes have matched nonempty prefixes.

## Primary fixed-query result

Both arms evaluated under the SAME base-policy contexts. Each scene receives
equal weight after averaging its prefixes. AUC has **7 valid scenes of 12**:
the other five contain only failed queries and remain in latent-error metrics.
The seven scenes span four memory-held templates with counts 3/1/2/1.
These templates were seen by the frozen base/probe. There are 70 validation
prefixes total, but these are not 70 independent scenes.

| Model | Policy-context ranking AUC | Mean latent L1 (12 scenes) |
|---|---:|---:|
| Base / empty memory | 0.82442 | 0.087097 |
| Unchanged R0 | 0.78498 | 0.085901 |
| A seed 0 | 0.78252 | 0.086175 |
| B seed 0 | 0.77730 | 0.086282 |
| A seed 1 | 0.77632 | 0.086279 |
| B seed 1 | 0.78151 | 0.086327 |

B minus A is **−0.00522** in seed 0 and **+0.00520** in seed 1. The two-seed
mean is −0.000014. All four fine-tuned models have slightly lower policy-context
AUC and higher latent L1 than unchanged R0 on this panel. These are descriptive
sample results, not a statistically established harm or equivalence claim.

## A concentrated failure and the limits of interpretation

On `00012:076`, all 32 query scores under correct policy context are **exactly
zero**, for all ten prefixes, in unchanged R0 and every fine-tuned model.
This scene has one successful query. Its AUC drops from base 0.87097 to 0.5
because all predictions tie. A post-hoc sensitivity check omitting this scene
changes policy-memory minus base from negative to positive in every model
(+0.0057 to +0.0158). The primary metric still includes it; this sensitivity
shows concentration, not a justification for excluding a difficult scene.

The saved stable-sort first-success rank changes from 5 to 4 despite complete
score collapse: rank 4 is inherited query order, not evidence of useful
discrimination. Query construction includes terminal successes and high-score
actions, so arbitrary tie ordering is particularly uninformative here.

Wrong-scene AUC is 0.8492 for R0 and 0.8590–0.8622 for fine-tuned models, above
correct-context AUC on this panel. Donors match K and prefer another template;
they are a fixed, limited control. Much of the contrast comes from avoiding
zero-score collapse in `00012:076`. This does not establish that wrong memory
is generally useful or that memory is ignored.

The scorer implements a tanh-based soft contact sum (`redesign_lib.py`,
`readout`). Saturation is a plausible explanation for exact zeros, but the
saved scores do not distinguish numerical saturation from a substantial
predicted geometry error. Before another training wave, inspect predicted
distances, pre-transform margins and stable high-precision scores for this
case; extend that audit to all seven evaluable scenes. Do not change the
primary metric retroactively. No new diagnostic model forwards were launched
as part of this status check.

## What this adds to the earlier evidence

Training-scene diagnostics show R0 AUC 0.7425 → 0.8151, but wrong-scene context
also reaches 0.8086. This is resubstitution and does not establish specific
evidence use or generalization. New collection has 98.76% of 404 base-policy
training failures outside the old encoded union; this demonstrates different
coverage, not that all relevant deployment mismatch has been eliminated.
The initial-only collection encoding and packed full-clip first token are
not bit-identical (228-sample mean absolute latent difference 0.002364,
maximum 0.876953). Both A/B arms share conventions, preserving the narrow
comparison, but exact historical deployment alignment is not established.

This pilot changes the **memory support source**. It does not post-train the
base world model, isolate pure on-policy query sampling, refresh collection
with the learned memory policy, or evaluate adaptive cross success/AUCCESS.
The tiny conditional panel and shared initialization limit conclusions.

## Verification

`pipeline/verify_deploy_pilot_results.py` independently recomputed pairwise
win/tie AUC, stable first-success ranks, and per-scene aggregates from saved
per-query arrays for all **1,750 rows** in five evaluations. Query IDs, labels,
context IDs and actual K match across models; base/empty scores and latent
errors match exactly. All checks pass; output `RESULTS_VERIFICATION.json`.
Remote completion records verify 2,000 steps for all arms. Raw result hashes
are recorded in the verification output. No additional training was launched.
