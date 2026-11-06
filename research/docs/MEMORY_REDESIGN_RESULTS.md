> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# R0–R4 screening results

All five variants completed 20,000 training steps, the 64-scene fixed-query
panel, and 100 cross-f1 development tasks. All variants have lower observed
AUCCESS than the matched base; R1–R4 are also below the new R0 reference.
This is an observed ordering at one seed/budget, not an established ordering
of methods. See [the interpretation review](MEMORY_REDESIGN_ANALYSIS_REVIEW.md)
for confounded interventions, uncertainty, and corrections to causal claims.

| Method | AUCCESS | Success@10 | Success@100 | 00014 solved /25 |
|---|---:|---:|---:|---:|
| Base, no memory | .4541 | 47% | 62% | 17 |
| R0 mixed-context reference | .4420 | 46% | 61% | 16 |
| R1 failed contexts | .4277 | 45% | 58% | 13 |
| R2 grouped queries + balanced loss | .4201 | 41% | 60% | 14 |
| R3 R1 + R2 | .4233 | 44% | 57% | 12 |
| R4 R3 + ranking loss | .3962 | 42% | 51% | 6 |

The matched baseline is `results/cx_clm_base.json` (.4541), which uses the
same closed-loop evaluator. The older `cl_cross_f1_base.json` (.4565) is a
different evaluation output and is not substituted here. Task IDs and
shortlist solution counts agree across all five runs and the matched base.
All methods have identical success@1 (27%), consistent with empty gating.

Relative to base, R0 gains/loses 0/1 solved tasks; R1 0/4; R2 1/3;
R3 0/5; R4 2/13. R4 improves template 00002 AUCCESS but loses heavily on
00014. A global aggregate alone would hide this transfer failure.

On the memory-only holdout, 29 of 64 scenes have nonempty failed contexts.
For these scenes at requested K=1, mean ranking AUC is:

| Method | Base | Correct context | Wrong-scene context |
|---|---:|---:|---:|
| R0 | .6697 | .6554 | .6546 |
| R1 | .6697 | .6571 | .6468 |
| R2 | .6697 | .6397 | .6349 |
| R3 | .6697 | .6444 | .6545 |
| R4 | .6697 | .6571 | .6512 |

Correct evidence has lower mean ranking AUC than base in this panel. Its
observed advantage over wrong-scene evidence is small or inconsistent. All
five correct-minus-wrong conditional resampling intervals include zero, and
25/29 donors are from the same template. This control is inconclusive about
evidence specificity; it does not establish that the model lacks useful
scene-specific corrections. This is also not the unseen cross-template panel.

Across the full 64-scene panel, every variant increases successful-query
latent L1 while reducing failed-query latent L1. At K=1 the base values are
.083710/.087006 (success/failure); R4 gives .084397/.086663. These are aggregate
latent-error changes, not a demonstrated cause of cross ranking loss. Failed
query error actually increases in 19/29 nonempty R3 scenes despite its mean
decrease. Successful-query L1 increases in 29/29 scenes for R0–R3 and 28/29
for R4, so that descriptive finding survives equal scene weighting.
Base and empty-memory scores match exactly in every saved panel.

Before an architecture rewrite, separate actual context-size effects, support
selection, query sampling and class weighting, and quantify seed/template
variation. The current model already uses query-conditioned memory attention;
the prior recommendation to add it was imprecise. Nonempty-context gating or
explicit retrieval would be distinct, still unverified changes. Larger-pool
policy traces and multiple held-out template splits remain useful controls.
No additional training wave has been launched.

Scope: one seed, bounded cached support reservoirs, memory-only internal
holdout (the base/probe saw those templates), and reused cross development
tasks. No statistical significance or final-test generalization is claimed.
R0 also differs from historical memory training in data union and batching,
so its improvement over historical memory is not attributable to one factor.

Verification: all 500 per-task AUCCESS values independently recomputed;
task/pool alignment and all empty-gate scores checked; eleven remote/local
result-file SHA256 hashes matched. See
`results/memory_redesign_20260928/PAIRED_ANALYSIS.json`, `SUMMARY.json`, and
`verified_remote_sha256.json`. Frozen code and registration are preserved.
