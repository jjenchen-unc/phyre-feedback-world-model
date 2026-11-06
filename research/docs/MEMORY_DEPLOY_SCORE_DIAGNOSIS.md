> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Deployment memory scorer diagnosis

2026-09-29. **Completed:** Slurm array33987 and summary33988 produced all
five model outputs. No jobs remain queued/running for this project. No training
or live scorer change. Original pilot metrics remain unchanged.

## Completed replay changes the interpretation

All 1,400 replay rows exactly reproduce original scalar scores. Independent
NumPy recomputation from saved distances reproduces stable scores and scene
macro AUC; six raw/summary file hashes match local/remote copies. See
`results/memory_deploy_score_diagnostic_20260929/VERIFICATION.json`.

With identical predicted distances, changing only the final score arithmetic
reverses the observed memory-versus-base mean AUC ordering on this panel:

| Model | Original tanh32 AUC | Stable float64 / log-space AUC |
|---|---:|---:|
| Base | 0.82442 | 0.83907 |
| Original R0, policy context | 0.78498 | 0.85406 |
| A seed 0 | 0.78252 | 0.86763 |
| B seed 0 | 0.77730 | 0.86240 |
| A seed 1 | 0.77632 | 0.84402 |
| B seed 1 | 0.78151 | 0.84739 |

These are the same seven AUC-defined scenes, conditional fixed queries;
no updated cross AUCCESS or adaptive success result. Base also receives the
stable transform. B−A remains inconsistent: −0.00522 / +0.00336. Fine-tuned
models are now above unchanged R0 at seed0 and below at seed1; the original
claim that every fine-tune underperforms R0 is specific to the old scorer.

For 00012:076, all models' policy-context AUC becomes 0.93548 (success rank3)
from the all-tied0.5; base remains0.87097 (rank5). Yet geometry remains poor:
for original R0 at prefix1, successful-query minimum surface distance is
91.77px, versus base5.24px and true2.59px. Its policy scores lie between
2.76e−79 and5.33e−35. Numerical collapse and geometric prediction error
coexist; recovering relative ranking does not validate predicted physics.

Across seven successful-query scenes, base surface-distance MAE is25.97px;
R0 is31.74px, fine-tunes31.85–32.61px. Future-time green-position error rises
from31.61px to34.91px /35.03–35.76px. The twelve-scene failed-query mean errors
improve. Different class denominators must not be conflated.

True-geometry and observed-latent probe AUC are1.0 in all seven AUC-defined
scenes. This rules out universal failure to read true trajectories on this
panel, not probe mismatch on predicted latents. Wrong-scene stable AUC remains
higher than correct-context AUC for all five models; specific evidence use is
still not established. Float32 sigmoid alone is not fully sufficient: R0
policy AUC0.84069 versus float64/log0.85406. Preserve precision through sorting.

Recommended next step: evaluate a stable scorer consistently for base and
memory in the full closed loop; this has NOT been launched or implemented in
the live policy. Candidate collection and historical training also used the
old scoring path, so post-hoc reranking is not a complete end-to-end correction.

## Earlier saved-output audit (retained as background)

## Verified from existing outputs

Across original R0 and all four fine-tuned models, the only validation scene
with an entirely tied candidate set is `00012:076`. All 32 scores are exactly
zero at all ten policy-context prefixes. Under base, no scene has an entirely
tied set. Wrong-scene context collapses this scene only at its first prefix.
These counts describe repeated query-prefix records, not independent samples.

Zero scores are broader than this one case. Base has 952/2,240 zero query-prefix
scores; original R0 with policy context has 1,170/2,240; four fine-tuned models
have 1,149–1,194/2,240. A zero on an actual failure is not itself evidence of a
bad ranking. The relevant issue is losing discrimination among candidates.

The score-zero case has one successful query in position four of the saved
query list. Stable sorting all zeros gives rank four automatically. Its base
rank is five. Thus rank 5→4 here does not demonstrate ranking improvement;
with uniformly random tie-breaking, the success rank would have expectation
16.5 over the 32 candidates. Auxiliary terminal-success/high-score query
selection makes the saved order unsuitable as neutral tie-breaking evidence.

## Latent loss has a different pattern for success and failure

The table averages each class within each scene, then scenes equally. Successful
queries exist in seven scenes; failed queries in all twelve, so these columns
have different denominators. Values are future-latent L1, not geometric error.

| Model, evaluated with policy context | Failed queries (12 scenes) | Successful queries (7 scenes) |
|---|---:|---:|
| Base | 0.087498 | 0.077818 |
| Original R0 | 0.085588 | 0.080528 |
| A seed 0 | 0.085863 | 0.081145 |
| B seed 0 | 0.085860 | 0.082025 |
| A seed 1 | 0.086080 | 0.080716 |
| B seed 1 | 0.086051 | 0.081321 |

Every memory model increases successful-query L1 in **6/7 scenes** relative
to base; only `00012:218` improves. Thus the mismatch between average latent
loss and successful-candidate quality extends beyond the one collapsed scene.
This supports investigating transfer and objective alignment. It does not
identify training class imbalance, a particular architecture, or incorrect
residual direction as the causal mechanism in this pilot. Global latent L1
is not the scorer's geometry metric, so increased L1 alone cannot explain AUC.

## A real numerical weakness in the final score

Checkpoint rule is `['soft', 4, 1]`. For surface distance d in pixels, the code
sums `0.5 * (1 + tanh((4 - d) / 2))` across 32 times. Mathematically this equals
`sigmoid(4 - d)`, but the tanh implementation loses the negative tail when
tanh rounds to −1. The scalar readout runs in float32 outside the model's
bf16 autocast; attributing this directly to bf16 would be incorrect.

A synthetic sweep on local RTX 3070 / Torch 1.13.1+cu117 found the first zero
at d=22.02183 px on a 0.00001-px grid. At d=23, tanh32 is zero while sigmoid32
is 5.6028e−9; at d=30 they are zero and 5.1091e−12. This tests arithmetic,
not actual model geometry. The exact threshold may differ on the remote
Torch/GPU backend. Mathematically the score is positive at finite distances;
an exact zero loses information, even when a near-zero probability is reasonable.

This leaves two separate questions:

1. Does a stable computation recover useful relative ordering among the
   tiny scores? It may, or it may expose an equally bad or worse ordering.
2. Why does memory move the predicted geometry into the low-score region?
   A stable scalar transform cannot by itself fix incorrect trajectories.

The observed zero-score failure therefore has a credible numerical component,
but a numerical-only explanation of memory failure is not established.

## Controlled replay design (now completed)

`diagnose_deploy_scores.py` reruns all five models on the same twelve scenes,
all prefixes, base/random-policy/correct-policy/wrong-scene conditions. It must
reproduce every original scalar score exactly. It retains per-time distances
and green positions and compares original tanh32, sigmoid32, float64 sigmoid,
and log-sum-exp of log-sigmoid on **identical float32 predicted distances**.
It adds true-latent probe predictions and actual simulator geometry controls.
`analyze_deploy_scores.py` then reports scene-macro ranking changes and
success/failure geometry errors, including the seven AUC-defined scenes.

Interpretation after replay:

- If stable scores restore ordering with identical predictions, numerical
  evaluation loss contributed. Any deployment success effect still needs a
  matched reevaluation and must not overwrite the original primary result.
- If stable ordering stays poor and geometric error rises with memory, focus
  on evidence transfer/prediction quality before scaling collection.
- If true-latent probe scores also rank poorly, inspect the probe/scoring
  surrogate separately; real-geometry controls help locate that limitation.

The earlier queue delay has ended; all replay outputs and ANALYSIS.json are
available in `results/memory_deploy_score_diagnostic_20260929/`. The synthetic
and saved-output audit remains in OFFLINE_AUDIT.json. No new training launched.
