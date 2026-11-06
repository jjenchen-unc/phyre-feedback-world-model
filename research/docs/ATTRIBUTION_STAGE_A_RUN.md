> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Attribution stage 0/A execution registration

2026-09-30. User authorized the proposed next step: data audit and stage A of
`MEMORY_CAUSAL_ATTRIBUTION_PLAN.md`. No retraining, new simulator collection,
oracle correction, support-source intervention or D1/D2 confirmation in this run.

## Frozen scope

Existing D0 cross-f1 100 scenes, all1,024 historical candidates. Base and true
latent/geometry controls once per scene; originalR0 is the primary checkpoint,
A_s0/B_s0/A_s1/B_s1 are predefined replication checks. No winner selection.
Checkpoints, metadata and saved stable-base trace hashes are frozen in manifest.
Training-pack template sets are compared with this panel; all D0 exposure is
explicitly recorded. New D1/D2 manifests and power planning remain future work.

Contexts use the completed stable-base policy's actual failed prefix: K1 is
its first failure (empty when first attempt succeeds); K4 only where four real
failures precede success. Never fabricate failures or continue beyond success.
All models receive the SAME context. Query IDs exclude the union of available
K1/K4 supports, at most four failures, identically for every condition. This
also matches K1/K4 query sets; it is an offline remaining-query panel, not an
exact deployment-time query distribution at the first attempt.

Primary reporting: nonemptyK1 scenes, with class and AUC availability counts.
Also report all100 including empty scenes, K4, and K1 restricted to the SAME
K4-eligible scenes. All-failure-query scenes remain in loss/error metrics with
undefined AUC. No outcome-based query augmentation. Average within scene,
then equally across scenes. No inference over unseen-template populations.

## Prediction/readout and parity

Preserve original closed-loop batch32, bf16 forwards, float32 probe and NumPy
surface geometry. Stable float64 log-sigmoid sum uses identical distances.
Forward all1,024 candidates (including support rows) to preserve original
batch layout; only the frozen query mask enters reported metrics.

Every scene's base scores must exactly reproduce BOTH legacy/stable channels
of the previous closed-loop snapshots. For nonemptyK1, each memory's scores
must reproduce its first-update snapshot (common initial action/history).
K4 uses the fixed base history, which need not equal a memory policy's history.
All five checkpoints' empty paths are checked on the first full batch per scene.

## Measurements

Future latent L1 and RMS L2; per-time L1; all object positions; surface-distance
and green/dynamic-goal errors; stable AUC, first-success rank, Hit@10, tie rate,
strict pairwise ranking flips and success/failure score margins.
All/success/failure metrics have query and scene denominators.

Contact readouts at surface distance0 and at scorer threshold tau are explicitly
geometry proxies, NOT the simulator's success predicate. Save distances for
later inspection; report time disagreement, first-near-contact time error only
when both predict/observe it, missed-event rate, total/longest sampled duration
errors and distance error at true near-contact times. Sampling uses simulator
step units; future 31 samples at spacing32, never seconds inferred from them.

GT-geometry additionally reads every saved raw trajectory. Require exact action,
solved label and feature alignment, stride8/1,017steps/128samples. Compare32
points (steps0..992) with125 points over the SAME interval, using trapezoidal
time weights normalized to32 samples for both resolutions. Keep original
32-point unweighted score separate. This is denser sampling, not every physics
step; events between recorded frames remain a limitation.

Only initial geometry enters predicted readout. True query future is used for
targets and labelled oracle controls. Full future latents are not duplicated
to disk; pack indices, checkpoint hashes and saved geometry/error/score arrays
permit reproduction. NPZ outputs plus atomic per-task summary enable resume.

## Execution and verification

Audit manifest → eight-scene smoke → full ten-shard array (max2 concurrent) →
aggregate verification. Eight smoke scenes are the first two task IDs per
template, fixed before new metric results. Smoke evaluates R0 and all truth/base
controls, plus every checkpoint's empty path. Main runs all five memory models.
Smoke is an engineering check and does not select models or contexts.

Final aggregation requires all100 task summaries, recomputes scores/AUC from
saved geometry and class-wise L1 from per-query arrays, and verifies every
scene's historical parity/empty/dense-data checks. It publishes complete only
after all required outputs pass. Source is frozen before job submission.

Run: `results/attribution_stage_a_cross_20260930/` in local/shared project.
Implementation: `pipeline/{prepare_attribution_a,evaluate_attribution_a,
attribution_a_metrics,summarize_attribution_a}.py`.

## Completion status

Audit34223, smoke34224, all10 evaluation shards34225 and summary34226 have
produced complete verified outputs. Eight smoke scenes passed; all100 main
scenes passed exactbase/K1 replay, allfive empty paths and1024rawtrajectory
alignment checks. SUMMARY.json complete=true andaudit_passed=true. No queued
or running jobs remain. Slurm accounting is disabled; completion is established
from checked artifacts and final logs, not sacct history.

Coverage:100scenes,73nonemptyK1,62K4,57nonemptyK1withsolutions. Remote raw-array
verification passed. Local recomputation checked2160macrovalues and100task
audit records;103downloadedfilehashes matchremote. See VERIFICATION.json.

Results and interpretation: [stage A results](ATTRIBUTION_STAGE_A_RESULTS.md).
No later experimental branch has been launched.
