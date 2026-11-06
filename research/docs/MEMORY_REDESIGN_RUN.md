> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# R0–R4 first-wave execution registration

2026-09-28. User authorized execution of MEMORY_REDESIGN_PROPOSAL.md.
This file freezes the first screening wave before training results are observed.

## Scope and practical differences from the broader proposal

- Five newly initialized, matched gated-memory models, one seed (0), 20,000
  optimizer steps each, eight queries from one scene/context per step.
  This is a new matched reference, not an exact reproduction of the historical
  eight-scenes/one-query-per-scene training batch.
- Frozen existing cross-f1 base and probe. Memory: residual, 16 time steps,
  10x10 spatial pooling, per-attempt encoder, hard empty-context gate.
- Existing cluster and s16r16 encoded training packs are deduplicated by
  rollout file. All R variants share this union, not just the old cluster pack.
  No new simulator/video-encoder dataset is required for this first wave.
- Per scene, seed 20260928 partitions actions into disjoint support/query
  reservoirs without consulting outcomes. They remain fixed across variants.
- Base and historical gated-memory policies replay on the support reservoir,
  selecting by scores before exposing each selected label, stopping at first
  success or four attempts. These are genuine prefixes in a bounded cached
  reservoir, not full 1024-action deployment-policy collection.
- Failure-context episode target: 30% base traces, 30% historical memory
  traces, 30% diverse same-scene failures, 10% empty. Policy episodes sample
  scenes conditionally from those with real failed prefixes. R0/R2 share this
  same conditional scene schedule even though they use mixed supports, so
  scene difficulty is matched across variants. First-try-success scenes remain
  eligible in the diversity/empty categories. A logged diversity fallback is
  used only if no training scene supplies any prefix for the chosen policy.
  Short prefixes truncate requested K. Actual proportions and fallback counts
  are reported. This first wave does not yet refresh traces
  with newly trained R policies.
- All variants select the same scene at each step. R0/R1 share uniform queries;
  R2/R3/R4 share the grouped queries. Separate RNG streams ensure context draw
  count cannot change the query set. Requested K sampled from {1,2,4}.
- R2/R3/R4: up to 2 diverse successful queries, 2 base-high-score failures,
  2 failures nearest a successful query-reservoir action, 2 uniform candidates;
  shortages filled from remaining unique query candidates. Single-class
  episodes remain, without a ranking term. Pair queries share one context.
- R0/R1: unweighted latent L1. R2/R3: class-normalized prediction loss.
  R4: same as R3 plus 0.02 * mean softplus(-(success_score-failure_score)/4).
  Ranking weight and temperature are fixed for this wave; no dev sweep.
- AdamW lr 3e-4, weight decay .05, OneCycle 3% warmup, clip norm 1, bf16 model
  forward and fp32 differentiable frozen-probe readout. Query initial features
  only enter the scorer; query future features are metric targets only.

## Validation and reporting

Four of the sixteen original train templates are reserved for NEW memory
validation, selected once by seed 20260928. Existing base/probe/historical
memory have seen those templates, so this is explicitly memory-only holdout,
NOT full pseudo-cross validation. Train all five variants on the other twelve.
Sixteen deterministic validation scenes per reserved template form fixed-Q
K=1/4 panels; zero-failure episodes are retained and effective K reported.
Correct and wrong-scene evidence controls match available K. Baseline and
hard-empty scores must agree.

Evaluate final 20K checkpoints (no checkpoint selection) on the same existing
100 cross-f1 dev tasks, each with 1024 candidates, max 100 attempts, at most
10 updates and keep=4. This reused dev panel is exploratory; the two prior
00014 failures are not a tuning objective. No test-set claim or multi-seed
generalization claim is permitted from this screening wave.

The previous TTA result uses up to 10 remembered failures; treat it as a
historical reference only, not a memory-budget-matched R0–R4 comparison.
This first wave focuses on the five authorized data/objective variants.

## Execution and checks

New files are separate from the existing training/evaluation scripts. Source
will be copied to an immutable run-specific code directory before expensive
jobs. Steps: numerical/gradient preflight → four trace-preparation shards →
dataset merge and smoke train → five training jobs → fixed-Q and closed-loop
evaluation → automatic aggregate. Checkpoints include optimizer/scheduler/RNG
for resume. Dependent jobs run only if preceding stages succeed.

Preflight verifies NumPy/Torch geometry and frozen-probe score parity,
finite nonzero latent gradients, no dependency on query future geometry,
empty-context identity with nonzero adapter weights, success-stop semantics,
support/query exclusion, and paired sampler equality across variants.

Output data and summary: `results/memory_redesign_20260928/` on oprime.
Weights: `/mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928/`.
Job IDs, measured smoke timings and final launch details will be appended.

## Submitted pipeline

- Preflight 33799 passed: max scorer discrepancy 1.5259e-5, finite nonzero
  input gradient, 100 paired-sampler trials, future-input and empty-gate checks.
  Initial preflight 33798 was canceled to fix repeated NPZ metadata loading;
  it did not produce a model or experiment result.
- Prepare array 33800 (four one-GPU shards).
- Merge 33804, after successful preparation.
- Sampler audit 33808, after merge; verifies 2,000 paired episodes per variant
  against actual records, disjoint template partitions, and trace integrity.
- Smoke 33805, R0/R4 each 12 steps, after successful sampler audit.
- Train + fixed-Q + closed-loop array 33806 (R0–R4, one GPU per variant),
  after successful smoke. Twelve-hour wall limit per variant; actual runtime
  and final counts will be reported from logs.
- Aggregate 33807, after all training/evaluation tasks exit. It marks missing
  stages incomplete and never substitutes absent outputs with zeros.

Frozen source snapshot: `results/memory_redesign_20260928/code/` on oprime.
Stage logs: `logs/mr_{prepare,merge,smoke,train_eval,summary}_*.out`.

Pre-training refinement while preparation was running: policy-first-success
scenes were common in the small reservoir. The sampler was changed to the
conditional scene schedule above before any R training, preserving the 60%
policy-failure mixture rather than mostly falling back to random negatives.
Preparation is unaffected. The final sampler is independently checked against
the complete merged dataset before the smoke stage is accepted.

## Verified launch status (2026-09-28 21:03 UTC)

Preparation, merge, audit and training smoke completed successfully. All five
33806 tasks are running, with finite losses/gradients and saved checkpoints:
R0/R2 at 7,400 steps, R1 at 8,700, R3 at 8,600, R4 at 6,300 of 20,000.
Measured training throughput is about 0.25–0.35 seconds/step. Remaining
training is roughly 50–85 minutes at this point; evaluation is additional.

The final union is **99,419 unique actions**: pack row counts 51,161/52,276,
unique counts 50,953/52,276, and cross-pack overlap 3,810. The preliminary
99,627 estimate subtracted only cross-pack overlap, overlooking 208 duplicates
within the cluster pack. The actual Store correctly removed both kinds before
preparation and training; this correction changes reporting only.

Training uses 1,200 scenes. Memory-only held templates are 00001, 00004,
00012, 00022, with 64 validation scenes. Full-dataset audit passed 2,000
paired episodes per variant: failed-context variants used 630 base traces,
588 historical-memory traces, 582 diverse-failure episodes and 200 empty
episodes, with zero policy fallbacks. Actual K can be below requested K due
to first-success stopping. R0 and R4 smoke each completed 12 optimizer steps
and wrote checkpoints; R4 included a finite, nonzero ranking term.

Evaluation integration smoke 33813 runs both evaluators on two scenes with
the 12-step smoke_R4 checkpoint. These are software checks, not reported
R0–R4 experiment results or a checkpoint-selection criterion. The five full
training jobs continue independently; aggregate 33807 remains dependent on
their training and evaluation completion.

## Post-run review correction

The frozen `diverse_failure` support category actually samples failures
uniformly without replacement; it does not optimize action or outcome
diversity. Interpret the earlier "diverse failures" wording as an inaccurate
description of this implementation, not an executed diversity intervention.
No frozen code or output was changed. The actual-K distributions also differ
between mixed-context and failed-prefix variants (mean K 2.1043 vs 1.5359),
so those comparisons estimate combined recipe changes, not failure-only
support effects in isolation. See `MEMORY_REDESIGN_ANALYSIS_REVIEW.md`.
