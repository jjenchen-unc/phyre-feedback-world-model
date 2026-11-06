> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Deployment-distribution pilot registration

Authorized by the user's "好 跑吧" after the fixed-episode → small collection →
single-factor A/B plan. Run directory: `results/memory_deploy_pilot_20260929`.
This registration is written before collection outcomes or A/B training are
observed. The completed train diagnostic is reported separately below.

## Existing-model diagnostic

Select five of the previous memory-training scenes per template, seed
20260929, for 60 scenes. Evaluate R0–R4 final checkpoints with the same fixed
query reservoirs, base failed prefixes K=1/4, base/empty/correct/wrong-scene
conditions as the prior 64-scene validation panel. These are training-scene
resubstitution measurements, not unseen actions or a new generalization test.
The earlier 64-scene panel is memory-only holdout, not full world-model holdout.

Completed first-stage observation: 22/60 scenes have a nonempty base failure
prefix. At K=1, base macro ranking AUC .7425; correct context R0–R4 gives
.8151/.8087/.8144/.8031/.8255. Wrong-scene context gives
.8086/.7983/.8004/.7900/.7927. This establishes observed in-training benefits;
the relatively small correct/wrong contrasts do not establish specific evidence
use. No automatic choice of a winning model is made from these results.

## New collection

- Fixed 60 training scenes above plus four scenes per memory-held template
  (16 total); selection recorded before results in `selection.json`.
- Original deployment pipeline: first 5,000 cache actions; validity filtering
  only; initial-scene LeVJEPA encoder, frozen PCA/base/probe; vectorized base
  scoring; top-1,024 shortlist; per-query probe reranking as in the memory
  evaluator. No success-label filtering. Initial encoding uses the deployed
  duplicated-first-frame encoder, without future observations.
- Two policies on the same shortlist: A uniform random permutation; B frozen
  base/scorer ordering. Execute each until the FIRST actual simulator success
  or ten attempts. No fabricated continuation past success. Cache success
  labels are retained only for retrospective checks and never select attempts.
- At common failure-prefix lengths j=1..min(failures_A, failures_B), compare
  the last min(j,4) failures from A and B. Thus scene, episode and actual K
  match. This conditions the training comparison on both policies reaching the
  state; excluded/first-success scenes remain in collection statistics.
- Each scene has common 32 query targets, disjoint from both policies' failed
  support sets: retain up to two terminal successes, add up to eight high-score
  candidates, then fill uniformly. Terminal successes are explicitly auxiliary
  supervised targets, never historical evidence. This query distribution is
  deliberately shared across A/B; it is not claimed to be pure on-policy
  query sampling. No positive quota or filtering of single-class episodes.
- Replay/record real simulator trajectories using the existing 1,017-step,
  stride-8 recording convention, and frozen LeVJEPA/PCA encoding. Reuse an
  existing rollout only after action/status/frame-count checks.

## Matched A/B fine-tuning

This is a bounded memory fine-tuning pilot, not another full architecture
comparison. Both arms start from the SAME preexisting R0 final checkpoint;
the base and probe remain frozen. No gradient steps are taken on the base.

- A uses random-policy failed contexts; B uses base-policy failed contexts.
- Two paired fine-tuning seeds (0,1), 2,000 steps each, eight query actions from
  one common scene/context per step; identical query draws, actual K, and RNG
  schedule across A/B. These seeds do not replicate pretraining initialization.
- Half of episode draws replay the old R0 sampling identically in both arms;
  half use the new paired episodes. Replay limits dependence on a tiny new
  scene sample. Its fraction is fixed, not tuned on validation.
- Plain future-latent L1; AdamW constant lr 3e-5, weight decay .05, clip 1,
  bf16 forward. No class weighting, ranking term, new retrieval or gate.
- Need at least eight training and two validation scenes with honest paired
  nonempty prefixes. If not, report insufficient pilot coverage and skip A/B;
  do not invent failures, select replacement scenes by outcome, or train on
  validation. A full draw audit and 12-step A/B smoke precede training.

## Evaluation and limits

Evaluate every trained arm on identical held-template query sets with BOTH
random and base-policy contexts, plus base/empty/wrong-scene controls. The
primary pilot comparison is A versus B under the SAME base-policy contexts,
macro-averaged by scene (prefix-rich scenes must not get extra weight).
Evaluate the unchanged R0 initialization too, to distinguish fine-tuning
damage from support-source effects. Retain single-class-query metrics as
undefined AUC rather than dropping the scenes from all reporting.

This is fixed-query evaluation on memory-only held templates, conditional on
paired prefixes, not adaptive deployment success@100 or untouched cross test.
No claim of cross improvement can be made from it. A subsequent refreshed
memory-policy collector is not part of this first A/B intervention.

All scripts, task selections, provenance and job IDs are mirrored to
`oprime:/path/to/phyre-unc/results/memory_deploy_pilot_20260929`.
Weights live under `/mnt/meg/jjenchen/phyre-unc/runs/memory_deploy_pilot_20260929`.

## Collection and numerical checks completed before A/B training

All 76 scenes were scored and replayed. Collection produced 3,555 real
rollouts, zero simulator/cache solved-label disagreements, 51 eligible paired
training scenes, 12 eligible validation scenes, and 458 paired prefix states.
Rendering took 121 seconds; 12-worker collection took 219 seconds. A 42-rollout
encoder smoke completed in 70.4 seconds with the temporal/spatial token-order
checks passing. Full encoding, packing, paired audit and training smoke precede
the queued A/B jobs.

On the 60 selected training scenes, the old small support pools have base
first-attempt success 63.33%; the new selected-policy collection has 13.33%.
Of 404 newly observed base failures, 98.76% are outside the old encoded union
for those scenes. Common query targets are 8.18% positive, and 38/60 scenes
have at least one positive query. These describe a substantial data change;
pool selection, size and encoding route differ, so do not attribute the change
to candidate count alone or treat it as a model-performance comparison.
See `collection_analysis.json` for train/validation separation and exclusions.

An initial-input route discrepancy was measured: collection uses the deployed
initial-only duplicated-frame encoder; historical packed evaluation uses the
first token from a full recorded clip. On 42 already collected actions of the
preselected smoke scene, matched-input scorer implementations agree within
1.82e-5, but changing the initial latent route gives mean score difference
.07364, max .93470, Spearman .99869, top-10 overlap 10/10, and a changed top-1
candidate (both top-1 candidates in this subset are successful). This is NOT a
full 1,024-candidate agreement check and does not prove equal policy behavior.
Collection choices are not retroactively altered. A/B uses identical input
conventions across arms; interpret it as the specified initial-only collection
policy, not bit-identical replication of the historical packed closed loop.
Full packing audit also records first-token differences across scenes.

Job chain: diagnostic33838; render33839 → score33844 → collect33845 →
encode33857 → pack/audit33858 → smoke33859 → A/B train/eval33860 → summary33861.
Additional collection smoke33862, encoding smoke33863, collection analysis33864,
and initial-route score audit33865 completed or run independently as recorded
in `jobs.json`. Frozen source SHA256 manifest is preserved in the run folder.
