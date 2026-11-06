> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Review of the R0–R4 interpretation

The measured scores are reproducible; the mechanistic and method-selection
conclusions were stronger than the design supports. This review changes the
interpretation, not frozen code, checkpoints, or measured results.

## 1. Observed ordering is not an established method ordering

All experiments use one training seed and one reused cross development fold
with only four templates. The 100 tasks are not 100 independent samples of
novel mechanisms. The implementation has not demonstrated convergence or
robustness to training seed, loss weight, or template split.

Post-hoc paired bootstrap: 20,000 resamples of tasks within each observed
template, retaining the template weights. The intervals below are descriptive
95% empirical resampling intervals, conditional on these templates and fitted
models. They omit training and novel-template uncertainty, do not establish
population coverage for the selected panel, and are not multiplicity-adjusted.

| AUCCESS difference | Estimate | Conditional interval |
|---|---:|---:|
| R0 − base | −.0122 | [−.0294, +.0047] |
| R1 − R0 | −.0143 | [−.0336, +.0037] |
| R2 − R0 | −.0219 | [−.0448, +.0004] |
| R3 − R0 | −.0187 | [−.0430, +.0050] |
| R4 − R3 | −.0271 | [−.0579, +.0016] |
| R4 − base | −.0579 | [−.0883, −.0282] |

Thus R4 versus base is a stronger negative observation on this fixed panel,
but it is not the comparison isolating the added ranking term. R4 versus R3
is that comparison, and it remains uncertain. Intervals containing zero are
not evidence of equivalence or proof of no effect.

Template dependence is substantial: excluding 00014 changes R1−R0 from
−.0143 to +.0050 and R4−R3 from −.0271 to +.0015. This is a sensitivity
analysis, not grounds to remove 00014 from the benchmark.

## 2. The data comparisons bundle multiple interventions

R1 versus R0 changes support outcomes, policy/action distribution, and actual
context size. Mean actual K over all 20K episodes is 2.1043 for R0/R2 and
1.5359 for R1/R3/R4. Short first-success prefixes account for much of this.
Equal requested K and matched queries do not hold the amount of evidence fixed.
Therefore the experiment compares recipes; it does not isolate the effect of
failure-only memory. There is no evidence yet that K itself caused the loss.

R2 versus R0 changes query selection AND loss weighting. Success fraction in
sampled queries falls from 42.82% to 36.30%; the balanced loss gives each class
equal aggregate weight when both are present. These are different notions of
balance. The experiment cannot identify the contribution of either change.

The `diverse_failure` category is actually uniform sampling without replacement
from same-scene failures (`redesign_sampling.py`, lines 33–35). It has no
explicit action/outcome diversity criterion. Query successes do use farthest
point selection. The registration's phrase "diverse failures" overstates the
implemented support sampler; the category name is retained in frozen outputs
for provenance, not as evidence that diversity was tested.

## 3. Wrong-scene controls do not establish memory irrelevance

Only 29/64 internal-validation scenes have nonempty contexts. At K=1, 25/29
wrong-scene donors belong to the same template as the recipient. Each scene
uses one deterministic donor; there are no multiple donor draws or separate
same-template/cross-template control estimates.

Correct-minus-wrong mean AUC differences range from −.0101 to +.0103; all five
conditional resampling intervals include zero. Fixed-panel resampling also
does not model dependencies induced by shared donor/recipient scenes. The
result is inconclusive about evidence specificity, not proof that useful
scene-specific information was not learned. Same-template evidence can be
partly useful, and a weaker control contrast can mask specificity.

The panel is memory-only holdout: the frozen base/probe saw its templates.
It does not directly diagnose the new models on unseen cross templates.
The two earlier cross failures were selected as diagnostic examples; their
strong effects cannot establish prevalence across the full cross panel.

Requested K=4 actually gives K={0:35, 1:16, 2:5, 3:3, 4:5}. There are only
five scenes with four contexts. This panel is insufficient to conclude that
increasing memory size generally fails to help.

## 4. Latent error is an observation, not the demonstrated causal mechanism

The successful-query L1 increase is robust to equal scene weighting: at K=1
it occurs in 29/29 nonempty scenes for R0–R3 and 28/29 for R4. This strengthens
that descriptive finding. It does not show that this increase causes cross
ranking losses: latent dimensions are not weighted by decision relevance,
the panel uses different templates, and score ordering can improve despite
larger latent error.

Failed-query L1 decreases only in aggregate. It increases in 12/29, 10/29,
16/29, 19/29 and 15/29 scenes for R0–R4 respectively. In R3 a majority of
scenes worsen even though the mean improves. "Helps failures, hurts successes"
must be qualified as an aggregate pattern, not a universal behavior.

The frozen scorer's numerical parity and finite gradients establish that R4
optimizes the implemented objective. They do not establish that its surrogate
score remains calibrated under optimization or aligns with cross performance.
Only one ranking weight (.02) and temperature (4) were tested. No conclusion
that ranking objectives generally fail, or that probe exploitation actually
occurred, is warranted.

## 5. The proposed architectural remedy was imprecise

The existing model ALREADY conditions memory attention on the query: query
tokens contain the initial scene and candidate action, and `q(x)` attends to
memory keys/values in every base block (`common.py`, lines 164–174). Therefore
"add query-conditioned memory" does not specify a new capability.

A learned gate for NONEMPTY contexts, explicit sparse retrieval, or an
outcome-relevant representation would be distinct modifications. None has
been shown necessary or effective by these results. Empty gating is already
implemented and verified. Recommending an architecture replacement before
separating data, optimization, and representation explanations was premature.

## Defensible conclusion and next evidence

These five trained models did not beat the matched base on this development
panel. The prescribed recipe changes did not produce an observed improvement
at this seed/budget. The evidence does not yet identify whether support
distribution, context size, objective/optimization, scorer mismatch, or
architecture is the dominant cause.

Before an architecture rewrite, the narrow informative controls are: match
actual K when changing support type; separate query sampling from class loss
weighting; compare R3/R4 across seeds and held-out template splits; and use
fixed cross queries with multiple same-template and cross-template donors.
Scorer alignment and successful-candidate protection should be measured on
those same queries. Keep all post-hoc development choices distinct from a
future untouched evaluation. These are recommendations; no new training was
launched during this review.

Evidence and reproduction:
`pipeline/review_memory_redesign_results.py` generates
`results/memory_redesign_20260928/REVIEW_METRICS.json` from saved raw results.
Original run code/checkpoints and registration history remain preserved.
