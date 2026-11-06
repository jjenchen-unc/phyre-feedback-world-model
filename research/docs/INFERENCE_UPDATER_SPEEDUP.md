> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Inference and updater implementation

The extended verification now passes exact original/accelerated/repeated-original
comparisons in cross and within under controlled numerical settings: 65 slots,
130 episodes per history, including eight-support stress cases. See
[the validation report](UPDATER_ACCELERATION_VALIDATION.md) for the configuration,
raw results and limits. The default-kernel control also makes the original code
diverge from itself; it still has no bit-exact-history guarantee.

The implementation reduces repeated transfers and preparation in action ranking.
Inference reproduces checked reference scores exactly. The updater remains an
optional backend; running base/Q0 jobs have not been changed, and no new full
updater training was launched. Uncontended speedup remains unmeasured.

## What is reused

`pipeline/fast_inference.py` prepares initial object queries once per candidate
chunk. The probe still processes each object's query with the original tensor
shape, but its outputs are copied to the CPU together. The original NumPy geometry
calculation and float64 ranking scores are retained. Predictor batch size remains
32, including the original last partial batch. This avoids the rounding changes
introduced by the earlier prototype that merged object attention batches.

`CandidateRanker` reuses these layouts over repeated trials. It runs the predictor
again on every call, using the supplied current memory state. Tried actions are
masked and ties select the lowest original candidate index. `PreparedScorer`
provides the same calculation through the existing tensor scorer interface. Its
bounded cache keys contain the actual initial feature values, so changing a scene
cannot accidentally reuse another scene's layout.

`pipeline/fast_updater.py` has the same loss-call interface as the earlier staged
updater. It uses the prepared scorer to collect supports. Observation loading
and tensor lifetime follow the original implementation. The predicted trajectory
and residual are recomputed for every current token state. Full backpropagation through the support sequence is
retained. The query loss remains latent L1 plus geometry loss, with default geometry
weight 0.1. The differentiable geometry loss uses the original implementation;
the inference scorer is only used for discrete support selection.

The two training episodes per optimizer slot still backpropagate sequentially.
This change does not batch different scenes together, change support/query draws,
increase the predictor batch size, truncate gradients, or reuse old residuals.
It targets ranking and data-transfer overhead; it does not remove the dependency
between successive trials.

## Use

For a new updater runner, import this loss instead of the old staged loss:

```python
from fast_updater import updater_loss
loss, event = updater_loss(model, geometry_probe, scorer_probe, rule,
                          source, episode, verify=True)
```

The model must be in its updater training stage, with predictor/initial tokens
frozen and the updater trainable. Both probes must be frozen and in evaluation
mode. `source` uses the existing Store/Pack interface. Keep the existing optimizer,
loss scaling, schedule and episode sampling. Freeze the new implementation and
record its hashes when registering a future formal run.

For static evaluation of a completed matched-study checkpoint, run on the server:

```bash
.venv-ml/bin/python pipeline/evaluate_fast_predictor.py \
  --panel cross --arch q0 --weight 0.1 --seed 0 \
  --output results/fast_eval_cross_q0 --verify-reference
```

The output must be outside the registered matched-study and previous memory-run
directories. By default it evaluates the full 100-task panel. `--limit-tasks 2`
runs an explicitly labelled subset for an integration check. Saved output includes
checkpoint, backend and input hashes; resuming requires matching provenance.

For repeated memory inference, retain a `CandidateRanker` instance for the fixed
pool and call `rank(current_state, tried_mask)` after each observed trial. Its
returned selected prediction is the one used in that ranking, suitable for the
current-state residual. Do not pass a prediction from an earlier state.

## Earlier checks (superseded by the extended controlled validation above)

- Six CPU and six CUDA tests pass: exact original scores, partial batches, ties,
  tried-action masks, exhaustion, absence of future-feature leakage, current-state
  and weight changes, static geometry, cache content invalidation and eviction.
- Real scratch base/Q0 checkpoints: cross and within each pass a full 1,024-action
  comparison on one task. Every score equals the frozen evaluator and saved result.
- Real memory checkpoints: cross and within each pass three consecutive trials,
  including every score and selected action, while updating from the observed
  residual under the current tokens. This is a short regression check, not another
  full method evaluation.
- CLI smoke evaluates two cross Q0 tasks and reproduces their saved scores exactly;
  this is explicitly recorded as a subset, not a full 100-task evaluation.
- Free-running updater comparisons do **not** pass long-trajectory equivalence.
  Initial runs have identical actions/losses but small gradient differences. A
  later 16-slot cross rerun diverges at slot 15: one arm uses one support and the
  other two. Within retains identical events but differs in final weights.
  These failed checks remain saved; the backend is not enabled in formal training.
- A 16-slot original-versus-original control itself has gradient variation up to
  1.16e-10. This establishes that bit-exact backward reproducibility is unavailable
  in that configuration; it does not prove that all optimized-run divergence is
  harmless or establish its unique cause.
- The stronger isolated check, `benchmark_paired_updater.py`, restores identical
  incoming model **and optimizer** state before each reference/optimized step,
  then advances the reference for the next pair. It tests 16 slots / 32 episodes
  per panel, preserving complete support replay. Events must match exactly;
  gradient/weight checks record bit equality and the initial strict bounds
  `rtol=1e-6, atol=1e-8`. Both panels match all action/support/loss events exactly.
  Cross maximum gradient/weight differences are 1.16e-10 / 6.67e-7; within maxima
  are 2.98e-8 / 9.61e-7. The initial strict checks fail and remain preserved in
  `PAIRED_{panel}.json`. A separate, explicitly post-hoc assessment uses installed
  PyTorch 2.7.1 default FP32 tolerances (`rtol=1.3e-6, atol=1e-5`); both pass even
  its absolute bound alone. See `PAIRED_VALIDATION.json`. This supports numerical
  agreement of individual steps, not equality of long independent trajectories.
  The maintained paired checker reports both criteria. No full new updater
  experiment or convergence claim follows from these checks.
- GPU observation caching was removed from the final updater. Its isolated short
  check had 1e-10 to 1e-9 gradient differences; the scorer-only short check passed.
  This conservative choice does not identify the underlying CUDA mechanism.
- Strict deterministic mode stops on original adaptive-pooling backward; warn-only
  mode also reports Flash Attention backward nondeterminism. These diagnostic
  settings are not enabled in formal training.

These checks share GPUs already allocated to our training jobs and do not allocate
a fifth physical GPU. Their timings include contention and cannot establish an
isolated end-to-end speedup. In the first inference check, full ranking time ratios
range from 0.91 to 1.35; that spread is not a reliable speed estimate. No claim of
a verified training-time reduction is made from shared-GPU timing.

The old `pipeline/performance_20261012/batched_readout.py` prototype is not adopted:
it changed some scores and failed exact updater trajectory/gradient checks. The
new implementation deliberately preserves each attention call's arithmetic shape.
