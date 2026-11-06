# Updater acceleration validation — 2026-10-12

Verification finished. With controlled numerical settings, cross and within each
pass 65 training slots / 130 episodes per independent history, including two
eight-update stress episodes. Actions, supports, losses, gradients and updated
weights match exactly across original, accelerated and repeated-original runs.
Under default kernels, even original-versus-original chooses different supports.
This supports numerical nondeterminism as the source of the earlier divergence;
no scorer or current-residual semantic difference was found in these checks.


The earlier acceleration preserves fixed-weight inference scores, but independent
training runs can diverge after very small gradient differences. A relaxed FP32
comparison does not establish that the optimizer or support trajectory is the same.
This follow-up explicitly tests reproducibility and implementation equivalence.

## Controlled comparison

Use the same completed residual-memory seed-0 checkpoint, training episode draws,
AdamW settings and original full BPTT, separately for cross and within. Compare
three independent histories: original, accelerated, original repeated. Each starts
from the same checkpoint and RNG seed. No trajectory is forced to match, and no
intermediate weights are copied across histories.

The acceptance condition is exact equality of every event (actions, supports,
losses), every gradient hash and every updated trainable-weight hash. The frozen
predictor and geometry probe must stay unchanged. During the accelerated arm,
every candidate chunk is also scored by both scorers using the very same predicted
latent; their score tensors must be exactly equal before selection.

Start with 16 optimizer slots / 32 episodes per history and panel. If this passes,
extend to 64 normal slots plus one stress slot (two episodes). The stress pool
contains 32 actual failed training candidates; both episodes require eight updates.
Their queries remain from the separate query half. This tests well beyond the
previous divergence at slot 15 and exercises the maximum trained support depth.
These
are implementation checks, not experiments measuring the method's success rate.

## Numerical control

The initial diagnostic allowed only disjoint pooling bins. Both real panels
rejected that assumption: their actual pooling includes overlapping bins. The
initial toy 20×20 → 10×10 check was insufficient; its logs and failed real pilots
are retained.

`updater_validation_numerics.py` enables strict deterministic Torch algorithms.
Every predictor/scorer forward remains unchanged. A dispatch wrapper intercepts
only adaptive-average-pooling backward. Two explicit diagnostic modes compare:

- `cpu`: use ATen CPU pooling backward and return its result to the original GPU.
  The derivative is the same averaging derivative, with CPU accumulation order.
- `cuda`: preserve the original CUDA pooling backward, disabling the strict guard
  only for this operator and restoring it immediately afterward.

Both arms in a comparison receive the same mode. Other unsupported nondeterministic
operations still raise an error. Record pooling shapes/calls and repeat the original
history to verify actual reproducibility. These settings are diagnostic process
settings, not changes to the currently running formal training jobs. The CPU path
is for first-order training/backpropagation, not a higher-order-gradient claim.

## Artifacts

- Runner: `pipeline/validate_updater_trajectory.py`.
- Outputs: `results/updater_validation_20261012/`.
- Initial guarded pilots: `PILOT_{panel}.json` and logs (rejected overlapping bins).
- Controlled pilots: `CONTROL_{panel}.json` (CPU pooling backward).
- Pooling ablation: `POOL_CUDA_cross.json` (original pooling backward).
- Unit check: `pool_unit.log`; unchanged forward and reproducible backward checked
  on both overlapping and disjoint shapes.
- Each result records checkpoint/code/episode hashes and exact comparisons.
- Checks share our current GPU allocations; elapsed time is not an isolated speed
  benchmark. The four formal base/Q0 jobs retain their original frozen code.

The 16-slot pilots pass exactly in both panels with CPU pooling backward. Cross
also passes with original CUDA pooling backward while other kernels are strict.
Actual pooling is 14×14 → 10×10. Every accelerated score checked against the
original scorer is identical: 926 cross and 884 within scores in the CPU pilots.
CPU versus CUDA backward changes some values across configurations, so comparisons
must be made within a configuration; do not conflate those histories.

Extended results are complete and independently audited in `SUMMARY.json`:

| Run | Original vs accelerated | Original vs original repeat |
| --- | --- | --- |
| Cross, strict kernels, original CUDA pooling backward | All exact | All exact |
| Within, strict kernels, original CUDA pooling backward | All exact | All exact |
| Cross, strict kernels, CPU pooling backward | All exact | All exact |
| Cross, default kernels, original CUDA pooling backward | Diverged | Diverged |

“All exact” includes all 65 event lists, gradient hashes and updated-weight hashes,
plus unchanged frozen predictor and geometry probe. The accelerated scorer matches
all 5,368 checked candidate scores in strict cross and all 5,117 in strict within.
Both eight-update episodes finish all eight support updates in every arm.

The default-kernel control first differs in gradients/weights at slot 2 in both
comparisons. Its original-vs-original support selection first differs at slot 15;
original-vs-accelerated first differs at slot 21. Slot numbers are zero-based.
Even in that diverging optimized history, all 5,392 scorer values equal the original
scorer evaluated on the same current prediction. These observations separate scorer
correctness from changes in the predictor caused by accumulated gradient variation.
Do not infer that the optimized version is “more stable” from the later split in
this single run.

All cross controls use the same checkpoint and episode hashes. The strict-CUDA and
default-CUDA comparisons use the same workspace configuration, original pooling
operator and dispatch instrumentation; the strict-algorithm flag is the intervention.
The evidence identifies numerical-kernel reproducibility as a cause of the observed
branching. It does not isolate one particular CUDA kernel as the sole cause.

## What this permits

The accelerator passes the extended implementation-equivalence check **under the
validated numerical configuration**. It recomputes current-token predictions and
residuals, and retains full BPTT. Future comparisons requiring identical histories
should use the same numerical configuration in both reference and accelerated runs.
The default-kernel path still has no bit-exact-history guarantee, including when
comparing the original code to itself.

The validated CUDA configuration retains original pooling backward as an explicit
exception to Torch's strict guard. Its exact reproducibility is empirical on these
histories, not a guarantee across all shapes, devices or future episodes. The CPU
pooling comparison is a separate diagnostic, not a requirement for inference.

The validation settings have not been inserted into the formal base/Q0 study.
No full updater retraining was launched. These checks share allocated GPUs, so
there is still no uncontended end-to-end speedup measurement or full-training
convergence claim.

## Reproduce

On a GPU allocation, from the shared project directory:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
.venv-ml/bin/python pipeline/validate_updater_extended.py \
  --panel cross --slots 64 --reference-repeat --audit-scorer \
  --pool-backward cuda --output results/new_validation/cross.json
```

Use a fresh output path. Repeat with `--panel within`. `--default-kernels` runs the
negative control and can exit nonzero when its exact-equality gate fails; its full
result is persisted first. `--pool-backward cpu` runs the pooling-backward ablation.
The context must cover backward as well as forward. The script does this explicitly.

Re-audit saved results without a GPU:

```bash
python3 pipeline/summarize_updater_validation.py results/updater_validation_20261012
```
