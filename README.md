# What Can a Physical Attempt Teach a World Model?

COMP 669 project: feedback-conditioned latent world models for PHYRE.
We ask whether observing a failed physical attempt helps a model rank untried
actions in the **same reset scene**.

## Start here

- **[Latest progress report](research/docs/EXPERIMENT_PROGRESS_20261022.md)** (Chinese): completed experiments through October 21, recorded October 22; published November 6, 2026.
- **[Research code and evidence](research/README.md)**: pipeline, experiment snapshots, registered plans, result tables and audits.
- **[Latest cross comparison](research/results/support_context_ablation_20261021/CROSS100_TABLE.csv)**: all support-feature ablations, including both seeds.
- `phyre_feedback/`: standalone model, memory, geometry loss, ranking and adaptation components.
- `results/milestone.json`: historical October 18 classifier-first results and per-scene first-success ranks.
- `results/pretraining_dev.csv`: all 24 matched pretraining runs; prediction errors only.
- `reference_runs/`: experiment drivers for inspection, with machine-specific paths redacted.
- `SOURCE_MANIFEST.json`: original-file and exported-file hashes plus extraction changes.
- `RESEARCH_MANIFEST.json`: provenance and public-file hashes for the later research snapshot, including path redactions and omitted large historical files.
- `RELEASE_VALIDATION_20261106.json`: publication checks, supported test environment and reproduction limits.

This is a **code-and-results milestone release**, not a pretrained agent download.
The reusable components and result audit run locally. Full experiment drivers
require the external data packs, frozen dependency snapshots, encoder/probe weights,
trained predictors, and registered manifests used by the original cluster runs.
Those assets and the scheduler/RPC deployment configuration are not distributed
here. Reference drivers are not turnkey training commands. No new GPU results
are claimed for the reorganized package.

## How actions are chosen

An action is a red ball's `(x, y, size)`. The simulator resets after every attempt;
the model's memory persists within the task and resets for a new task.

1. An initial-scene classifier scores the official 100,000 action proposals.
2. Retain the highest-scoring 1,024 legal actions, without filtering by success labels.
3. Predict future video features for candidate actions; a frozen probe decodes
   geometry; a contact-based scoring rule orders the candidates.
4. Execute one untried action. Only its real observation is available as feedback.
5. On failure, update memory and rerank. Stop on success or after 9 attempts.

**Completed classifier-first experiment:** attempt 1 is the classifier's top action;
Q0/predictor reranking starts at attempt 2. This protocol is shared by all four arms.

**Completed Q0-first experiments:** the same classifier pool is reranked by initial
Q0 before attempt 1. Both updater training and evaluation use this new policy.
The full nine-attempt comparison is documented in the
[October 19 report](research/docs/MEMORY_OVERNIGHT_INVESTIGATION_20261019.md).
The later head experiments below study only the first feedback and next ranking.

## Latest findings through October 21

Three completed diagnostic rounds added 16 short updater fits, 6 feedback-head
fits, and 20 support-feature head fits. Within each round, data, candidate pools,
first actions and the relevant training controls are matched. These are different
training procedures, not an equal-compute comparison across rounds.

The latest head experiments use cross fold 1, 100 scenes, the same frozen Q0 and
classifier-selected 1,024-action pools. The first Q0 action solves 3 scenes; the
remaining 97 are reranked after that failure. Training uses 53 failed-first scenes
and 424 fixed uniform queries, including 32 successful actions.

| Method / extra head inputs | S@2, seed 8419 | S@2, seed 8420 |
|---|---:|---:|
| Q0, no update | 5/100 | 5/100 |
| Static learned readout | 6/100 | 6/100 |
| Query-only MLP | 5/100 | 6/100 |
| Full support context, zero residual | 9/100 | 8/100 |
| Full support context, real residual | 8/100 | 8/100 |
| Query raw distance trajectory | 6/100 | 8/100 |
| Support predicted distance features + query–support distance differences | 8/100 | 10/100 |

**Interpretation:** predicted support geometry is a promising ranking reference;
there is still no consistent additional benefit from the observed residual. The
last row uses information available before executing an action, so it is not
evidence that observation-driven memory works. It also does not consistently beat
the full zero-residual control. One fold, two head seeds, many explored settings
and mixed validation results do not establish a robust winner.

S@2 is actual success within the common first attempt and one subsequent attempt,
reconstructed from cached simulator labels. The CSV's `top1/3/5/8` columns count
successful coverage among the 97 first-failure scenes after **one** reranking;
`top8` is not repeated-feedback S@9. This differs from the historical nine-attempt
table below. No new experiments were run for this November publication.

## Model and training

- **Base:** action-conditioned axial self-attention predictor; no memory prefix.
- **Q0:** the predictor plus 32 learned initial prefix tokens; no online update.
- **Residual updater:** cross-attention updates tokens using the tried action and
  `encoded observation - current prediction`. Predictor weights stay frozen.
- **Zero-residual control:** the same update mechanism with residual set to zero;
  selected action/current state still affect the update. It is not no-update.
- **AdaJEPA-style TTA:** adapt the last predictor block and output normalization
  using observed latent error, then reset weights between scenes. In this pilot
  it starts from Q0. This is an inspired baseline, not an official reproduction.

Matched pretraining: random predictor initialization, paired backbone seeds/data
order, 8 epochs, batch 8, latent L1 + lambda * geometry loss. Cross uses 52,276
trajectories; within uses 52,349. Lambda values: 0, .03, .1, .3; three seeds at .1,
one seed otherwise; 24 trainings total. Encoder/PCA/probe are frozen.

Updater pilot: 64 train scenes per panel, 256 slots of two episodes; support caps
1/2/4/8; 8 disjoint query actions. Full BPTT with the final state's future-latent
L1 + .1 * geometry loss. Current residuals are recomputed, never cached.
The new updater starts fresh from the same Q0 parents, not from old updater weights.

## Historical classifier-first results (2026-10-18)

S@9 = fraction of scenes solved within nine attempts. Cross fold 1 and official
within fold 0 each have 100 evaluated scenes here.

| Method | Cross S@9 | Within S@9 |
|---|---:|---:|
| Classifier original order | 4% | 44% |
| Fixed Q0 after classifier first attempt | 11% | 53% |
| Residual updater | 9% | 55% |
| Zero-residual updater | 12% | 54% |
| AdaJEPA-style TTA | 10% | 57% |

All four feedback arms share the candidate pool and first attempt. Reranking
helps; additional feedback gains are small and inconsistent. These are one-seed
pilots, not a demonstration of reliable generalization.

In a separate static oracle diagnostic (10 cross, 25 within scenes, each with
1,024 candidates), replacing predicted future latents with encoded real videos
raises Base S@9 from 10% to 100% on cross, and 40% to 96% on within, through the
same probe/scorer. Every selected diagnostic pool contains a successful action.
This isolates a large predicted-representation gap, but does not distinguish
prediction error from predictor/probe mismatch or prove that mean latent L1 is
sufficient for good decisions. It is not a deployment algorithm.

The old shortlist-based ranking comparisons were withdrawn after a selector
protocol mismatch. They are **not** included in the reported results. Their
independent training checkpoints/dev errors remain usable. The full geometry
weight/seed sweep has not been reevaluated under the new candidate protocol.

## Local checks and result reproduction

```bash
python -m pip install -e .
python scripts/summarize_results.py
python scripts/summarize_latest_results.py
python scripts/verify_research_snapshot.py
python -m unittest discover -s tests -v
```

The summary and snapshot verification scripts only need Python's standard library. Model tests use small
synthetic CPU tensors to check state, freezing and gradient invariants; they do
not measure PHYRE performance. Full scoring uses the original 32-action batches;
changing batch arithmetic can change BF16 rankings.

See `reference_runs/README.md` for the experiment-driver map and asset requirements.

## Upstream resources

- PHYRE simulator: https://github.com/facebookresearch/phyre
- Simulator fork used by the experiments: https://github.com/vaora-proj/VAORA/tree/main/phyre
- Frozen video encoder identifier: `galilai-group/LeVJEPA-VideoMix-Large`
  (revision `e831a0347737fcaa660b39c57d41c109de399845`).

Upstream datasets, model weights and software retain their respective terms.
No third-party weights or simulator source are redistributed in this release.
