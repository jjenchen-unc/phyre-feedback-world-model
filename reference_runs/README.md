# Experiment source reference

These are the training/evaluation drivers used by the milestone experiments.
Absolute cluster paths have been replaced with `/path/to/...`; executable
model/loss/policy definitions are otherwise retained. See SOURCE_MANIFEST.json
for hashes of both the original source and the exported file.

| Directory | Purpose |
|---|---|
| matched_base_q0_20261012 | Paired random-initialized Base/Q0 pretraining |
| selector_feedback_pilot_20261016 | Completed classifier-first online-support training/evaluation |
| q0_first_feedback_20261018 | New Q0-first training/evaluation, still running at release |
| oracle_small_1024_20261017 | Small paired predicted/oracle latent diagnostic |

These drivers depend on registered `PLAN.json` files, model/probe/PCA checkpoints,
latent data packs, per-task classifier candidate pools, and historical loader
snapshots. Those integration assets are not part of this public release. They
must be supplied and their paths/checksums registered before cluster execution.
The standalone `phyre_feedback` package is the importable component interface.

The oracle run's original base-score parity guard detected a BF16 batch-shape
difference against the earlier 100k ranking run. Recovery compared freshly
computed Base and oracle predictions with the same fixed batch size of 32,
retaining original scores separately. Reported results use this repaired run.

Do not execute old shortlist evaluation sections and treat them as current
method evidence. In particular, the matched pretraining driver's historical
ranking evaluation was withdrawn; only its training/dev records are retained
in the milestone release. The completed feedback pilot is classifier-first;
the new Q0-first experiment is a distinct policy with newly trained updaters.
