# Research code and experiment evidence

This snapshot publishes project-owned code and compact experiment records through
October 21, 2026, with a progress report dated October 22. Start with
[the progress report](docs/EXPERIMENT_PROGRESS_20261022.md). Publication date is
November 6; this export does not represent new training or evaluation.

## Latest experiment map

| Directory under `results/` | Completed work | Main evidence |
|---|---|---|
| `success_objective_loop_20261021` | 16 short updater fits; BCE, pairwise, calibration and distance supervision; readout and oracle diagnostics | `FINAL_FINDINGS.json`, `CROSS100_FINAL_TABLE.csv`, `INDEPENDENT_ANALYSIS.json`, `FINAL_INDEPENDENT_AUDIT.json` |
| `feedback_readout_loop_20261021` | Six independent ranking heads, two seeds; real/zero residual and query-only controls | `FINDINGS.json`, `CROSS100_TABLE.csv`, `AUDIT.json`, `REPLAY_AUDIT.json` |
| `support_context_ablation_20261021` | Twenty heads isolating support action, predicted geometry and relative features | `FINDINGS.json`, `CROSS100_TABLE.csv`, `AUDIT.json`, `CONTROL_REPLAY.json` |

`pipeline/` contains shared training, model, probe, scoring and inference code.
Each `results/<experiment>/` retains its original relative source layout, including
runtime snapshots and scheduler scripts. Earlier experiments and dated reports
are historical evidence, not interchangeable with the latest protocol.

## What is and is not runnable

The public summaries can be checked without downloading any models:

```bash
python scripts/summarize_latest_results.py
python scripts/verify_research_snapshot.py
```

These commands verify published per-scene metric arithmetic, CSV agreement and
export hashes. They do not rerun a simulator, independently infer scores, or verify
unpublished raw arrays. The archived audit records describe those checks performed
in the original experiment environment.

Full experiment drivers additionally need the simulator, encoder/probe/predictor
weights, classifier action pools, cached latents/observations and original
dependency manifests. Weights and binary data arrays are not included. Absolute
machine paths were redacted to `/path/to/...`; private IPs, where present, were
replaced with loopback. Source and export hashes are listed separately in
[`RESEARCH_MANIFEST.json`](../RESEARCH_MANIFEST.json). Hash guards in historical
plans refer to the original files and must not be treated as valid launch
registrations for the redacted copy. These are inspectable research snapshots,
not turnkey cluster commands. Use `phyre_feedback/` for the standalone components.

One syntactically invalid historical benchmark attempt is preserved with a
`.py.txt` extension and explicitly marked in the manifest; it is not executable.
Six large historical JSON detail files are omitted and listed in the manifest.
Compact latest results, per-scene metric audits, plans and source are included.
Some historical document links refer to large arrays, reports or presentation
assets that remain in the private research workspace.

## Validity and interpretation

The [withdrawal record](results/INVALID_RESULTS_20261016.json) applies to the old
shortlist comparisons that used a fixed older base predictor to select candidates.
Their source is retained for provenance, not as current performance evidence.
Matched pretraining errors are distinct from withdrawn ranking comparisons.

The latest studies keep the classifier candidate pool and first Q0 attempt fixed.
The zero-residual head still receives support action and predicted geometry;
all scenes being reranked have a failed first attempt. It is not a no-context or
no-failure-information control. Support geometry is predicted before a trial,
not the observed trajectory. A gain using that context does not by itself show
learning from execution feedback.

Next proposed comparison, **not run**: selected support's predicted trajectory
versus the same scene's candidate-pool mean predicted trajectory, with matched
query features, training budget and first-failure histories.
