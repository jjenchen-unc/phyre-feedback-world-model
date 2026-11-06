# Active: current cross probe adaptation (2026-10-20)

User approved adapting the current ranking probe, with observed-only further-training
control and a shared 50% observed / 50% current-prediction head. Freeze predictor,
updater, original geometry-supervision probe, encoder, shortlist rule and scorer.
No slide update, predictor training, or repo push authorized in this turn.

Root: results/current_probe_adaptation_20261020 (local and shared server).
Source scripts registered in PLAN.json. Initial ranking head is the historical
cross predicted_mix_s0 adaptation, identical initialization for both new heads.
2000 steps, batch128, AdamW3e-5 wd.05 cosine clip1; geometry-only loss unchanged.
Target draws identical. Current L1/Smooth updater checkpoints seed0; raw/centered/
zero plus Q0. No updater weights changed.

Important verified correction: CURRENT updater pilot has64train scenes on12templates,
not1200/full16. Existing classifier pools cover these64 plus100test scenes. This
adaptation intentionally uses original probe400-scene training universe (16templates),
chooses4fit+2validation/template independent of outcomes, and builds missing pools
with the SAME classifier full100k top1024 procedure. Fit64/validation32 are scene
disjoint; upstream probe saw them, so validation is adaptation-heldout only.
All96 verified in official ball_cross_template fold1 training split; test100 uses
5 disjoint templates. No success or Near filter.

Collected states: Q0 and current online old-head policy after1/3/5failures for each
L1/Smooth x raw/centered/zero; stop on success. Uniform8untried queries,8future times;
exact original32candidate forward shapes; FP32 storage avoids extra rounding.
Fixed-state validation compares all3heads on same full1024 ranks and cachedgeometry.
Gate predeclared: mixed lower predicted geometry loss, higher AUC, no worse top8
than BOTH old and true-control; equal scene/availablevariant/depth weights. Exploratory,
not statistical evidence. If pass, automatically run100cross scenes,7variants,
2newheads x ownfirst/common-oldQ0first,9attempts; oldhead reference already available.

Completed: code compile/bash syntax; official split/input96; canonical A6000 scene
encoding exactly reproduces reference32scenes. Encoder full-rollout/initial output
matches canonical cross AND within caches exactly. GPU preflight reproduces full1024
old Q0 scores; future-feature mutation leaves ranking unchanged; model unchanged.

Active chain JOBS.json:36557encoder validation and36558GPU preflight COMPLETE;
36559 pool build COMPLETE (96/96);36560cache-smoke,36561paired3stepheadsmokes
and36562smokeaudit COMPLETE/PASSED.36563collection16shards RUNNING (18/96 scenes
complete at latest direct check, all16workers healthy, ~8min elapsed). Full100k
classifier ranking96/96 complete. No formal head training or fixed evaluation yet.
Next36564dataaudit ->36565twoheadtrainings
->36566headaudit ->36567fixed-validation16shards ->36568decision ->36569conditional
closedloop32shards,max24concurrent ->36570summary ->36571afteranyencodercleanup.
Two A6000 selector workers score96scenes in parallel; two canonical encoders on
same GPUs4/5, sufficient memory headroom. Owned timeout PIDs: encoder2998800/2998803,
selector2998805/2998807 (Oprime). Encoder48h bound, selector2h bound. No otherusers'
processes touched. Never copy/read auth.key.

Preflightattempt1 retained in root/preflight_attempt_1, logs36526–36540. No probe
training occurred. Classifier A5000 replay differs up to7.63e-6 from canonicalA6000;
unsetting CUBLAS override did NOT fix it (36556). A6000 diagnostic reproduces100k
scores exactly. Corrected chain computes classifier scores on A6000 (code/score_scenes.py),
pool workers consume verified hashes; predictor/probe remain validatedA5000 arithmetic.
Old encoders stopped automatically. No tolerance weakened; original artifacts untouched.
Slurm scheduler rejects explicit mem requests (no configured RealMemory); removed only
resource directives before successful submissions, using existing cluster convention.

Next: monitor chain and external selectors; verify first pool completion, smoke gradient
checks and target-draw identity. Root/watch.py is a local same-thread completion/failure
monitor; WATCH_STATUS.json is latest observation. If failure, inspect active JOBS.json,
not olderfailed jobs; preserve receipts before any source/plan revision. No speculative
performance claims until FINAL_SUMMARY. If gate fails, closedloop intentionally skipped.

Remote project /path/to/phyre-unc; sunflower and oprime shared filesystem.
SSH flags BatchMode=yes GSSAPIAuthentication=no ControlPath=none ConnectTimeout=8.
Remote .venv-ml/bin/python; PHYRE CPU runtime/sunflower_phyre_python.

Previous completed work/state archived in this root/PREVIOUS_STATE.md. Canonical
latest completed loss results: results/loss_design_iteration_20261019/FINAL_SUMMARY.json.
Within Smooth raw seed0S9=61vsL156 butseed1=50vs58; no robustloss winner. Cross Smooth
centered15vszero14, L1centered15vszero9 on100scenes, onefold/seed. Probe training
scope audit: results/probe_cross_training_scope_20261020.json. Current slides remain
seed0 snapshot; no newslide edits. Previous loss/transfer encoder jobs cleaned up.
