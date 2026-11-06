# Idle: next-direction analysis completed and audited (2026-10-21)

Latest request: user asks conclusions from the authorized iterative diagnostic loop.
Three bounded rounds complete under results/next_direction_loop_20261020.
All Slurm jobs36662–36671 including arrays COMPLETED; squeue user jjenchen empty.
8 probe heads, discovery16, confirmation16 complete. No new predictor/updater fit,
closed-loop run, slides or repository push. Frozen PLAN/code unchanged.

Independent local audit passed:440 raw1024-score vectors,108 raw files; recomputed
stable ranking and pairwise AUC; checked every record and aggregate, same candidate
indices/labels/first action/tried masking and matched Q0/zero comparisons.
COMPLETION_AUDIT.json, independent_audit.py, RAW_SHA256SUMS.txt retained.
FINAL_SUMMARY local/remote SHA bb40b07df9ee5a677875743f0f7c172f7f6382a83a8d4a386f0059a7c18605ca.
Geometry summaries inspected, not independently recomputed from latent caches.

Design: freeze predictor/updater; Smooth-centered memory chosen on FIT64 by
residual-minus-zero top8 rule, only18vs17/53 (Q018). Four probe-training arms x2
samplingseeds20261021/22; same head initialization,2000steps each. Q0-only mixed,
memory-only mixed observed/predicted50/50, memory predicted-only, memory goal-object
only mixed. Training53 eligible scenes133 memory states1064 queries65 positives.
See PLAN and archived STATE_DURING_EXECUTION.md for exact protocol/provenance.

Discovery16 has12 first-Q0-failure scenes; confirmation16 has10. Static next8
coverage on same1024pool minus identical failed action, NOT closed-loop S@9.
Discovery top8: originalQ0 5/12; originalhead memory4/12; predicted-only adapted
memory8/12,7/12; samehead zero6/12,6/12; adaptedQ0 4/12,4/12.
Confirmation top8: originalQ0 1/10; originalhead memory3/10; predicted-only adapted
memory2/10,3/10; samehead zero2/10,3/10; adaptedQ0 4/10,2/10.
Selected predicted-only adaptation improvement does not replicate. Residual and zero
confirmationtop8 identical on EVERY scene; top1 residual advantage only1scene in
seed20261021,0 in seed20261022. No stable residual benefit established.
Memory50/50 comparator confirmationtop8 1/10 bothseeds vsold3/10 despite reported
mean geometry .6755->.6453/.6452 and taskobjects .9348->.7525/.7383. Closest-of8sampled
geometry worsens .5458->.7462/.7210; suggest time-local error matters but NOT proof
of actual contact-time mechanism. Geometry units/objective are weighted probe loss.
Task-object-only head showed discoverytop1 4/12,3/12 butzero5/12,3/12 andtop8both6/12;
it was NOT selected/confirmed. Do NOT claim this arm failed confirmation.

Interpretation: geometry-only readout adaptation is not yet a reliable fix; no proof
that latent success information is destroyed, scorer alone is at fault, or more data
cannot help. Positive ranks may come from generic update rather than residual content.
Best next hypothesis: success-sensitive training on same-scene action pairs, retaining
geometry anchor and same-first-action Q0/zero controls; fixed reader first isolates
updater change. A diagnostic ranking-head-only control would help separate readout
from representation. This is a proposed follow-up, NOT submitted or proven effective.
Do not scale geometry-only probe adaptation based on discovery gains.

Limits: one cross-model fold;32 diagnostic scenes from cross TRAIN templates, previously
examined adaptation validation; no new cross test generalization evidence. Two probe
samplingseeds, NOT independent worldmodel seeds. Firstfailure subgroup small12/10.
Predicted-only changes exposure at fixedsteps; domain specialization alsochanges exposure.
No ongoing study jobs. Monitoring completed; avoid restarting original study.
Remote sunflower:/path/to/phyre-unc; .venv-ml/bin/python.
SSH BatchMode=yes GSSAPIAuthentication=no ControlPath=none ConnectTimeout=8.
Never read/copy auth.key. Previous probeadaptation state archived PREVIOUS_STATE.md.
