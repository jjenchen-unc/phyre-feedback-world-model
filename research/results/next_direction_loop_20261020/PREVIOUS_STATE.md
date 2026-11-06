# Complete / idle: current cross probe adaptation (2026-10-20)

User-approved small diagnostic completed. No active owned Slurm jobs; encoder cleanup
confirmed; both selector workers96/96 complete. Completion follow-up delivered/consumed.
No slides changed, no predictor/updater training, no repo push. No new experiments queued.

Root: results/current_probe_adaptation_20261020 (local and shared server).
Authoritative PLAN.json, DATA_AUDIT.json, HEAD_AUDIT.json, DECISION.json,
FINAL_SUMMARY.json and COMPLETION_AUDIT.json. Independent local recomputation of all
372states x3heads from1024-score arrays agrees exactly with recorded AUC/top1/top8;
scene/variant/depth macro aggregation independently matches. 404downloaded fixed files
match remote aggregateSHA896beef47b45eb03241ba422b2723350009ee2478b48236cdfbf47899780bb9e.

Fit64scenes,validation32scenes:4+2per16officialcrossfold1train templates, sampled from
originalprobe400scene universe independent of outcomes. Validation is adaptation-heldout,
not held out from upstream training. Current cross100test5templates untouched by this
adaptation's training/validation. Current updaterpilot originally64scenes/12templates;
newclassifier100ktop1024pools built to cover selected96scenes/16templates.

Freeze Q0/predictor, updater, encoder, original geometry-loss probe and scoring rule.
Onlyrankingprobe changes. Both newheads same historicalcrosspredicted_mix_s0 initialization,
same2000steps/batch128/targetdraws, geometryloss10xyL1/256+barangleL1. Observed-onlycontrol
vs50%observed+50%currentpredicted. Sharedhead acrossQ0,L1/Smooth xraw/center/zero.
Fit847states/validation372: initialQ0 and after1/3/5failedonlineoldheadupdates, stopon success.
Uniform8untriedqueries,8futuretimes; full1024ranking. Modelcheckpoint hashes unchanged.

Macro results (equal scene, then availablevariant, then depth):
old: predictedgeometry .7216142055, observedgeometry .0611950638, AUC .7326425165,
nexttop1 .3720238095, staticremainingtop8coverage .5751488095.
true_control: .7492401255, .0430044164, .7183598590, .2470238095, .5431547619.
mixed: .7077092215, .0940631530, .7197400252, .3169642857, .5543154762.
AUC31definablescenes; othermetrics32. Top1/top8 are averaged fixed-state diagnostics,
NOT new closedloopS@1/S@8. Geometryloss is probeoutputgeometry, NOT latentL1 (latentsfixed).

Important disaggregation: Q0 all32scene geometry .7451->.7083, AUC .7428->.7342.
All6memory-inputvariants have WORSE meangeometry with mixedhead (22scenes with first
failure). L1center geometry .6862->.6951/AUC .6810->.6741; Smoothcenter geometry
.6470->.6642/AUC .6882->.6920. Smoothraw/center/zero AUC smallgains, L1variants declines.
Do not describe overall geometrydrop as a general memory-readout improvement; Q0/scene
composition matters. No pairedcrossvariantpopulation claim: Q032 versusmemory22 scenes.
Mixed vsold per-sceneAUC15better/16worse;geometry20better/12worse, observedgeometry26worse.

Registeredgate requiresmixed lowerpredgeometry,higherAUC,no-worsetop8 thanBOTHold/control.
Gate FAILED. Closedloop intentionallySKIPPED; no newS@1/3/5/9. Entireboundedexperiment
complete, not blocked. Conclusion: this shared geometry-supervised readout adaptation
fails to improve overallranking; does not prove allprobeadaptation impossible, ruleout
mismatch, or establish irrecoverable predictor information loss. Onefold/seed smallpilot.
Next action none authorized; report verified results, propose furtherdiagnostics only.

Runtime: corrected chain36557–36571complete; noownedjobs. Bothheadtraining45sec each,
collection dominates. Firstpreflightattempt classifier A5000 vsA6000 differs7.63e-6;
unsettingworkspace didNOTfix. Kept full100kclassifier andencoder oncanonicalA6000,
predictor/probe onvalidatedA5000. Whole100kclassifierexact, encoderexactbothprotocols,
Q01024scores exact, futuregeometryinputmutation invariant, pairedheadtargetdrawsexact.
Attempt1 sources/receipts preserved; never relaxedtolerance orchangedoldartifacts.

Prior workingstate archived root/STATE_BEFORE_COMPLETION.md; earlierexperimenthistory
root/PREVIOUS_STATE.md. Other completedresults: loss_design_iteration_20261019 final
(Smoothwithinseed0gain not replicatedseed1); probe_cross_training_scope_20261020 audit.
Slides remain priorseed0snapshot and unchanged. Remoteproject /path/to/phyre-unc
on sunflower/oprime sharedfilesystem; .venv-ml/bin/python. Never copy/read auth.key.
