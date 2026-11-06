# Verified: cross probe training task scope (2026-10-20)

Audit: results/probe_cross_training_scope_20261020.json with full scene IDs,
manifestSHA and exact overlap checks. Originalprobe crossfold1 train manifest
_s16r16_sub400:400scenes=25each x16templates,13021rollouts;8epochs;4group
OOF template calibration thenfinalfitall400. Templates1,3,4,5,6,9,11,12,16,
17,18,19,20,21,22,24. Adapted rankingprobe trains331scenes/5289queryactions
on12templates3,5,6,9,11,16,17,18,19,20,21,24; only77scenes overlaporiginal400,
so notsimplysubset. Adaptationcalibration29scenes on1,4,12,22, no gradient.
Current cross evaluation100scenes on8,10,13,15,23: no scene or template overlap
with originalfit/adaptationfit/adaptationcalibration. No experiment changes.

# Complete: iterative analysis and completion notification (2026-10-20)

All loss/transfer stages complete, no owned Slurm jobs remain; both Oprime
encoders stopped per ENCODER_CLEANUP.json. Monitor session20370/cell809 ended,
self-followup delivered and consumed. Final/joint reports synced locally:
results/loss_design_iteration_20261019/{FINAL_SUMMARY.json,REPORT.md};
results/support_transfer_iteration_20261019/{JOINT_SUMMARY.json,JOINT_REPORT.md}.
Recomputed counts from every first_success mapping: all100/group exact.
Within seed1 raw-input L1 S1/3/5/9=29/45/49/58 vsSmooth29/40/47/50;
Smooth vsL1 S9 gains2/losses10,net-8. Seed0 was56vs61 (+5): improvement didnot
replicate. Both initialization and episode draws change, not baseQ0/fold.
No seed1 centered/zero evaluated; do not extrapolate seed1raw tocombinedmethod.
Recommendation: prioritize identifying error-content benefit with matched
zero/action/memory/update controls before scaling a selectedloss; no newjobs.
Current slide snapshot remains seed0 and needs refreshed limitation before
presentation, but no further slide changes authorized on this probe question.

Probe implementation verified against current loader and remotecheckpoint:
AttnProbe D128,S196,d256,two4headcrossattention+MLP blocks; 14D initialobjectquery,
outputxy + doubledangle. Training geometry usesoriginalprobe bothprotocols;
rankingwithin usesoriginalweights,crossusespredicted_mix_s0 (oldbase/oldmemory
predictionadaptation2000steps). Rule bothsoft tau4/s1. Frozen currentruns,
geometry gradients still pass through probe to prediction/updater.

# Complete: slides include Smooth L1 centered/zero results

Latest files: artifacts/milestone_20261018/COMP669_Milestone_20261018.{pptx,pdf},
COMP669_Milestone_manuscript.txt andSpeaker_Notes.txt. Previousversion archived
in archive_before_smooth_centered. Mainrow nowSmooth+centeredwithin61/cross15,
explicitcrosszero14 limitation; backup samecheckpoint raw/center/zero fullbudgets.
Main6rows base/Q0/AdaJEPA/L1/L1center/Smoothcenter; MSE/normmatch removedfrom
backupforfocus; preservedsource reports andarchives. Nextstepcontrolstudy marked
complete; nextmechanismquestions proposed, withinseed1 lossreplication running
at snapshot. No new experimental jobs orrepo push.
10main+3backup,1172mainwords,675splanned;VALIDATION.json passed76tablecells+4
callouts inPPTX/PDF,13exactnotes,identicalTXTexports,URLbothformats,PDFbounds.
Changed5pages visuallyreviewed,2tablesfullsize;8unchangedpages textexact.
Slide request complete; background loss experiments remain active below.

# Complete: support-query transfer investigation (2026-10-19)

All jobs 36303–36308 completed; FINAL_SUMMARY.json and REPORT.md under
results/support_transfer_iteration_20261019 verified. Same raw-trained128 updater,
Q0 first action, common classifier1024 pool, 100 scenes/protocol.
Raw/centered/norm-matched full S1/3/5/9:
cross 3/7/10/12, 3/6/10/15, 3/8/10/14;
within 29/45/50/56, 29/46/51/57, 29/48/53/59.
Norm-matched vs raw S9 gains/losses cross2/0,within3/0; pilot, not robust cause.
Cross first-only action gate S9=9 vs raw13/shuffled13/uniform11: no benefit.
Within gate skipped because train/eval association screen failed.
Fixed first-query diagnostic eval64queries: cross wrongdirection20/overshoot21,
within13/25. Prediction-space MSE decomposition, not proved memory-space cause.
Small fixed-history comparison did not distinguish next-success counts.
36309 joint report still waits loss chain. Encoders remain owned by loss chain;
36297 cleanup waits both studies. No slides or repo push.

# Active: loss-design iterative verification (2026-10-19)

User authorized proposal, execution and several result-driven iterations; no deadline.
Scope bounded at two training rounds, max8newupdaters. Newroot:
results/loss_design_iteration_20261019. Allcode registered; Slurmchain submitted:
36283canonicalencoder validation ->36284twoGPU loss/gradient+rolloutsmokes
->36285smokesummary ->36286twoGPUfixeddiagnosis/calibration
->36287fourtrainings ->36288audit ->36289raweval32shards ->36291decision.
36290newmodel fixedquerydiagnostics followsaudit;36292center/zero64shards and
36293conditional4seed1trainings followdecision;36294audit2->36295eval2;
36296FINAL_SUMMARY+REPORT dependsalleval/diagnosis;36297afteranycleanup.
OwnOprimeencoderPIDs3736276/3736279 launcher,GPUs6/7 sharedidle,20hbounded.
Never copy/print auth.key. No otheruser process stopped.
Stage0 fixed-history train8/eval8 cases perprotocol: samefirstfailedsupport and
8queries; compareQ0/rawL1/rawL1centered/centertrained, error+ranking. Training-only
SmoothL1beta andlossstate-gradient-scale calibration, fixed beforetraining.
Round1 freshMSE andSmoothL1 cross/within, rawinput,128slots, sameQ0andinitializer.
ReusedL1uniform128 control requires exactloss/gradient/policy replay andinitcheck.
Round2: perprotocol candidate max(S9,S5,S3),L1preferredties. If S9>L1 andS5>=L1,
replicate candidate vsfreshL1 withupdaterseed7920 andindependentepisode draws;
Q0/foldunchanged. Else skip extra training. Always testMSE/SmoothL1centered/zero
inputs andfixed-query errors. Automatic FINAL_SUMMARY.json/REPORT.md andcleanup.
36283encoder validation COMPLETE;36284bothsmokes and36285summary COMPLETE.
36286 diagnostics and all four 36287 trainings COMPLETE; training walltimes
crossMSE2h47m/crossSmooth2h57m/withinMSE2h33m/withinSmooth2h30m.
All 36288 audits and 36290 fixed-query diagnostics COMPLETE.
36289 raw evaluation COMPLETE, all four methods100/100; 36291 ROUND1 and
DECISION complete with full score/pool/provenance audits passed.
Cross L1/MSE/Smooth S1/3/5/9=3/7/10/12,3/7/8/8,3/7/10/12.
Within L1/MSE/Smooth=29/45/50/56,29/42/50/56,29/46/51/61.
Smooth vs L1 S9 gains/losses cross4/4,within8/3; singlefold/seed evidence.
Decision: no cross replicate; within pairedfresh L1 vsSmooth seed1.
36292 all input controls COMPLETE; all800newrollouts100/group. Independently ran
remote evaluate.collect on all12 raw/centered/zero groups: score/pool/selection/
provenance/unchangedmodel checks passed. INPUT_CONTROLS_SNAPSHOT.json saved.
Smooth raw/center/zero S1/3/5/9 cross3/7/10/12,3/9/12/15,3/10/12/14;
within29/46/51/61,29/48/54/61,29/43/53/57.
Centered vszero S9 cross1gain/0loss;within7gains/3losses. Zero retains attempted
action+memory/update scheduling; not no-feedback/no-update. No robust content
benefit claim from cross1scene. MSE center/zero cross3/7/9/10 both;
within29/42/52/56 vs29/39/49/54.
36293_2/_3 seed1 within L1/Smooth stillRUNNING ~47min,28/128slots both,
26optimizersteps. No seed1 evaluation yet. Slides NOT updated with these later
input controls (last slide snapshot remains raw seed0loss).
Cross beta.055252,MSEscale.821289,Smoothscale1.048265;
within beta.068665,MSEscale.943775,Smoothscale1.057155.
Next: monitor input controls and within seed1 replication -> joint report. No new causal conclusion from single seed.
Do notclaimnewresults untilreceipts; branch/analysis runsautomatically. No slides orpush.

Preflightattempt1 failed due common.py shadowing originalmodel commonmodule;
no training executed. Renamed helper loss_runtime.py, imports verified through
fullsource dependencychain. Oldplan/jobs/code retained preflight_attempt_1.
Oldencoder services stopped; replacement services andjobchain above active.

# Complete / idle: centered-residual updater training (2026-10-19)

Both fresh trainings finished128slots:36207_0cross2h57m,36207_1within2h35m
(jobwalltimes includepreflight). Sameinitial/reader/draws/activepattern/optimizer
steps asrawuniform128 verified; cross122steps/192activeepisodes,within120/186.
Bothaudits36208/36210 andall48evalshards36209/36211 COMPLETE.
36212SUMMARY passed600rollouts (100scenes x2protocols x3inputs).
Evidence: results/centered_residual_training_20261019/{SUMMARY,COMPLETION_AUDIT}.json.

S1/3/5/9 cross newlycentertrained/centerinput3/7/10/10; newrawinput3/7/8/9;
newzeroinput3/9/9/10. Rawtrained/raw3/7/10/12;rawtrained/center3/6/10/15;
Q03/7/8/11,base3/4/7/7,tta3/5/6/9.
Within newcenter29/47/51/56,newraw29/44/50/56,newzero29/46/49/54;
oldraw29/45/50/56,oldcenter29/46/51/57,Q029/42/47/50,
base26/36/39/46,tta29/45/51/58.
Interpretation: centeredtraining didNOT improveS9 vsrawtraining on thispilot:
cross12->10,within56->56; inference-onlycentering remains15/57. Within S3/S5
+2/+1 vsraw. Onefold/seed perprotocol; no universal failure orcausalproof.
Newcenter vsnewzero S9cross10/10,within56/54; semanticfeedbackbenefit notproved.

Allownjobsfinished. Cleanup36213 initiallyfailed60sstopconfirmation; laterboth
ownedOprimeencoderPIDsabsent,STATUSstopped. Confirmationrerun passed and
ENCODER_CLEANUP.json recorded. No resource remains active. No slides orpush.
Next action: reportresults; furthertraining requiresnewinstruction.
Historicalstate below retained; its earlierproposals/status are superseded.

# Complete / idle: milestone slides updated with latest comparison (2026-10-19)

User explicitly authorized currentresults in slides; supersedes earlierkeepout
for this comparison. Artifacts artifacts/milestone_20261018: currentPPTX,PDF,
manuscript.json,COMP669_Milestone_manuscript.txt,Speaker_Notes.txt,VALIDATION.json.
Archived priorversion in archive_before_base_centered_results (alloldfiles).
10main+3backup unchanged. Slide6 defines5methods;7mainS9tablebase/Q0/tta/
uniform128/centered;9nextcentered-inputtrainingproposal;10takeaways;11fullS1/3/5/9
backup replaces outdatedpairedcounts;12training128slots. Englishscriptsupdated.
CurrentcrossS9 7/11/9/12/15;within46/50/58/56/57. Centering inferenceonly sameckpt;
baseownfirst,othersQ0first. Onefold/seedpilot,notrobustwin claim.
All50resultcells verified inPPTXandPDF againstSUMMARY;13notes exact,URLhyperlinks
presentbothformats,all13pages reviewed/changed6pages fullsize,noclip.
1157mainwords;675secplanned (11m15s),45secreserve,notmeasuredrehearsal.
No newtraining orrepo push. Olderexperiment audits saying slidesunchanged refer
to their earlier completion snapshots; currentauthorizedrevision isVALIDATION.
No pending work. Deliver currentPPTX/PDF/manuscript links.

# Complete / idle: original no-Q0 base comparison (2026-10-19)

User asked to include original base. Completed200sameclassifier1024poolscenes,
crossfold1/withinfold0. No newtraining/encoder/simulation. Twofullscoreexact
smokespassed (basefastvslegacy andQ0vssaved); all200scores locallyverified.
Report: docs/PLAIN_BASE_COMPARISON_20261019.md.
Results/audit: results/plain_base_comparison_20261019/{SUMMARY,COMPLETION_AUDIT}.json.
Base/Q0pretrainmatchedscratch8epochs,initialbackbone/draws/presentations/steps
exact; PRETRAINING_MATCH.json. Crosslambda0,within.03,seed0.

S1/3/5/9 cross base3/4/7/7,Q03/7/8/11,tta3/5/6/9,
uniform3/7/10/12,centered3/6/10/15.
Within base26/36/39/46,Q029/42/47/50,tta29/45/51/58,
uniform29/45/50/56,centered29/46/51/57.
BasePRIMARY uses ownfirstaction. ExtraQ0-firstthenbaserank controlcross3/7/10/10,
within29/39/41/48. Do notclaimallgainvsbasecomesfromfeedback orsamefirstbase.

Jobs36184smoke,36186smokesummary,36187full16GPU,36188summary allcomplete.
Noownactivejobs. No slideschanged orpush. Centered-inputtraining stillproposal,
NOT launched. Overnightreport/auditkeptunchanged; newcomparisonseparatereport.

# Complete / idle: PHYRE overnight investigation (2026-10-19)

User deadline was TODAY 2026-10-19 09:00America/New_York (13UTC), overriding
initialgoalmetadata Oct20. Eight rounds complete beforedeadline; fourfreshupdater
trainings,1800newclosedlooprollouts, fixedhistory and260initialresidualcases.
Finalreport: docs/MEMORY_OVERNIGHT_INVESTIGATION_20261019.md.
Evidence: results/memory_overnight_investigation_20261019/PLAN.json,
COMPLETION_AUDIT.json, PAIRED_ANALYSIS.json,FRESH_TRAINING_AUDIT.json,
FRESH_PANEL_cross/within.json,FIRST_UPDATE_cross/within.json.

Recommendation (proposal only, NOT submitted): retainuniformqueries andtrain
freshupdater withresidualcentering enabled intraining/evaluation; twoarmspanels,
matchedraw/centered/zero controls. This iscandidate direction,notcausalrootproof.
Fresh128 S9 crossQ011/raw12/center15/zero9;within50/56/57/55.
Balancedquery rawcross12/within58,center11/56: no consistentbenefit fromraising
positivequeries to~50%. Uniformcenteringhelpsat128and256inbothprotocols,
butsameinit/evalpanels NOT independentreplication. Withinzero retainsmuchgain;
do notattributeallbenefit tophysicsfeedback. First-update accuracydoesnot
predictallclosedloopbenefit. CrossfirstsupportL1train->eval+22.6%,withinflat;
this isQ0supporterror,notpostupdatequerylossoroverfittingcausalproof.

Alljobscomplete; noownqueued/runningSlurmjobs. OwnOprimeencoder0/1stoppedandPIDs
absent; receiptENCODER_CLEANUP.json. Do notprint/copyauth.key. Slides/manuscript
sixhashes unchanged; newfindings NOT addedtoslides. No gitpush/publication.
No activeexperiment remains. Nexttraining requiresnewtask, notautomaticlaunch.

# Historical reference

Dated records below preserve earlier context and may contain superseded active
headings. They are not a current job queue; use the completed report above and
actual code/results before reusing any prior conclusion.

# Complete / idle: conversational manuscript embedded in milestone slides (2026-10-19)

User confirmed VLA-JEPA paper-sharing manuscript as style reference (despite
initial COMP790 name; reference deck labeled COMP669). Rewrote13English scripts
with short spoken paragraphs, figure-guided explanations and natural transitions.
Current artifacts/milestone_20261018/manuscript.json is canonical; build_slides.py
embeds scripts as plain SpeakerNotes, with evidence only in slide_manifest.json.
COMP669_Milestone_manuscript.txt and Speaker_Notes.txt are identical rehearsal
exports, with slide titles/time targets and three clearly labeled backup scripts.
PPTX/PDF regenerated; all13 notes exactly match manuscript, and all13PDF pages
pixel-identical to prior deck. Visible slide XML/relationships unchanged.
10main slides,1118spokenwords;675sec target (11m15s),45sec reserve under12min;
110wpm suggests10m10s speech plus pauses, not a measured rehearsal. Backup3pages
excluded from time. Prior deck retained in archive_before_manuscript.
VALIDATION hashes current. Residualfusion results remain separate, not in deck.
No jobs, no repo publishing, no pending work. Deliver PPTX and manuscriptTXT.

# Complete / idle: milestone clarity refinement and diagnostic record (2026-10-19)

User requested separate experiment record and slide refinement; NO fusion results
in slides. Wrote docs/RESIDUAL_FUSION_DIAGNOSTIC_20261019.md with setup, measured
numbers, parity checks, scope and limitations; all evidence links resolve.
Updated artifacts/milestone_20261018/{build_slides.py,PPTX,PDF,Speaker_Notes.txt,
slide_manifest.json,VALIDATION.json,slide_*.png,preview_*.jpg}. Prior version in
archive_before_control_clarity. Slide4 explicitly defines Q0 and Q0-first action;
slide5 defines residual and attempted-action input; slide6/7 Zero-residual control
label replaces ambiguous No-error, action still present, separately trained zero
explained in notes; slide7 title scoped to pilot. Slide9 proposed ranking-loss,
action-to-action transfer and seed/fold experiments, no new diagnostic numbers.
10main+3backup,560sec target,1090spokenwords. All13 contact-sheet pages reviewed;
changed4/5/6/7/9 full-size reviewed. Main result8cells unchanged and verified;
PPTX/PDF hyperlinks valid. No new experiments or repo push. No pending work.

# Complete / idle: residual/action fusion diagnostic (2026-10-19)

User-authorized read-only diagnostic completed. No retraining or simulation.
Artifacts: results/residual_fusion_diagnostic_20261019/{PLAN,SUMMARY,STATUS,
VERIFICATION}.json, cases/*/*/{result.json,update_*_scores.npz}, code/diagnose.py.
Sunflower36004_0..3 and afterok36008 all COMPLETED exit0; wall12m35–13m57.
All artifacts retrieved. 10cross+10within first-failure scenes, template-stratified
hash ordering without later gain/loss selection. First plus fourth/latest earlier
update:40states. Same current memory/action/tried mask and original real-policy
history across interventions, same residual-trained checkpoint (not trainedzero).
All40 before/after real score vectors exactly reproduce saved original evaluation;
instrumented real update exactly equals actual updater. Checkpoint hashes intact.
Local independent recomputation validated240variant score arrays (top1/top10/AUC).

Findings (first-update medians, cross/within): projected residual centeredRMS /
action centeredRMS .453/.409. Zeroing residual changes postLN tensors by relative
L2 .349/.306 and update delta by .408/.864. Thus residual affects updater; no
support for complete erasure by additive fusion/LN. Across20states/panel, zeroing
changes next action10cross/16within. Shuffling pooled time-space residual content
at fixed positional embeddings changes update delta only .294%/.647% first and
.304%/.387% later. Next action changes0/20cross,2/20within. This is weak sensitivity
to explicit residual-position alignment in this sample, not proof no geometry
information: feature content can carry position; attention aggregates a set.
Time/space embedding RMS ~.0021/.0022 versus projected residual ~.110/.084 and
action~.269/.229. Small positional amplitude is a hypothesis for weak alignment
sensitivity, not established cause or evidence concat fixes it.
First-update median AUC change vs preupdate: real -.0784cross(6defined)/-.0628within
(10defined); zero -.0656/-.0185. No global success claim; later within real has2
nextaction successes vszero0. Selected first-failure small panel, two states/task
are correlated; interventions may be OOD. Pool abs-retention~.73–.74 measures
cancellation only, not percentage task information lost. No new closed-loop S@9.
Next suggested investigation: position/content-preserving controls before any
architecture retraining. No additional experiment launched or pending.

# Complete / idle: corrected public release and milestone slides (2026-10-18)

Verified authenticated API user.login == user-specified jjenchen-unc BEFORE
publication. Correct PUBLIC repository created and pushed:
https://github.com/jjenchen-unc/phyre-feedback-world-model
API owner jjenchen-unc, isPrivate=false, default main. Local and remote SHA both
c3dabc09fcde47cc813d573c207b46ec694d0c25. Working tree clean; origin correct.
Commit author uses verified account noreply. No code changes since tested export.
Erroneous earlier repository under other account was changed to PRIVATE and
left intact; do not publish there or reuse its URL. No deletion performed.

Local code export phyre-feedback-world-model:28curatedfiles, core5CPUtests passed,
result arithmetic and17export hashes verified; external weights/data excluded.
Final slides in phyre-unc/artifacts/milestone_20261018:
COMP669_Milestone_20261018.pptx, PDF, Speaker_Notes.txt, build_slides.py.
Top-down revision completed per user approval (2026-10-19):10main+3backup,
560sec(~9–10min) talk. New explicit hypothetical predicted-contact/observed-short
schematic before pipeline. Function-first labels for classifier/predictor/memory/
readout; architecture retains editable proposal-style tokens and now explains
training on different actions in same scene, prediction loss not direct success.
Main result is first result slide, familiar/within column before newtypes/cross:
Q050/11,res57/8,zero50/13,TTA58/9. Oracle follows as separately sampled diagnostic,
explicitly unavailable during deployment; only Q0 vsrealfeature bars60/96within,
20/100cross. Coverage moved to backup3; paired outcomes/training details backup1/2.
Old updater narrative remains absent. PPTX/PDF/notes/manifest updated together;
all13pages visually checked and actual displayed result columns source-verified.
Repo hyperlinks correct; VALIDATION.json hashes current. Prior deck archived in
archive_before_topdown. User-facing comprehension not empirically tested.
No pending slide work, no new experiments or repo push in this revision.

# Complete: Q0-first updater training and evaluation (verified 2026-10-19)

All4training arms35963 completed exit0 (~4h58–6h03), audit35964 completed0.
All32evaluation shards35965 and finalaudit35966 completed0. STATUS complete,
training_complete=true/evaluation_complete=true/noerror; complete800method-task
results. Audit finished2026-10-19 02:32UTC (Oct18 22:32EDT). No pending jobs.
SUMMARY/STATUS/TRAINING_SUMMARY/EVALUATION_COMPLETION and832evaluationJSONfiles
mirrored locally. Independently checked32complete shard receipts/provenance,
all per-task S1/S3/S5/S9 counts for100scenes per method/panel.
Policy: classifier100k->legal1024, Q0+scorer chooses FIRST attempt in train/eval;
then own feedback ranking; max9attempts. Same initial Q0 across4methods.
Cross S1/S3/S5/S9:
q0 3/7/8/11;residual3/4/7/8;zero3/7/9/13;tta3/5/6/9.
Within:
q0 29/42/47/50;residual29/40/49/57;zero29/42/46/50;tta29/45/51/58.
Paired gained/lost atS9 vsQ0:crossres2/5,zero3/1,tta2/4;
withinres12/5,zero7/7,tta16/8. Within residual +7points is pilot evidence,
cross residual -3points. No robust/general benefit or causal mechanism established.
Allsameone-seed pilot scope. Classifier-first parent remains separate below.
Slide deck has now been updated with these completed results per user request.
Public code release README/results still reflect earlier classifier-first
snapshot; no repository update/push performed in this slide-only turn.
No new experiments launched.

# Latest verified: updater pilot training AND evaluation complete (2026-10-18)

Array35925 and completion callback35926 finished. STATUS stage=complete,
training_complete=true,evaluation_complete=true,error absent. All800method-task
results complete: cross/within100scenes x q0/residual/zero/tta;32shardreceipts.
Completion callback verified provenance, candidate IDs/labels, action argmax,
no duplicates, shared first action, success stopping, exact model reset.
S9 /100: cross q0=11,residual=9,zero=12,tta=10;
within q0=53,residual=55,zero=54,tta=57.
Small differences, not evidence of robust benefit; cross residual below Q0.
SUMMARY.json, STATUS.json, sunflower_backend/EVALUATION_COMPLETION.json synced
locally under results/selector_feedback_pilot_20261016. Encoder STOP requested
by completion callback. No pending training/evaluation in this pilot.
Next: analyze paired successes/failures and update magnitude if requested.
Historical running/next-action notes below superseded by this verified header.

# Active: updater evaluation running (2026-10-18)

User askedfinishfixandverification. DONE; resumedpriorauthorized4updatertraining.
Transport6videoparity+4concurrentclientsPASS, bothA5000reader/scorerexact(max0),
all4fullpreflightsPASS. Finalvalidation35915 complete39sec, preflight35916 all0:
crossresidual366.5s,crosszero295s,withinresidual250.1s,withinzero159.7s.
Canonicalencoder median.955s/video; transport/codecmedian.122s,max.141s with
nowdedicatedA6000 vsold~30sfilequeue. Sixlocalprotocoltestspass.
Receipt results/selector_feedback_pilot_20261016/sunflower_backend/VALIDATION_SUMMARY.json
syncedlocally with all4preflightreceipts, code, policy, logs (EXCLUDINGauth.key).

Formaltrainingarray35920_0..3 COMPLETED, all exit0; finishcallback35924 COMPLETE0.
All4 receipts:256/256slots, checkpoint hashes verified by callback; latest remote
STATUS training_complete=true,evaluation_complete=false. Cross residual/zero each
222optimizersteps,4h04m38s;within residual/zero each248steps,4h41m59s/4h41m53s.
Local receipt: results/selector_feedback_pilot_20261016/sunflower_backend/TRAINING_SUMMARY.json.
Training finished; formal evaluation array35925 submitted and first16 RUNNING on
Sunflowernode001/node002.32shards,4perarm,concurrency16;cross/within100tasks each,
q0/residual/zero/tta,budget9,800method-task evaluations. Callback35926 afterany
validates32shardreceipts + every action/rank/label/reset then writes summary and
stops encoder. EVALUATION_JOBS.json synced locally. No newQ0training.
Previous3smokerows archived in sunflower_backend/archived_eval_before_20261018_102417.
Current evaluation starts clean with trained checkpoint and backend provenance.
Canonical A6000 GPU4 encoder restarted, launchPID2262527,24hbounded.
Frozen runtime/encoder/model/loss unchanged; new sbatch + completion script only.
Oracle35taskdiagnostic COMPLETED(exit0,alltasks); analysis verified below.
Q0100k COMPLETE:crosscoverageQ051/base54/scene61;within91/94/98 per100tasks.
Nootheruserprocesses touched. Authkeyprivate rpc/auth.key,gitignored;NEVERsync/log.
launch_rpc.py supports guarded--restart onlyafteroldPIDgone,24hdefault,waitready.
RPCcodehash8b5558ce...,runtime0ae3e91d...,policy909bf0d6... frozenwhiletraining.
Callbacks/launch scripts distinctfrom model/loss/frozenexperimentcode.

Fixes: authenticatedTCP losslessNumPy removesNFSrequestpolling; canonicalA6000
encoder env unchanged(noCUBLAS/DISABLE), predictorA5000CUBLAS4096:8. Onlyimmutable
observationscached, residualrecomputedwithcurrenttokens. Testfixturefixes:
initialscene int64colorcodes validated0..7 ->losslessuint8; eightfailurestress
selectsfirsteligible registeredtrainingpoolonlyiflastnaturalscene lacks8failures.
Trainingmodel/loss/sampler/query/actionselection UNCHANGED. Preflightstillchecks
checkpointloss/gradientexact,currentresidualreplayexact,reader/headfrozen,
8supportBPTT,zeroresidualinvariance,TTAupdate/reset. Firstattemptreceipts archived.
35906startupfailure;35913expiredservicefailure;35914cancelled. Allsuperseded.

Next: monitor35925 and callback35926; ensure all800results validated before
reporting fullcomparison. Oracle analysis verified below.
Do notextrapolate2hotcachepilot slots tofulltrainingETA;actualjournalsnowavailable.
Historicalnotes below maycarryobsolete running/deferredstatus; thisheadercontrols.

# Completed oracle diagnostic: verified results (2026-10-18 UTC)

35/35 tasks complete, both workers exit 0; score-file hashes and first-success
ranks checked. Same classifier-selected top1024 candidate IDs/order as original
100k experiment, verified for all35 scenes. Static reranking S1/S9:
cross(n10): base10/10%, Q0 0/20%, oraclelatent80/100%, truegeometry100/100%.
within(n25): base20/40%, Q0 28/60%, oraclelatent92/96%, truegeometry96/100%.
AUCCESS100 base/oraclelatent: cross .15867/.95239, within .46630/.95634.
Same frozen probe/scoring rule; oracle replaces future latent with encoded true
video, common initial latent. Main bottleneck is predicted representation or its
compatibility with probe; this intervention cannot uniquely separate them.
Probe/scorer work much better on true latents, but are not perfect.
All selected pools contain successes (also in original coverage records); do not
extrapolate sample100% coverage to full panel. Small sample, not full100k oracle.
Original cached positive counts differ on a few marginal actions; all methods
here use common freshly simulated pool labels, checked against oracle.
Archived-vs-fresh base numeric batching discrepancy retained in recovery/REPAIR;
comparison uses common fixed32 batching, no silent tolerance relaxation.
Evidence: results/oracle_small_1024_20261017/SUMMARY.json and recovery/.

# Active priority: small oracle attribution diagnostic (2026-10-17)

User authorized small verification, ONLY1024 oracle candidates/task. Clarified
primary question: is poor BASE ranking caused by predictor, geometry probe, or
scorer? Q0-related training may proceed concurrently but diagnostic has priority.
Scope: cross fold1 5templates x2scenes=10; officialwithin fold0 25templates x1=25.
Deterministic SHA256 selection within existing100scene/panel eval sample, no
outcome selection. Official complete test split is500tasks perpanel, not100;
cross has5testTEMPLATES, within25. New diagnostic35scenes total.

New runner pipeline/oracle_small_1024.py, frozen remote copy and PLAN under
results/oracle_small_1024_20261017. SAME classifier100k->legal top1024 pools.
Primary paths: base->fixedprobe->fixedrule; truefuturelatent->sameprobe->rule;
actualsimulatorgeometry->same rule. Q0/staticselector secondary controls.
Per-action latentL1, predicted-objectxyMAE, surface distances/MAE, success/failure
strata; staticS1/S9/S100/AUCCESS100/AUC +forcedselectorfirstS9. Poolcoverage ceiling
explicit; cannot establish full100k selector quality or unique causal attribution.
Base/Q0 probe+rule exact identity checked; freshbase score must equal savedfull100k
subset exactly, diagnostic readout must reproduce scorer exactly. CanonicalA6000,
float64simulationactions, commoninitiallatent preserved. No optimization/training.

Plan frozen before simulation. Chunk32 atomic checkpoints, tasklocks permit2GPU
dynamic scheduling,16hboundedworkers. Controller launch.py GPU4/5 freeguards,
first smoke32actions perpanel then ONLY on pass launch2formalworkers. Otherusers
untouched. Attempt1smoke caught emptyQ0chunkloader, failed before trajectories or
formalworkers; archived under attempt1_failed_smoke. Fixed via existingteam.load.
Attempt2 SMOKE PASSED bothpanels32actions:cross45.86sec,within34.84sec,
including encoder+base+probe diagnostics. Basefull100k score parity exact;
base/Q0probe identity exact; diagnostic scores exact. Local metric edge checks pass.
Original workers1064238/1064240 BOTH EXITED1 at archivedbase score exact check,
NOT running continuously. One cross scene00008:532 complete, cross46chunks,
within1chunk =>47/1120chunks. Failure task00008:144 chunk448, action82942:
old100k batch1 score2.944629575947819 vs currentbatch32 2.9449668413386965.
Diagnostic repeat atB1 reproduces archive exactly. Other2/1024differences ~1ULP;
full ranks differ. Do NOT loosen tolerance or silentlyreusearchivedbase aspaired
freshbase. Engineering diagnosis in recovery/BATCH_DIAGNOSIS.json.

Recovery registered in recovery/REPAIR.json with originalplanhash, runtimehash
and47reusablechunkhashes. Originalplan/code/failedlogs preserved. Newruntime
recovery/oracle_recovery.py computes base andoracle with same32batch shape;
oldbase scores retained separately with maxdiff/count/ranking-change audit.
Basepred repeat exact and geometry-readout score exact remain required.
Existing47chunks passedstrictoriginalparity, can reuse without simulation.
Recoverycontroller2432242 targetedsmoke PASSED crosschunks448/672/864 and
freshwithinchunk32; times46.45/42.09/35.42/37.25sec. RecoverySMOKE receipt synced.
Formalworkers2442662/2442663 launchedGPU4/5; STATUS=running,16hbounded.
51/1120chunks available at resume; followworkerlogs toconfirmcontinuedprogress.
Originalfirstresult valid: task00008:532 firstsuccess base34,Q078,scene22,
oraclelatent1,truegeometry1; AUC.30765/.44173/.75564/.99778/.99792 respectively.
Basepred decodedxyMAE42.56px vsoracle2.06px, distanceMAE55.56 vs2.84px.
ONE selected scene only, not aggregateevidence. Report caveat withnumbers.
SunflowerQ0 full100k SUMMARY complete=true verified2026-10-17; summary synced.
Top1024coverage cross Q051/100 vsbase54/100 vs scene61/100; withinQ091/100
vsbase94/100 vs scene98/100. S1 cross all1%; withinQ016/base21/scene19%.
No newQ0training. Firstoracle scene scorehash and5rank/AUC values independently
recomputed andexact;23successfulactions/1024. No aggregate conclusion.
Next after recovery: monitor STATUS, logs and SUMMARY; analyze paired paths.
Updatermigration remains deferred pending transport bottleneck. OriginalETA no
longer applies due unnoticedstop; provide remainingETA after confirmedresume.

Oprime six-GPU recovery35360 CANCELLED (not started) to prevent duplicates.
Q0 Sunflower35860 was throttled32 and shards32–39 REQUEUED preserving partials;
now restored throttle40 after oracle timing moved to Oprime. Q0 ranking continues;
summary35861 retains original afterok dependency. Restore throttle40 when slots
no longer needed. No model experiments/checkpoints deleted or reinitialized.

A5000 encoder check35902 completed: initial-image latent EXACT with
DISABLE_ADDMM_CUDA_LT=1, but full trajectory encoding NOT canonical with either
setting (meanabs.0025–.0043,maxabs up to2.23). Do NOT mix those targets into
canonical observation caches. Evidence sunflower_backend/ENCODER_CHECK_0/1.json.
Use A5000 for predictors/updaters and A6000 for observation encoding.

Execution draft under results/selector_feedback_pilot_20261016/sunflower_backend:
runtime.py adapter preserves frozen model/loss/sampling; only replaces timing-only
preflight initial_scene_encode call with RPC. Adds backend provenance. Not yet
submitted/validated; ENCODING_POLICY.json not created; updater still NOT started.
encoder_service.py provides canonical initial/rollout/scene_measure RPC, parent
plan hash and service hash checked,15minrequesttimeout,8hbounded service lifetime.
Two guarded services on Oprime physical4/5 were STOPPED via STOP_ENCODERS.json;
controller1014309 and children1014348/1014349 exited. Cross-host filequeue took
~30sec/request despite ~1.5–2sec encoding; Sunflower cost35904 CANCELLED.
Do not restart draft RPC training as-is: transport overhead unresolved.

Oracle direct cost pilot COMPLETED on Oprime physical4/5, bounded600sec workers
1026960/1026962, results/oracle_diagnostic_20261017/benchmark_local.py.
Both32-action panels simulated/encoded NEW observations, CPU prefetch overlaps
GPU encoding, canonical A6000 encoder and common initial-latent exact check.
Receipts and scores synced locally under benchmark_local/{cross,within}.
Warm mean pipeline: cross1.113544sec/action,within1.380638sec/action;
encoder1.022360/1.340709sec. Full32 wall44.463/51.681sec excluding predictor load;
batched32 scorer1.483/.202sec. Only ONE task per panel sampled, so task-duration
variation remains unknown. This is a timing pilot, no representative result.
Two A6000, one panel each:20tasks x1024 ideal~3.9h, budget4–6h;
200tasks x1024 ideal~39.3h, budget2–3days. Full100k200tasks contains
cross7,056,401 +within7,275,392 =14,331,793 legalactions;
fixed-panel allocation bottleneck~116days, budget4–5months. Excludes queue time;
no claim A5000 fullvideo throughput/canonical equivalence verified.
Cost pilot complete. User now selected small35task scope; see active header.
Updater remains deferred until oracle diagnostic is secured.

Explained to user: currentbase/Q0 scores are latent->geometryprobe->surface
distance->fixedlog_score, not directsuccessclassifier. Oracle latent throughsame
probe/scorer is scorer-limited diagnostic, NOT guaranteed successupperbound.
Candidate pool remainsclassifier-selectedtop1024 fromfull100k until explicitly
changed. Training/eval firstattemptselectorargmax, laterrerankremaining. Oracle
proposal should compare samepool base/Q0/oraclelatent and optionaltruegeometry
control toseparate predictor vsprobe vsdistance-rule errors; no oldinvalidmetrics.

# Active: Q0 full100k selector comparison (2026-10-17)

User explicitly requested Q0 rank the SAME official100k actions, not only rerank
classifier1024. Separate run results/q0_coverage_100k_20261017; maintained runner
pipeline/q0_coverage_100k.py, frozen code copies coverage renderer/encoder/scorer.
Cross fold1 q0_g0_s0,within fold0 q0_g0.03_s0; same100tasks each and common legal
mask from completed base coverage. Use learned initial tokens without supports,
feedback, parameter updates, or retraining. Render/encode all valid actions;
never restrict to an existing shortlist. Older Q0 scores/latents used ONLY as
numerical validation fixtures; invalidated old method conclusions stay invalid.

UNC SSH restored after user requested retry. Sunflower shows5nodes/40idleGPUs.
Smoke array35859 PASSED both panels: exact reference scores (maxerror0), z0/features
exact, fixed initialtokens, common validIDs exact. Raw512 cost20.58s cross/30.25s
within onA5000. Full ranking35860 all40workers RUNNING; summary35861 pending
afterok ranking. Plan/JOBS
under new run. New summary compares base/scene/Q0 S1,S9,S100,top1024coverage on
same labels/legal candidates. Parent completed score files remain untouched.
No change to currently submitted classifier-selected updater pilot.

Local compile, sbatch syntax and synthetic ranking/audit checks passed: success
outside base top1024 retained byQ0, illegalhighest-score excluded, changedlegal
mask and alteredscorehash rejected. GPU smokes test Q0 score parity, canonical
z0/features, initialtokens unchanged, common legalIDs. Both smoke/q0_<panel>.json receipts passed. Fullrun uses batch32 and original single-image
encoder; approximate scale of prior fullbase run, no new speed claim. Each512raw
chunk resumable with plan hash and cumulative elapsedtime. No modifications to
frozen code after PLAN registration. Next: monitor40worker completion and summary;
fix only demonstrated failures, preserve source plans/results.

# Active: full100k candidate coverage comparison (2026-10-16)

User wants NEW trained base predictor+scorer vs initial-scene classifier ranking
ALL100000 official cache actions and keeping1024. Primary metric: fraction of
same100eval tasks perpanel with >=1successful retained action. User explicitly
chose cross base lambda0 / within base lambda.03, seed0 as starting configs;
not optimality claims. Classifier matched-s16r16 existing seed0 checkpoints.
Runroot results/candidate_coverage_100k_20261016. New pipeline sources
pipeline/candidate_coverage_100k.py and candidate_coverage_100k_cpu.py.
No historical-model candidate preselection. Both methods' final top1024 use
same simulator-valid action set; stable original action-index tie breaking.
Scene classifier scores all100k; base renders valid actions, encodes initial
image canonically, predicts and scores batch32. Cache labels used for coverage,
not new full-rollout simulations. Independent training/representation differences
remain explicit. Historical shortlisted arrays are numerical test fixtures ONLY;
old cancelled jobs stay cancelled, invalid method conclusions stay invalid.

All four numerical smokes PASSED with exact reference scores; base z0/features also exact.
Coverage COMPLETE: all400 model/task records and SUMMARY.json complete=true.
Top1024 task coverage: cross base54/100,scene61/100; within base94/100,scene98/100.
All100tasks perpanel have a success in full legal100k. Selector winner BOTH scene.
Measured per-task mean walltime: cross base4512.87s,within4574.08s on A5000;
classifier cross8.30s,within8.36s on A6000 (cached initial scene tokens; excludes
initial encoding/legal filtering). Base medians4560.88s/4674.03s, overall range
2648.92–6058.30s.200base tasks ~252.4worker-hours across40workers, ~6–7h wall.
Code inspection: base predictor/scorer batch32 but inherited recovery_runner.Encoder.initial
loops SINGLE IMAGE; renderer serial peraction,512raw chunks written/read through NPZ,
no CPU/GPU overlap. No per-component profiling yet; cannot assign measured percentages.
Only scores (100k) and masks saved, not full100k latent arrays. CPU labels
exported once. Frozen PLAN/code hash guard applies; do not edit active code.

User additionally AUTHORIZED after coverage: profile deployment full100k top1
first attempt then feedback reranking top1024, and train/test updater variants
Q0/residual/non-residual/AdaJEPA. See FOLLOWUP_REQUEST.json. Scope resolved: user chose
ONLY better-coverage selector separately perpanel. Tie -> lower measured
initial ranking cost. Total4fresh updater trainings (2panels x residual/zero);
Q0/TTA no offline training. Q0 has NO token/weight updates, only fixed rerank
after externally selected first failure; explained explicitly to user. Need train pools with SAME new100k selector and forced first trial;
old first-action Q0-argmax assumption must be removed. Non-residual means zeros,
not an observation-only alternative. Profile avoids unnecessary full observation
encoding for zero branch. Q0 is static reranking after shared selector firsttrial.
Followup pipeline submitted; formal training NOT started. User explicitly said validation passes then run.
Runroot results/selector_feedback_pilot_20261016. PLAN now exists, both selectors=scene;
64train scenes/panel,256slots x2episodes,100eval tasks/panel,max9attempts.
A6000 scene-token encoding COMPLETE. Sunflower data array35818 FAILED exact
classifier-score check versus A6000:479/1024 different,maxabs5.722046e-6.
No method-level conclusions affected; original full100k comparison completed independently.
Oprime controller35325 stopped on failure; failed-dependent audit35819 canceled.
Failure archived FAILURE_A5000_ARCHIVED.json. No tolerances weakened, no frozen
code/PLAN edits. Execution-only recovery routes unchanged data collection to6A6000:
Oprime35335 collection+original audit; pilot35336 afterok35335. RECOVERY_JOBS.json
records wrappers/hashes; recovery_a6000.py/sbatch outside frozen code,6workers.
Same6GPU guard restricts tophysical0–5 and noexisting GPUprocesses. Original GPU
preflights remain required before4trainings. Check recovery_data_*.out,FAILED.json,
DATA_AUDIT.json,STATUS.json and SUMMARY.json. No repeat100k base scoring needed.
Latest status 2026-10-16 21:39 server time: Oprime35335 recovery COMPLETE,
all328pools (128train+200eval) and DATA_AUDIT passed. Pilot35336 FAILED during
preflight, no formal trainings started: controlled_backward strict mode requires
CUBLAS_WORKSPACE_CONFIG=:4096:8, missing in pilot environment. Failed-dependent
status may list active children but Slurm job terminated; do not report those as running.
Failure archived FAILURE_CUDA_ENV_ARCHIVED.json. Execution-only recovery35360
submitted on6A6000, currently PENDING(Priority). New wrappers recovery_cuda_env.py
and .sbatch preserve frozen code/PLAN. Before original pilot, compare env unset vs
:4096:8 exact predictor32 outputs/scores, single-image encoder and full-rollout
encoder for one train scene perpanel, plus exact equality to pool initial latent.
CUDA_ENV_CHECK.json gates original full preflight and trainings; inspect FAILED
and cuda_recovery_35360*.out. CUDA_RECOVERY_JOB.json records execution hashes.
Latest live check (2026-10-17):35360 still PENDING(Priority),noCUDA_ENV_CHECK,
noALL_PREFLIGHTS_PASSED,no model files for any of4runs. Existing STATUS.json
is from FAILED35336 and NOT current running work. Oprime physical GPUs0–3 now
busy~40GiB each;4/5free;6/7others~2.8GiB.35360 requests6GPUs/36CPU/180GB;
do not say formaltraining started or numericalfix verified. Scheduler estimated
startOct26 is provisional, not a reliable ETA. Actual queue includes otherusers.
No claims that the setting or later GPU checks passed before evidence appears.
Four CPU regressions passed locally/remotely, runtime imports and sbatch syntax
passed. GPU preflight awaits new pools; do not claim it already passed.
PLAN created after completed coverage chose both scene selectors. Source files
pipeline/selector_feedback_data.py, selector_feedback_run.py,
selector_feedback_launch.py; frozen copies and dependencies in run/code.
Frozen run/code/deploy_pool_cpu.py intentionally preserves float64 simulation
actions (model inputs float32) and DISABLES old5000candidate mode. Do not
replace it with the unchanged historical pipeline/deploy_pool_cpu.py.
New pool has simulation_action for original physics values, action for model.
Rerank ties use selector pool order; initial selector ties original action IDs.
No historical shortlist metrics reused; old arrays used only for engineering
references. Q0/TTA smoke outputs on first task are new protocol results, reused
by full eval after validation. Old cancelled experiments stay cancelled. Summary also includes selector_static
(original selector order), computed from existing labels/scores with no added
training; separates Q0 reranking effects from feedback update effects.

# Idle: historical shortlist experiments INVALIDATED and cancelled (2026-10-16)

User explicitly requested that ALL results obtained using fixed historical base
predictor + scorer candidate preselection be marked INVALID. This includes new
rerun deploy_pool_updater_20261016, which indeed used the same selector.
The assistant's prior rationale was reuse of comparable evaluation pools; this
was not the intended current-checkpoint deployment protocol. Do not reuse these
results as evidence of method effectiveness or for milestone claims.

Authority: results/INVALID_RESULTS_20261016.json; per-run VALIDITY.json and
legacy *.validity.json sidecars. Report banners override earlier conclusions.
Raw metrics, immutable plans/code and checkpoints preserved for audit, not
silently rewritten. Training records/checkpoint existence, engineering checks,
and coverage audits are not automatically false. Configuration recommendations
based on the invalidated ranking results are also withdrawn.

Cancelled Sunflower collection35696 (40 GPUs), training35703/35708/35713/35718,
all dependent audits/evaluation/summary and other registered jobs. Four formal
trainings had NOT started: preflights/cost pilot only. Cancellation snapshots
in deploy_pool_updater_20261016/CANCELLATION_SUNFLOWER.json.
Cancelled Oprime continuation35316 FIRST, then oracle/services35314, as oracle
uses the same rejected pools. CANCELLATION_OPRIME.json records this. Verified
no remaining queued jobs or matching oracle/encoder processes after cancel.
Do not resume these runs or continue their pending dependency graphs.

Coverage audit remains valid: official first5000 cache has >=1 success in
cross train1143/1200, within train1136/1200, cross eval90/100, within eval97/100.
Historical top1024 eval coverage75/100cross,95/100within. Lost15/2 tasks retain
legal successful actions before shortlist. Cache-vs-resim task coverage agrees.
See deploy_pool_updater_20261016/FIRST5000_SUCCESS_COVERAGE.json.

Next task is not launched. Previously discussed new-Q0 shortlisting is only a
proposal; do not silently substitute it or claim a new experiment is running.
Future protocol must explicitly identify candidate selector and checkpoint,
and train/eval pools; historical notes below are archival and superseded where
covered by the invalidation rule.

# Historical notes — affected results below are INVALID, not current conclusions

# Completed: initial-scene baseline (2026-10-16)

results/milestone_scene_baseline_20261015/SUMMARY.json: same100tasks/panel,
1024pools, S@1/3/5/9 cross6/7/7/8%,within26/39/48/63%. Q019%/54% at9.
Historicalseed0 matched-s16r16 classifiers,52276cross/52349withintrainpairs;
no retrain/adaptation. Raworders/labels/actions audited, manifesttrainingfiles
matchbasepack. Different1024dim action-free initialencoder vsWM PCA128,
not compute/objective/representation matched. Checkpoints state_dict only;
trainingprovenance from preserved manifests/results. Don'tmix old0.5604AUCCESS.

# Idle: milestone memory experiment complete (2026-10-15)

User-authorized six-GPU reduced run Slurm35185 completed successfully.
Verified 2026-10-15 11:55 UTC: no active/pending/failed tasks, all4 trainings
10000slots/20000episodes, all40 eval shards/1000 scene-method episodes complete.
SUMMARY.json complete=true,raw_audit_passed=true. Results/metadata synced locally.
Run results/milestone_memory_20261015; scope docs/MILESTONE_MEMORY_RUN_20261015.md.
Wall time5.66h including preflight,train,eval,audit. Per-model training3.52–4.01h.
Cross fold1 Q0lambda0 seed0; Within fold0 Q0lambda.03 seed0. Legacy terminal-state
L1+.1geometry updater, caps1/2/4/8, frozen reader,current residual,scorer supports.
Each protocol100tasks,shared1024pool,max9trials. S@9 percentages:
Cross:base17,Q019,real17,zero18,Q0-TTA18.
Within:base50,Q054,real48,zero52,Q0-TTA53.
No dynamic method beats fixed Q0 at9trials in this one-seed panel; this does not
establish general failure of memory. Detailed paired changes and first-update
ranking diagnostics in SUMMARY.json; raw trajectories remain on remote.
72.1%Cross/74.5%Within training episodes had zero supports; actual optimizersteps
4768Cross/4486Within of10000scheduledslots (same for real/zero per panel).
This is preserved old recipe, not a reason to silently change training budget.
No further runs started. Next if requested: analyze paired losses and update
ranking diagnostics; do not launch larger multi-fold design automatically.

## Completed post-hoc failure analysis (2026-10-15)

2026-10-16 user challenged whether small-pool statistics apply after online
selection. Reverified current milestone PLAN source-manifest SHA and all four
episodes.jsonl SHA against complete.json on oprime. All 80000 logged episodes'
selected actions are within their fixed manifest candidate sets and disjoint
from query IDs. Frozen code fast_updater.py:18 restricts policy_rollout to
row['candidates']; selection/reranking and residuals are current-model online,
but the candidate universe remains cached legacy small pools, not eval1024.
Thus pool statistics apply to this latest run. Explain this implementation
scope explicitly; do not equate online selection with deploy-matched pools or
treat distribution mismatch as proven cause of performance degradation.

CPU analysis only; no additional GPU runs. Scripts pipeline/analyze_milestone.py
and analyze_milestone_pools.py; DIAGNOSTICS.json and POOL_DIAGNOSTICS.json synced.
Stored full scores permit exact capped-update policy replay on same pools:
cap0 reproduces fixedQ0; cap8 reproduces fullpolicy; cap1 independently verified
by lexsort excluding first failed action then testing next8 candidate labels.
Cap1 S9 cross real19/zero19/TTA25 vs full17/18/18; within real54/zero53/TTA51
vs full48/52/53. Within real cap1 vsfull gains6/losses0. These are post-hoc
comparisons, not independent confirmatory runs. Full cap sweep in DIAGNOSTICS.
First-update real best-success rank:cross38better/30worse/6tie of74 definable;
within39better/21worse/15tie of75. Within real first-update mean AUC +.00367;
thus do not claim every update hurts or blanket frozen-head failure.
Training support candidates median25cross/24within (range5–49/4–49), mean
per-scene success fraction42.85%/43.26%; query42.69%/43.36%. Test pools1024,
mean success fraction1.331%/5.207%, median.293%/1.172%. File-index hashes verified.
Real training reachdepth4:384/20000cross,402/20000within; reach8:25/20000,48/20000.
Evaluation real reach4:90/100cross,58/100within; reach8:84/100,53/100.
Excluding pools with no success still reach8:59/75cross,48/95within.
Concrete mismatch in support pool and depth exposure is demonstrated; that it
causes degradation remains a hypothesis requiring controlled retraining.
Counterfactual replay demonstrates later updates cause net solve losses relative
to freezing after1 for real, strongestwithin; not monotonic harm every turn.
No deploy query-latent loss measured in this run, so don't assert improved L1,
probe distribution drift, or residual inherently useless. Samefold/seed limits.
Next suggested: cappedupdate comparisons, match training pool difficulty/depth,
then objective comparison; no new training automatically authorized by analysis.

# Idle: multi-turn speedup checks and measured cost report complete (2026-10-14)

User asked to fill missing validation then estimatecompute/time; prior estimates
inaccurate. Completed validation/benchmarks only; no formal multi-foldstudy started.
Report docs/MULTITURN_SPEEDUP_COST_20261014.md; raw/frozen code/logs under
results/multiturn_validation_20261014/. SUMMARY.json audit_complete=true, raw
histories independently rechecked by pipeline/summarize_multiturn_validation.py.
All jobs ended; squeue empty at completion. Atmost6GPUs;541telemetry samples,
zero sampled same-GPU competingprocesses (CPU/storage host notexclusive).

New runtime pipeline/multiturn_runtime.py: fullBPTT/allupdatedstate queryloss,
wholeforward activationcheckpointing; currentpredictor residuals recomputed.
PredictorAdapter now supportsbase/Q0 and clearsstale .grad at construction.
No old scientificcode/output overwritten. Original eval35156 guardfailure,
35157cancel retained; evalv235159/35160 pass. Train35154/35155 pass exact
12natural+one19updatestressslots,2eps/slot,real/zero,cross/within. Actions/losses/
gradienthash/updatedweightsmatch; predictor/probefrozen. Full1024pool6episodes
120trials,scoreauditsat0/1/19exact. Maxlr1e-3/3step/19update base+Q0 addon
35167/35168 exact and resetexact; old/newbarebaseadapter1/3stepscompatibility.
Stresspools allfailure, notsuccess-rate experiments. CUDAcontrolledsetting only.

Timing A6000: base/Q0 full8epoch projected4.10-4.63h each from256stepbench;
updater2.12-2.61s/slot =>5.89-7.26h/10kslots, BUT72-78%episodeszero-updates,
52-62%slotsnooptimizerstep. Deep19updates twoeps50-86s/slot; don't extrapolate
naturalmeanifpolicyfailure/depthchanges. 1024ranking ~10.2s default (~2-3%save)
or12.8-13.1scontrolled (~12-15%save atpairedpostupdateauditstates). Tokensupdate
~.004s;TTA1/3steps~.054/.160s. 20trial fullstress260-301s excludescoreaudit,
cachehit differences explainmuchofepisodevariation, notmethodspeedranking.

Dataaudit35166 + cachecheck35169 complete: newfourfold task-actionunion154051,
134757inexistingpacks+1412standalone,17882needencode(~6.74GPUh),32sharedlatent
samples exact and1412headerschecked,PCAhashmatches. No newfoldpackingperformed.
Newneutral1024pools600scenes initialencode~11.36GPUh ifnoreuse. Oldthinnewfold
packs~6400notfullbudget. Existing rawonlypartlycoversmissing trajectories.

Costprojection:8newinitial34.46GPUh+12updater79.06GPUh (early-success assumption).
36mainevals102.3-202.6GPUh formean10-20trials;600calib24.9-50.2;1200online
diag29.0-53.8;encoding18.1. Subtotal287.9-438.2 excludesnewheads,offlinequery
diag,queue.6GPUideal48-73h;reserve4-6days conditionally, NOTguaranteedETA.
Onlytraining113.5GPUh,6GPUideal18.9h,reserve24-36h onceassetsready.
Old24trainingsactually129.64GPUh,3.2-9.6h/run; priorfast-short-sampleextrapolation
missedlonggaps and knownsharedGPUvalidation. Don'tuniquelyattributeallgaps.
Next if formalstudyrequested: finalizefoldspecificdata/headrecipe+execution
manifest, implementtrain/evalorchestrationusingverifiedprimitives,checkdatareuse,
thenrun; reestimateafter1000slotsfromactualsupportdepth/stepcount. No pending
requiredwork or activejobs for thisvalidation+costrequest.

# Completed: multi-turn experiment design (2026-10-14)

User requested experiment design from selected initial configurations, learned
updater with/without residual, AdaJEPA, multiple official folds and direct test
evaluation. Design docs/MULTITURN_UPDATER_ADAJEPA_PLAN_20261014.md plus
results/multiturn_design_20261014/DESIGN.json. No jobs submitted this turn.
Chosen starts: Cross Q0 initial lambda0 seed0; Within Q0 initial lambda.03 seed0.
Three folds each (Cross1/0/2, Within0/1/2), initial/updater seed0 in first round.
Six arms: static base,static Q0,real updater,independently trained zero updater,
Q0 AdaJEPA-style,base AdaJEPA-style. 36 main evaluations/3600scene-method episodes.
Matched full8epoch per-fold initial models:12total,4existing reused,8new.
12updater trainings; total20new predictor/updater trainings excluding data/probes.
Updater frozen backbone/Q0, own lambda.03 both protocols;10k slots/2episodes,
8queries, supportcaps1/2/4/8/19,loss averaged over actual updatedstates,fullBPTT.
AdaJEPA predictor-only observed L1,recent5;lr/steps calibrated on25testtasks per
anchor panel and architecture,then settings fixed across otherfolds.600calepisodes.
Main test pools changed to model-independent validity-only1024candidates; retain
old base-shortlists separately, rerun static baselines; historical19/54 not main
newpool results. Newfold data/probe/scorer provenance must be checked, not reuse
anotherfold-trained predictor or head blindly. Two online diagnostic arms plus
shared-support offline analysis planned; no additional updater training needed.
Subsequent implementation/19update verification and measured costs are complete;
see latest idle state above and docs/MULTITURN_SPEEDUP_COST_20261014.md.
Formal multi-fold orchestration and fold-specific assets remain unlaunched.
Design saved/synced. No active execution or blockers for the design request.

# Completed: meeting report recorded (2026-10-14)

User requested preserving experiment results for next week's presentation.
Report docs/MATCHED_BASE_Q0_RESULTS_20261014.md explains background, matched
setup, metric definitions, all24runs, three-seed mean/sampleSD, mixed outcomes,
limits, proposed Cross multi-turn starting checkpoint, and presentation outline.
CSV results/matched_base_q0_20261012/REPORT_METRICS.csv retains full precision.
Updated plan doc completion/status links. Recommendation only: Cross Q0 lambda0
seed0 for S9-first multi-turn pilot, lambda.03 secondary for S20; updater loss
weight remains undecided. No new training authorized/submitted by report request.
Task idle, no blockers. Report and CSV synchronized to remote project.

# Completed: updater acceleration validation (2026-10-12)

User asked first runverification. COMPLETE; validation work idle; formal matched
base/Q0 study below also complete. No new formal updater experiment or production
backend switch. Doc docs/UPDATER_ACCELERATION_VALIDATION.md, updated implementation
doc docs/INFERENCE_UPDATER_SPEEDUP.md. Results results/updater_validation_20261012/.

Extended comparisons:65slots=64normal+1stress(twoepisodeswith8failedsupports),
130episodes perhistory,3INDEPENDENT histories original/optimized/originalrepeat.
Same initialcheckpoint/RNG/draws/AdamW/fullBPTT; no restoring intermediateweights.
StrictCUDA cross+within and strictCPU cross ALL PASS EXACT events/actions/supports/
loss/gradienthash/updatedweighthash, frozenreader+geometryprobeexact. Auditedscorer
scores all exact (strictcross5368,strictwithin5117). All8updates executedinstress.
Summary independently recomputed fromrawhistories; SUMMARY.json audit_complete=true.

DefaultCUDA cross NEGATIVE CONTROL fails: ORIGINAL-v-ORIGINAL also splits!
Both comparisons gradient/weight firstdiffslot2; originalrepeat firstpolicydivergence
slot15,optimizedslot21 (zero-based). All5392fastscores still exact onSAMEprediction.
Supports numericalkernel nondeterminism as cause ofpreviousdivergence, not scorer
or stale-residual semanticdifference. Not unique attribution tooneCUDAkernel, not
claim optimizedmorestable, not overallmemorymethod-success claim.

Numerical configuration: CUBLAS_WORKSPACE_CONFIG=:4096:8, torchstrictalgorithms,
validation TorchDispatchMode intercepts onlyadaptivepoolbackward. CUDA mode uses
originalpoolkernel withscopedguardexception; CPUmode usesCPUderivative. Forward
unchanged. Actualpool14x14->10x10 overlaps. Initialdisjoint-onlypilotguardfailed,
corrected; logs/code retained. Unit checks14/16/18/20 gridsPASS unchangedforward,
CPUgradientexact/repeatable andCUDAgradientwithinFP32tolerance; flagsrestored.
StrictCUDA empiricalexactness on testedhistories, not universal guarantee of the
poolingexception. Defaultmode has no exacthistoryguarantee, even original-to-original.
This supersedes earlierposthoc relaxed-tolerance argument; exactcriteria neverloosened
inthisfollowup. Do NOT overwrite any historicalfailed outputs.

Code: pipeline/{updater_validation_numerics,validate_updater_trajectory,
validate_updater_extended,test_updater_validation_numerics,summarize_updater_validation}.py.
No changes to pipeline/{fast_inference,fast_updater}.py inthisturn. Earlier6CPU+6CUDA
inference tests andactualbase/Q0/memoryscorechecks passed; sources unchanged.
Newlongchecks allfinished (negativecontrol expectedexit1 withpersistedresult).
FourownGPU allocations only, sharedtimingNOT isolatedspeedbenchmark. No speedfactor
claim. Verification complete; no remaining required verification work.
New formal updater training not authorized.

# Completed: matched base/Q0 from-scratch weight study (verified 2026-10-14)

Latest user priority: retrain base and Q0 with IDENTICAL loss; investigate geometry
weight lambda. User answered both async questions explicitly: lambda is geometry
coefficient in latentL1+lambda*geometry; BOTH predictors start RANDOM, not pretrained.
Execution idle: all 24 trainings and evaluations complete; acceleration validation complete.

Registered/implemented/submitted run results/matched_base_q0_20261012.
Maintained plan docs/MATCHED_BASE_Q0_WEIGHT_STUDY.md; source pipeline/matched_base_q0.py.
Formal jobs use FROZEN results/.../code/{matched_base_q0.py,run_one.py}; do not mutate.
PLAN.json scratch=true via initialization='scratch', sourceSHA72f6b0ad29928ee45956426daca61875b1eb3e2ecfa6c0d266d82577899335b6.
Frozen runnerSHA7901742ad990e22b915840fead54cc7af1b5e029ff3f49f499005efb3f318f03.

Scope: official crossf1 and withinf0, arms base(no tokens)/Q0(32prefixslots original
self-attention, no support/updater). Lambdas0/.03/.1/.3; .1 seeds0,1,2;othersseed0.
24trainings+24static100taskevals. Same seed same RANDOM initialbackbone, same
example order/geometrytimes/AdamW/OneCycle/updates. Checkpoint reads originalargs
for dimensions only; model does NOT load pretrainedpredictor state_dict when scratch.
Original initialization reproduced: truncnormal time/space .02, zero outputprojection.
Q0 initialtokens sampledseed+7919; backbone commoninit separately reseeded. Zero
output means firstscratchstep token/backbonegradzero expected; tokenactivity checked
atstep1 afterprojectionlearns. Encoder/PCA/probes/scorers remain pretrained/frozen.

Full8epochs B8 lr3e-4 weightdecay.05 clip1 OneCycle3%warmup. Cross52276trajectories,
52272updates/418176exposures;within52349/52344/418752. Lambda loss implementation
shared for BOTH arms: futureframes1..31 latentL1 + lambda*geometry_loss imported
from pinned memory_geometry_followup.py; originalQ0 differentiable XY+bar-angle,
equalquery thenobject/time weights,8deterministicfuturetimes/query. Not oldXYonlyaux.
Probe gradient throughprediction preserved, no fast no_grad scorer inloss.
Q0backbone+initialtokens trained, unusedupdaterfrozen. Finalfixedcheckpoint only.
Newscorer acceleration NOT enabled. Q0 checkpoint_blocks=False after exact verification.
All old models/results preserved. With scratch, budget is originalbase-sized, not
an extra8epochs atoppretraining. Historical originalbase has ownseed/objective.

Smoke35024cross/35025within COMPLETE PASS. Verified same initialbackboneSHA,
zerooutprojection and base/Q0initialL1/geometry identical; newQ0loss matches oldexact;
8realupdates Q0checkpointTrue/False losses+weights exactlyequal; probeunchanged.
Short timings not robustspeedclaim:cross Q0 4.70->4.03s,within2.49->2.23s/8steps;
peaknocheckpoint~13.6GiB vscheckpoint~8.6GiB. No performance results yet.

jobs.json authoritative; original four dependency lanes (one job/run), now max6GPUs user-authorized:
crossbase35026..35031, crossQ035032..35037,
withinbase35038..35043, withinQ035044..35049.
Eachlane order .1s0,.1s1,.1s2,0s0,.03s0,.3s0. Firstjobs dependbothpassedsmokes;
Originally each nextjob afterokprevious (includes evaluation); scheduling amendment below supersedes Q0 lambda.3 and summary dependencies.
Latest verified 2026-10-14 16:44 UTC: ALL24 trainings and ALL24 evaluations COMPLETE.
Each evaluation has100tasks and raw_audit_passed=true. SUMMARY.json complete=true,
paired_training_verified=true. Last evaluation file timestamp2026-10-14 07:55UTC.
No own jobs remain in squeue; summary35050 completed and produced SUMMARY.json.
Results JSONs synchronized locally; model checkpoints/raw score arrays remain remote.
2026-10-14 user authorized six concurrent GPUs: released Q0 lambda.3 dependencies
35037/35049; summary35050 explicitly depended on all six unfinished jobs. Details in
schedule_expansion_20261014.json. Frozen training/evaluation code unchanged.
No active execution or blockers. Results ready for weight/base-vs-Q0 analysis.
No newmulti-fold runs authorized/submitted; otherfold train packs absent at earlier
check, onlycrossf1/withinf0 available. Expansion requires separate data preparation.
No newmulti-fold launch. Paired formal initializationSHA/devlossverifiedexact.
EachsingleGPU4CPU48GB24h; summary2CPU8GB. Data onsharedoprime, result artifacts /mnt/bum.
Logs logs/mbq_{panel}_{arch}_g{weight}_s{seed}_{job}.out.
Each job resumes via models/{panel}/{name}/resume.pt every1000steps+epoch;
atomiccheckpoint, no droppedtrainingdraws; initial/dev_epoch/progress/completeJSON.
No monitoring remains. Paired training verification passed in final summary.
User has not authorized updater training for this new study.

Evaluation uses unchanged corrected assets_fixed pools100tasks/panel1024actions,
actualsimlabels, B32originalscorer (crosspredicted_mix_s0/withinoriginal). Fixedrank
S@1/3/5/9/20 + AUCwithdefinedscene count; principalS9. Noautoregressivetrialsneeded.
Also fixed400devtrajectories/epoch for L1/geometry, sharedqueries acrossmodels;
these are NOT rankingpools, no matchedquerymechanism claim. logsweightedgeometry/L1.
Lambdaotherthan.1 one seed exploratory, cannot claim universalbestlambda.

Prior official smallmemory/TTA run COMPLETE before thisstudy; summaries both
complete/raw_audit_passed true,2000firstsuccesscases independentlyrecounted.
Results docs/QUERY_TOKEN_MEMORY_PROPOSAL_REVIEW.md topfinalsection.
MeanS9/S20 cross original12/16,Q017.6667/24,real17/24.6667,zero17/25,TTA18/25;
within original49/64,Q049.6667/62.3333,real47.3333/62.3333,zero48.3333/64,TTA47/61.
Originalbase usedlatentL1only(aux_probeempty,aux_lambda0,motion_beta0). Previous
Q0 pretrained+extra8epochs+geometry bundled effects; user now explicitlychose scratch.
Previousmemory SUMMARY_FIXED.json SHA7885f50cc68a506dbb9ef33cbf04a954d564eeec43291b589674958a45de2601;
TTA SUMMARY.json SHA f6c099c340298edfc0be9e5ac862563f968c694414955d9f66881145f2571835.

Deferred speedwork: pipeline/performance_20261012/{batched_readout.py,benchmark.py}.
Inference35020/35021 COMPLETE;5templates/panel,real_s0selectedactions matched all
100cross+62within trials. Some lower ranks differ; fastB32~1.37-1.59x fullranking
on earlycases, largerB64/128littlefurthergain. No universalnumeric-equivalenceclaim.
Training35022/35023FAILED updaterexactevents/grads/weights check; crossslot15support
count2->1. Notproductionenabled. Need laterdiagnose firstpaireddivergence/saveevents;
oldfailedscript asserted beforepersistingdiagnostics. Q0checkpoint removal8steps
passed there and reverified thisstudy with actualOneCycle+s scratchinit.
Earlier prototype jobs are done; new exact implementation verification is above.

Frozen evaluationnumerics: small_official_20261010/recovery_20261010/recovery_runner.py
alwaysinitialencoderB1 (oldB32large mismatch), predictorB32. Rebuilt200pools/labels.
Tinycross00008:197 action3393projectionshape diff verifiedencoderhiddenexact and
matchingPCAshapeexact; only verifiedunchangedtruthcachewritten, globalguarduntouched.
Retained status_20261011/OBSERVATION_VERIFIED.json. Newbatched kernels must verify
scores/ranking/selectedactions before reuse in memorysupportselection.

# Completed: corrected full-budget staged prefix experiment (2026-10-09)

Status: idle. User-authorized 9 trainings and18 evaluations ALL COMPLETE;
summary34721 complete=true/raw_audit_passed=true. No running/pending jobs.
Run results/prefix_staged_20261008, remote oprime:/path/to/phyre-unc.
Report docs/QUERY_TOKEN_MEMORY_PROPOSAL_REVIEW.md latestsection containsfinalresults.
Independent final_analysis/audit.py reconstructed ALL1800 rawstatic/dynamiccases:
1024candidateIDs/labels,stable ranking,untriedargmax,successstop,firstscores same
Q0 acrossbranches,original scores identicalacrossseeds,allcounts PASS. Local
summaryfirstsuccess maps independently recounted;summary/audit SHA verified.
LATEST_STATUS.json supersedes incomplete snapshots. No additional traininglaunched.

PrimaryS@9 means cross original47,Q048,residual50.667,zero49.333;
within original54,Q052,residual57.667,zero55.333. PerseedQ0/residual/zero:
crosss0 50/53/50;s1 46/46/48;s2 48/53/50.
withins0 49/56/51;s1 54/60/58;s2 53/57/57.
Residual-Q0 S9 cross+3/0/+5pp,within+7/+6/+4pp;
residual-zero cross+3/-2/+3pp,within+5/+2/0pp. ThusmeanS9improvesvsQ0
bothsplits;withinallseeds, cross2improve1tie. Residual-specificbenefit modest
andnotallseedspositive. No statistical/universal claim on reuseddevpanels.
S20means cross original51/Q055/residual56/zero56.333;
within67/68/73/73.333. Realresidual DOES NOT beat zeromeanS20either split.
Zero retains action/history/updatecount andfailurefeedback,not noinformation.

Architecture:32prefixwidth384 inoriginalaxialselfattention,no extra readadapter.
Stage1 originalpretrainedbackbone+freshactiveQ0 bothtrained,52276trajectories,
1600scenes/16templates,8epochs,batch8,52272updates418176presentations perseed.
L1+0.1geometry;extra fullbudget abovepretraining,not matchedtotalbudget.
Stage2 fixedbackbone/Q0, freshpairedresidual/zero updaters.1200scenes,
20000episodes/10000slots each;2scenes sequentialbackwards equalsceneweight,
uniformprefixcap1/2/4/8, actual scorer-selected mixedsupport-half candidates,
stop atsuccess,no repeats,no Near/no failure-prefilter. Truth-currentprediction
residual fullBPTT replay,not cached; no derivative throughargmax. Querygroups
separate. Initial-success episodeskeepdiagnosticQ0loss,no updatergradient;
ifbothscenesnoprefix skipoptimizer,scheduler stilladvances byslot.
Actualoptimizersteps4749/4758/4734 perseed identicallyinbothbranches;
zero-supportepisodes14482/14460/14519 of20K. CheckpointSHA/freeze/pairedinit
andparent linkage verified. Trainpool29868actions includes12952successes,
smallerthan1024 deploymentpool; do notclaim matchedfull deploydistribution.
Evaluation same100cross+100within,3seeds,fixedpredicted_mix_s0scorer,
fixedfinalcheckpoint;primaryS9,secondary1/3/5/20.20exceeds8trainedupdates.
Reuseddevpanels;within generatedsame-template,notofficialtest.

Original-vs-stage1 analysis COMPLETE:stage1_comparison/RANKING.json audits600
staticcases. GPUmatched400dev job34722complete:originalL1.119949206,
Q0mean.120932038(+0.82%);geometryloss1.021492375->.917749860(-10.16%).
Geometry normalizedtrainingloss,notpixelMAE;400dev differsfullrankingpools.
Randomprefix dev_initial~.1355 NOT originalbasebaseline. Stage1bundlesprefix,
extrafullbudget,geometry;not causalprefix-onlyablation. WithinAUCimproves
.847784->.864430(99defined)despiteS9drop;cross.724759->.731904(84defined).

Formal checkpoints/resume remainremote; summary/per-sceneresultJSON/auditslocal.
Next action: none required; report complete results. No active work or blockers.

# Completed: full ABCDE evaluation extension (2026-10-08)

All18 A/B/C memory model/split runs and12 D/E static runs complete.
Summary34672 complete=true,raw_audit_passed=true. Run:
results/query_token_dynamic_full_eval_20261007/SUMMARY.json downloaded.
Local independently recounted all30 model/split first-success maps (100each),
all S@1/3/5/9/20 counts agree. This is summary-level independent recount;
raw trace audit was performed by remote summarizer, not rerun locally this turn.
CrossS9 perseed A46/43/49 B49/38/47 C38/53/52 D47/47/48 E40/46/51;
means46.0/44.667/47.667/47.333/45.667 versus originalbase47.
WithinS9 A59/53/58 B62/61/65 C58/60/63 D65/60/57 E58/56/58;
means56.667/62.667/60.333/60.667/57.333 versus originalbase54.
B-D within+2pp,cross-2.667pp; C-Ewithin+3pp,cross+2pp; these descriptive
3seed means do not establish reliable memory effects or significance.
Oldmodels usecrossattentionreaders/randomfailedsupport,10Kupdates;
not evidence against corrected prefix/scorer-selected training and not a
converged/fullbudget scratch comparison. PrimaryS@9,not officialAUCCESS.
Both100scene panels are reuseddevelopment data;within generatedsame-template.
Report docs/QUERY_TOKEN_MEMORY_PROPOSAL_REVIEW.md updated with fullresults.

# Current specification correction (2026-10-07)

User clarified intended architecture: memory tokens are PREFIX CONTEXT in the
predictor self-attention sequence, not separate cross-attention read adapters.
Training support must be chosen by current predictor/tokens + scorer ranking,
updated after each observation, not uniformly random failed-support replay.
These supersede assistant-devised deferred-policy plan. Old implementations
DO NOT test this intended method. Keep all artifacts; label conclusions as
cross-attention-reader + random-failure-training variant only. No corrected
prefix/on-policy implementation or training submitted this turn.
Future implementation must preserve causal selection (do not prefilter candidates
to failures), current-state residuals, reset-per-scene and support/query separation.
Top report now explicitly records mismatch, responsibility and completedABCDE.
User questioned base cross/within60%. Independent raw audit PASSED:
dynamic/audit_baseline_panels.py -> BASELINE_PANEL_AUDIT.json. Both20scene
subsets solve12/20, different tasks/templates, selected by registeredtaskhash.
All15arms/seeds x20tasks per split baseline ids/labels/scores EXACT against
staged100scene originalbase outputs (600raw comparisons total). Summary matches.
Same predictor/scorer hashes, distinct packmeta. Crossremaining35/80 =>47/100;
withinremaining42/80 =>54/100. S20oldcross12/20,within14/20. No mixup found;
smallpanel representativeness limited, cannot generalize ABCDE20scene rankings.


# Completed: cross predictor/probe/update mechanism investigation (2026-10-07)

User-authorized follow-up completed using saved raw outputs, no GPU/new training.
Staged/mechanism_followup/investigate.py -> RESULTS.json; controls.py -> CONTROLS.json.
Exact crosspack/meta/head SHA and 100scene indices/labels/truth plus original
base raw geometry/scores matched old probe_adaptation_cross_20261001;600 paired
seed/scene/head Q0/update score audits passed. Report latest top section updated.
True geometry S@9=83; observed latent via original or adapted probe S@9=80;
Q0=46%,closedreal=45%. Both observed probes solve all positive-pool scenes in
00007(14)/00014(20) atS1. Probe reads true latents well; predicted-latent interface
is bottleneck. Does NOT separate predictor missing info from probe mismatch on
predicted latents, and oracle future inputs are not deployable.
Fixed original-policy K1 within00014:19positive scenes x3seeds=57 pairs.
All57 support needs farther geometric correction while successquery needs closer;
56/57 successquery actually moved farther,56/57 successgeomMAE worsened.
Needed mean distancechange-15.56px,actual+3.10; true-score-weighted times
needed-16.92,actual+3.88 (existing scorer soft weights, no new Near threshold).
Mean correction cosine totruth failure+.286,success-.384;55/57 successnegative.
Compatible with overgeneralizing failed support correction, not proof model
ignores action. Q0locked success-vs-failure pair margin worsens37/57;33of37
include successscorefall. Successgeom Q016.809,zero17.227,real19.854px.
Training query pool verified12702/29349positive=43.28%; expected equal-scene
weight42.69%,not measured actual draws. Cannot blame overall absence ofsuccesses.
Next if extending: same-query support controls, then separate loss-only experiment;
probe adaptation on current predicted latents needed to isolate decoder mismatch.
No follow-up training submitted; older jobs remain separate.

# Completed: staged Q0 predictor then updater (verified 2026-10-07)

User clarified replan: first train pretrained predictor+active learnable Q0,
then support/query train updater. Revised plan in
 docs/QUERY_TOKEN_MEMORY_PROPOSAL_REVIEW.md (top section).
User now prioritizes execution. Isolated implementation andregistration complete:
results/query_token_staged_20261005, frozen code11files+SHA256SUMS,
pipeline/query_token_staged.py andtest_query_token_staged.py.
CPU float64 tests PASS:batchprediction5.55e-16,gradient6.07e-18; activeQ0,
identityupdater,frozenreaderinputgrad,dynamicresidualBPTT,Q0invarianceafterstep.
Dataregistration PASS:1600stage1/1200stage2,epochbudget,train/crosstemplatedisjoint,
parentdatahashes,100withinprepared; CPU_AUDIT/DATA_AUDIT stored.
GPU smoke34538 COMPLETE/PASS:100updatesQ0+100real+100zero,rawpaneland9attempt
causalreplay forbothbranches. Trainlooptimes40.65/69.66/43.30seconds;
peak7.29/14.33/9.16GiB. Notperformanceevidence;formalmodelsstartfresh.
2026-10-06 checked: ALL9formaltrainingtasks COMPLETE; TRAINING_AUDIT34543 PASS.
Q0all3seeds52272updates(8epochs),real/zeroall6models10000updates.
Independent checkpointSHA andcalibrationrawAUC audit PASS:
run/INTERIM_TRAINING_CALIBRATION_AUDIT.json. Calibration32scenes (seeninstage1,
heldoutupdater),fixed1failedsupport;Q0meanAUC.813637,real.818091,zero.812641.
All3realseedAUC>ownQ0/zero. Meanreal-Q0+.004454,conditional scene95CI
[-.000185,.008837];real-zero+.005449,[.000988,.009707]. Notclosedloop/cross proof.
Q0devL1initialbase.119949 -> finals.122556/.120937/.120963 (mean.121485),
so no latentL1win fromstage1; geometryobjective differs fromhistoricalbase.
Q0eval34544all6tasks FAILED atfirstoutput:static_case omittedoutputdirectorymkdir.
No numericalmodel failure. Maintainedpipeline fixedmkdir +regressiontestPASS;
registeredfrozen code/manifest/checkpoints UNCHANGED. Recoverywrapperprecreates
100staticfolders perjob;sourcefix+wrapper+receiptinrun/recovery_20261006/.
2026-10-07T15:11UTC verified: all staged evaluations COMPLETE, including
secondary diagnostics. SUMMARY.json and WITHIN_SUMMARY.json both complete=true,
raw_audit_passed=true; downloaded and independently recounted per-task totals.
All 100 scenes per split, S@9; seed0/1/2:
Cross original47; Q0 42/49/47(mean46%), real41/47/47(45%), zero43/48/47(46%).
Within original54; Q0 57/53/55(55%), real60/58/56(58%), zero54/56/57(55.667%).
Within real-ownQ0 +3/+5/+1pp; real-zero +6/+2/-1pp. Cross real-ownQ0 -1/-2/0pp.
First-action success unchanged between each Q0 and its real/zero branches.
Cross independent raw-score audit preserved in CROSS_PRIMARY_AUDIT.json:
all greedy choices/labels/stops, baseline arrays and replay flags pass.
Cross real-Q0 and real-zero mean -1pp, conditional paired-scene95CI[-2.667,+.667]pp.
Within positive on this reused development panel; no new-test or universal
residual-benefit claim. Within generated same-template panel is not official fold.
Release34548 completed at2026-10-07T08:41:32Z; OLD_JOBS_RELEASE.json confirms
all six old holds released with returncode0. No staged jobs remain pending.
Old v1 within34483_0–4 running at15:11UTC, rest pending array cap; historical
status in older experiment sections below predates release unless otherwise noted.
No training rerun. Recovery34625 succeeded without numerical/code-manifest changes.
Newtrainarrays:Q0%3,real%2,zero%2 withaftercorrperseed =>max5newtrainingGPUs.
Evaluationfollowsfulltrainingaudit;Q0sixsplit/seedtasks%5 thenupdater12%5,
within/cross noall-cross-before-withinbarrier. Newrunjobs.json authoritative.
Stage1 trains originalpretrained backbone+Q0+read adapters, no support,3seeds.
User furtherrequires stage1 samefulltrainingbudget asoriginalbase, NOT10Kpilot:
verified historicalsummary andpackmeta:52276trajectories,1600scenes,16templates,
8epochs,effectivebatch8,6534updates/epoch=52272updates,418176presentations/seed.
Useoriginalfulltrainpack andper-epochshuffle/drop-tail;LRscheduler full52272.
Stage2 remains1200scene support/query;memorycal400seeninstage1, so cannotcall
stage1heldoutcalibration. Originalbasedev curves diagnostics,finalepoch8fixed.
Currentplan retains pretrainedinitialization+geometryobjective, so stage1 is
additionalbase-sizedbudget, not totalpretraining-budget oridenticalrecipe match.
Historicalbase12618sec (~3.5h) notnewruntimepromise; newsmoke timingrequired.
Stage2 freezes backbone,Q0,read adapters,encoder/probes; train updater only.
Two matchedstage2 branches:real dynamic residual vs train/evalzero residual,
3seedseach sharingstage1checkpoint/init/draws. Total9trainingjobs(3+6).
Initialupdater identity; Q0/firstaction predictions invariantthroughstage2.
No L0regularizer stage2:gradient to updater iszero with entireQ0path frozen.
Need gradients through frozenpredictor w.r.t.tokens; no fullforward no_grad.
Samecurrentpredictor residual, randomfailure support firststageofdiagnosis;
no Nearfilter,no policydata/loss/architecture capacity changes combined.
PrimaryS@9all100cross/100within; Q0static baseline freefrominitialscores;
separate realvszero updater forresidualincrement. Devpanel reuse disclosed.
Old same-time5arm/12-newruns draft superseded byuserstagedclarification.
Staged evaluation and requested within-failure analysis complete (2026-10-07).
CPU raw-score audit all300 paired seed/scenes PASS; no new GPU jobs.
run/analyze_within.py -> WITHIN_FAILURE_ANALYSIS.json; report top section updated.
Residual126failed seed/scenes =117 positive-pool Q0+updaterfail +6updateharms
+3no-positive(same1scene x3). Q0 positive-pool failures132,15 rescued (11.4%).
Q0 initial positive rank10–20:14/47rescued; >20:1/85. Main bottleneck is
insufficient top-rank correction, not wholesale updater damage or pool coverage.
Fixed1support77scenes: L1 .084939->.084857, allsurfaceMAE12.816->12.400,
successsurfaceMAE5.842->6.068; AUC76scenes .86033->.86458. Descriptive fixed
support diagnosis, not actual closed-loop latent observations or causal proof.
Template00019 original21/30->Q013/30->real12/30: most netregression stage1.
Loss averages latent and geometry, no direct ranking; randomfailedsupport is
not policy-matched. Objective/support mismatch plausible, NOT established root
mechanism or data-volume insufficiency. Next: matched top-score margin diagnostic
and separate loss/support controls if requested; older jobs separate and active.


Cross failure analysis complete (2026-10-07), saved run/analyze_cross.py and
CROSS_FAILURE_ANALYSIS.json; cross_diagnostics.py -> CROSS_MECHANISM_DIAGNOSTICS.json.
All300 closed seed/scenes raw audit PASS. Fail165 =112 positive-pool Q0+real
bothfail +48no-positive(16scenes x3) +5 updateharms; rescues2/114 eligible.
All rescues initially rank10; rank>20 rescue0/95. Available-but-failed median
initialrank96 vswithin28. Template00007 base0/Q02/real2 successes out75;
00014 base30/Q025/real22/zero25 explains all net updater-3.
Matched actual first failure185seed/scenes: Q0AUC.683888->real.680345,
zero.682322. 00014 bestsuccess-vs-bestfailure margin worsens42/51; not only
long-recurrence drift. Fixed original-policy support: AUC.694780->.689041,
scene95CI[-.013018,+.000922] includes0; L1.119118->.119198 (no overall L1 gain),
allgeometry56.608->56.467 but successfulgeometry32.266->33.808. Both probe
heads show degradation. Template14 successgeometry16.809->19.854(+18%).
Offline TRUE geometry +same scorer on full candidatepool: crossS1=81,S9=83,
maxcandidatecoverage84; withinS9=96/99available. Not deployable; bypasses BOTH
predictor andprobe, cannot separate their errors or prove geometry-loss benefit.
Main localized bottleneck is missing/incorrect success-relevant geometric
correction on unseen templates; objective/support/probe causes not causally
separated. No new GPU training/eval. Report top cross section updated.

# Completed: dynamic-residual five-arm query-token experiment (verified 2026-10-07)

Run results/query_token_dynamic_20261005. All15 models trained20000episodes,
10000updates; all15cross and15within eval COMPLETE. SUMMARY.json and
WITHIN_SUMMARY.json complete=true,audit_passed=true,15models each; raw audit
9780cross/9864within arrays by registered summary. No Slurm jobs in user queue
at latestcheck; Slurm accounting disabled, completionverifiedbyrun artifacts.
Downloaded both final summaries. Closed-loop panel20scenes per split, not100;
fixed-query panel100 per split. Three-seed meanS@9:
base cross60/within60; A56.667/60; B50/68.333; C51.667/68.333;
D53.333/56.667; E45/60 (percent). Not comparable to staged100scene rates.
Allmemoryarms use cross-attention reader and randomfailedsupport training,
not newly clarified prefix+training-time scorer selection. Scratch C/E have
short additional10K budget, not matching originalbase52K budget/convergence.
Earlier INTERIM_* analyses are historical subsets; final summaries supersede
old claims of pending cross/within evaluations. No new experiments launched.

A=frozen pretrained+memory; B=trainable pretrained+memory; C=random predictor+memory;
D=trainable pretrained/plain; E=random predictor/plain. Paired initialization B/D,
C/E. Frozen comparison reference is evaluation ONLY, not residual generation.
Encoder and geometry probe remain fixed in all arms. Full predictor checkpoints,
reference excluded from optimizer/checkpoint. Residuals are recomputed from current
weights/state every training replay; NO persistent residual cache or detach.
Full BPTT includes support predictions -> residuals -> state updates -> query loss.
At deployment, save only the best untried candidate's prediction from batch32 score
forward; after observing its failure, truth minus THAT prediction updates state.
Independent prefix replay recomputes chosen prediction with identical candidate
batch shape, checks every state; restored finalstate reproduces finalscore array.

20000episodes, scene batch2 =>10000optimizer updates, <=8queries/scene. Same
scene/query/times/support draws in allarms per seed, all1200scenes verified.
Uniform sampled prefix from random failures K1/2/4/8, equal scene weight,
L1+.1geometry,AdamW3e-4,OneCycle3%warmup,clip1,noNearfilter. Fixed20Kepisodes is not
convergence proof for random initialization. Budget audit requested by user: historical cross base summary verified52276unique
train trajectories,8epochs,batch8 =>52272optimizer updates/418176trajectory
presentations,12618seconds includingvalidation (~3.5h). Current eachseed10000updates,
159660–159687directquery presentations (~38.2%);querytargetpool29349,plus46617–47100
supportobservations(memory arms). Same predictor dimensions384x6layers/6heads.
Five-arm comparison matches ADDITIONAL training only; B/D inherit historicalbase
training whereas C/E do not. Shorter runtime is mostly smallerbudget, not proof
scratch converges quickly. C/E are fixed-budget pilots; no scratch-vs-pretrained
capability verdict without longer, calibration-checked convergence study.
Evidence run/TRAINING_BUDGET_COMPARISON.json. Existingjobs unchanged.
Before a faithful old-base replication, also check initialization recipe: current
common.Predictor initializes time/space tozero and outputlinear normally; available
train_predictor.py initializes time/space random and outputlinear zero. C/E matched
to eachother, but not a verbatim historical training recipe.
Random replay training is off-policy;
periodic policy collection/replay is not in this first fixed-budget stage.
CPU realenv audit PASS: batchvsserial prediction8.88e-16,gradient2.22e-16,
depths1/2/4/8,including gradient through dynamically predicted residual.
GPU smoke200episodes/arm PASS, exactpaired draws andinitialization confirmed.
Loop times A/B/C/D/E=55.23/62.67/63.78/43.39/43.90seconds; peak17.02/24.66/
24.66/12.00/12.00GiB. Extrapolated~22.4GPUh rawtraining for15runs; plan6–8elapsed
hours with5availableGPUs inclroutineoverhead, excludesqueue/full eval/within prep.
GPU smoke includes trainability/gradients, token-sensitive residual,
reference independence, state/model roundtrip, exacthistoricalbase scores,
fixedquerymetrics and3attemptclosed-loop causal state/score replay.

Formal eval aftertrain:100cross/100within fixedquery;20tasks/split closedloop20
attempts(full/reset-update), shared fixedbasepolicy failedprefix1/2/4/8 diagnostics
(full/reset-update/zero/static,64queries). Main S@9; S@20 extrapolation. The legacy
variant name 'latest' means reset updater input Q0; with dynamic residual it can
retain history indirectly, NOT strict latest-observation-only. Report limitation.
Calibration-only curves, finalcheckpoint selection. Within data waits original34413.

Superseded fixed-reference staging results/query_token_ablation_20261005:
cache34463 and5GPUsmokes34464 +summary34465 COMPLETE/PASS. NO formal training.
Keep as engineering record only; not approveddynamic-method performance evidence.
Its shared19GBresidual cache is NOT used by dynamic run. Smoke training-loop times
for200episodes A/B/C/D/E=41.8/48.0/48.9/42.4/42.0seconds, excludesinit/eval;
do not extrapolate these to dynamic residual memory without fresh measurement.
Historical proposal review docs/QUERY_TOKEN_MEMORY_PROPOSAL_REVIEW.md and original
frozenquery-tokenprototype remain background. No19support prerequisite/Nearfilter.

# Active: fresh memory L1+geometry training (2026-10-05 UTC)

User explicitly authorized retrainingmemory usingverifiedgeometrysupervision.
No R0 continuation, no Near-basedtrainfilter. Fullyimplementedandsubmitted.
Registration docs/MEMORY_FRESH_GEOMETRY_RUN.md; run results/memory_fresh_geometry_20261005/.
NewMemoryPredictor constructor ONLY, frozenbase/originalprobe; mode residual,
mem_steps16,mem_grid10,per_attemptTrue,gate_emptyTrue. Allo-projectionszero,
initialnonemptyoutputs EXACTbase. No oldmemorystate loaded.
Trainingall1200scenes/12templates(each100),29349query/12702success,
16916failedsupports; existinggroupedisolation,task/actiondedup preserved.
Calibration400scenes/4memoryheldouttemplates,basehasseen. NoN/H/Fselection.
Uniformscene;K1uniformfailedsupport;<=8uniformlabelblindquerysamescene;
separatesupportRNG. L1+.1originalprobeXY/bar-angle,8fixedfuturetimes/query.
Seeds0/1/2 each20000steps,AdamWmaxlr3e-4,OneCycle3%warmup,wd.05,clip1.
Fixedfinalcheckpoint;save0/500/2000/5000/10000/20000 plus250stepresume.
Curves24train_seen+8calibration+8cross,<=64hashfixedqueries/twoheads.
Final24train_seen+32calibration+100cross,fullqueries/fourheads. Within100
naturalpanel(allscenes regardlessN/H/F) waits originaldataencode/baseprepare.
Do notcompare64-querycurves directlytofull1024macro;D0reused,noclosedloopclaim.
PreviousrecommendationR0controls/458Nearfactorial withdrawn peruserdirection.
Freshbaselinefirst; additionalpairingsnotlaunched orsmuggledintothisexperiment.

Jobs34453smoke PASS:8updates,nonemptyinitialidentity,emptyidentityafterupdates,
geometrygradandencodergradafterfirstupdate,frozenbase/probe,rawmetricschecked.
Tiny4querysmokescenes singleclass;34454 independentfull1024mixedlabel4head
replay/audit PASS,including60000scheduledsamplerdraws andindependentnumericAUC.
34455 all3COMPLETE(20K);34456cross eval and34457summary COMPLETE/PASS.
Cross primaryAUC perseed .668392/.665295/.667909 vsbase.677158;mean.667199.
Training/calibration meanAUC improves; crossmean remains slightlybelowbase,
conditional58sceneCI[-.038112,.018308] includes0. No crossclosedloopclaim.
34458withininputs afterok34413;34459withineval afterok34455:34458;34460summary.
Receiptsrun/jobs.json. Frozenrun/code+SHA256SUMS; pipeline/fresh_memory_geometry.py.
Auditorpipeline/audit_fresh_memory_setup.py separatefromfrozentrainingcode.
Next:monitor freshwithin34459 andsummary34460;cross complete asabove.
OldcrosssupportSUMMARYcomplete/audit26208arrays; keep separatefromfreshmethod.

# Active: support sampling pilot and within extension (2026-10-04)

User explicitly authorized within evaluation; IMPLEMENTED AND SUBMITTED.
Registration docs/MEMORY_SUPPORT_WITHIN_RUN.md; run results/support_within_20261004/.
Same nine Uniform/Near/Mixed checkpoints, no retraining. 100 new generated scenes,
10 each from actual ten support-training templates. Existing compiled scenes all
had recorded exposure; use original generator's uncompiled parameter combinations,
original required/excluded flags, uniform seed20261004. Check task ID AND scene-goal
hash absent all8337compiled/2500recorded tasks. Isolated task binaries; no global
compiled/cache modification. Not official within fold; original diversity subset
selection not rerun, 00020 has no solvability flags. Do not uniquely attribute
within/cross differences to template transfer.
SelectionSHA879f6bec3143f5102f8a5582508450f3f381eb03bebba87422c5fe41f4134a27.
34405selection/34406smoke/34407candidates/34417independentaudit COMPLETE/PASS.
Smoke: two scenes x8actions, full simulation/encode/pack/base/memory,16rawarrays
verified; natural-first-failure path exercised, no N/H/F-eligible smoke scene.
Independent audit checked100freshscenes,16initial-frame matches andsimulator labels.
Datachain34408–34413 COMPLETE. Eval34414 indices0–3COMPLETE,4/5RUNNING,6–8queued;
summary34415waiting. Atmost2GPUs/stage; originalcross complete.
Fullfirst5000cacheactions ->same historical originalbase/probe top1024 shortlist,
then actualsim128frames andfrozenPCA128latents. No successlabel shortlist filtering.
Officialeval stablelog64,4heads,batch32; naturalfirstfailure andselectedN/H/F panels.
Interim naturalwithin76sceneAUC improvement recordedattop; all9modelsummary pending.
SelectedN/H/F panelcoverage51scenes; no fullrecipe comparisonconclusion.
Pipelinewithin_support_data.py/within_support.py; frozenrun/code SHA256SUMS.
Largeassets /mnt/meg/jjenchen/phyre-unc/support_within_20261004/.
Next: monitor both job chains, check dependency failures/logs; retrieve and audit
complete summaries/rawmetrics before claiming effects.

## Completed independent failure review (2026-10-04)

Explicitly user-requested independent agent review complete. Report
 docs/INDEPENDENT_MEMORY_FAILURE_REVIEW_20261004.md, shared mirror SHA verified.
Reproducible read-only audit results/independent_memory_review_20261004/
{audit_existing_data.py,DATA_FINDINGS.json}. No new inference/training/jobchanges.
Strongest causal evidence: objective misalignment from matched L1 vsgeometry
intervention. Not a unique root cause; no invalidating implementation defect found.
Actual currentpilot draws41.79–42.51%successquery;11–17/2000stepswithoutsuccess.
Near unique donors442–453 vsUniform1407–1431; Nmedian2/scene. Final171scenes525N
none initiallynear (do NOT substitute428/2758fullinventory statistic).
Natural firstfailures70/74far>=20px,2N,2middle; Near support shiftmeasured,
but forcedN stillbelowbase, thus notsoleexplanation. N is trajectoryproximity,
not actionnearness or support-query correctionapplicability; model alreadyconditional.
Parent supplementary INTERIM_CROSS_SEEDS01_DETAIL.json independentlychecked2292
primaryheadAUCarrays forpairedseeds0/1; selected37AUCscenes base.672911 vsall9arm/
donor combos .621706–.627160. Natural harmconcentrated00007, not alltemplates.
Recommended next controls (NOT launched): samepanelunfinetunedR0 +fixedtrainingpanel,
then multipledonors/queryapplicability, beforedataquantity orarchitectureredesign.
Cannot inferthispilotcausedallharm withoutsamepanelR0; no trainvsheldoutor scalingcurve.
Originalprobe vsadaptedheadinteraction remains, not a proved soleheadmismatch.

## Original support sampling matched training pilot

User authorized RUN after questioning pretraining validation. Old-model support
swaps are optional, NEVER a gate for training. Current work directly compares
Uniform/Near/Mixed K1 training with shared R0/query/loss/budget, 3 seeds each.
Design docs/MEMORY_SUPPORT_UTILITY_EXPERIMENT_PLAN.md; concrete registration
docs/MEMORY_SUPPORT_UTILITY_RUN.md. No subagents. No new gate/loss/architecture.

Run results/support_utility_20261004/ on local/shared oprime. Scripts
pipeline/support_utility.py, test_support_utility.py, audit_support_utility_data.py.
Frozen code/ and separate prep_code/ preserve executed code. Four CPU pairing /
cluster / top20 tests PASS. Jobs:34366inventory,34367/34368supportscores,
34369finalize COMPLETE;34371independentDATA_AUDIT PASS (470 potential scenes,
all100 natural cross starts). 34370 GPU engineering smoke PASSED,168rawarrays
verified. 34372train COMPLETE:all9models2000steps,frozenbase/probeunchanged.
34373eval:8/9complete; onlymixed_s2 RUNNING (133/364case outputs atcheck).
34374summary waitingfor eval;SUMMARY.json doesnotexist. No finaleffectclaim.
INTERIM_STATUS_SEEDS01.json: complete paired seeds0/1 natural-first-failure,
primaryadaptedp0,58AUC-definedscenes. Independently recomputed696AUCarrays via
positive-negative pair comparison. Meanacrosstwo seeds base.677158,
uniform.629969,near.623895,mixed.629534. Near worse thanuniform bothseeds;
allthree belowbase. Interimonly; no complete3seed/selectedsupport conclusion.

Common training:171/1200 scenes across10 of12 trainingtemplates,3777query,
1612successfulqueries,N525/H367/F1758 supports;uniformfailures2889 including
intermediate distances. Initially322scenes hadN+far;only171 alsohaveH+F.
64calibrationeligible;49eval scenes (max16/template,hashselected). Cross D0
all100 natural first-action panel,74firstfailures;AUC requiresbothclasses.
Cross selectedsupport panel39scenes:00000=23,00002=10,00007=6,00014=0.
Do NOT extrapolate selectedsupport effects to00014;naturalpanelretainsall100.
One hash-fixed donor/category for this bounded pilot. No fresh D1 or K4 or
closed-loop inference scheduled in this stage; those remain subsequent work.

Manifest SHA708788f5d7f8c1dc1e2250e6875850f7215abb841e00cad476eee715705d0a17.
Inventory SHA978a97d719ebee48263d005b415b690ae58f0ad86c1d00dc938c6523943d47b2.
5/20px sampled-geometry N/far boundaries,top20% allsupport base adaptedp0score,
actualfailedsupportonly. Discovery/support/query grouped20/40/40;knowncluster
transitive groups stay together,grid fallbackwidth.02;task/action aliasesdedup.
Training fixed2000steps/armseed,lr3e-5,L1+.1originalprobegeometry,finalcheckpoint.
Support RNG separatefrom sharedscene/query/time;actualdifferentfirstlossvalid.
Allfour readouts evaluate;crossbase scores/geometry/L1 exactreplay againstprior.

Next: monitor34373/34374 and the submitted within chain above. Retrieve and
independently audit raw metrics when complete. Current49calibrationtasks are
memory-heldouttemplates, not within; new generated within scenes are separate.
Report support usefulness and
naturalpanel transfer, not learnedselection from K1 alone. D1 exposure audit /
collection, K4 and closedloop remain later stages, not completed results.
Prior authorized taskcentric stage deferred behind support-data attribution;
fallback findings below remain posthoc, no new confirmation.

## Latest requested analysis: why coverage is lower than base

Completed trace-only diagnosis and exact counterfactual replay; no newGPUjobs,
modelinference/training,ornewtasks. Report docs/MEMORY_CLOSED_LOOP_LOSS_ANALYSIS_20261002.md.
Scripts pipeline/analyze_closed_loop_losses.py andverify_closed_loop_counterfactuals.py.
Run/LOSS_ATTRIBUTION.json andLOSS_ATTRIBUTION_VERIFICATION.json;14400policy/task
replays,633381actionchoices independentlychecked. Freeze10=actual andfallback1=base
foreverycase. Allcutoffs1/2/4/10 retained,exploratoryposthocnotnewtest.

Geometrymemory losesmean5.111base-solvabletasks/100,gains.889:net-4.222coverage.
46lostconfiguration/taskoccurrences are11distincttasks (NOT46independenttasks);
25occurrences00007,12in00014,9in00000. All46ranksworseafterfirstupdate;
44/46alreadybeyondremaining99attemptsunderfixedfirstupdatescores. Only2wouldbe
rescuedbyfreeze1;both00014:026 differentseeds (base4/7attemptsuccess).
Examplem0p1afterfirstfailure samequerypool:00014:376 base25→memory786;
00007:05726→312;00007:12520→132. Firstupdateharmbeforehistorydivergence.

Posthocexactpolicyreplay:keepcurrentfirst10attempts;ifallfail,startbaseordering
at11withalreadytriedactionsexcluded,totalbudget100. GeometrymeanAUCCESS .468703,
S10.47,S100.68 vs actual .462241/.47/.637778. All46lostoccurrencesrescued;
doesnotmeanfreeunionofsuccesses,originalmemorylategainscanbelost. Freeze1 instead
.461424/.461111/.643333:simplystopupdatesdoesnotfixfirstbadcorrection.
Fallback2=.469726/.486667/.68;fallback4=.469550/.486667/.68;allposthocnooptimal-
thresholdclaim. Earlybudgetspentcouldharmbasecoverageonnewtasks;notguaranteed.

Interpretation:twolevels—harmfulfirstsupportcorrection,andpersistentuseofthat
rankingafter10failedupdates. No uniqueproofwhycorrectionlearned:query-conditioned
memorymaymisjudgeapplicability;averagegeometrynotranking;historicsupport-half
K1trainingvsadaptedhead1024deploysupportshift;readouttrainedonoldR0notnewgeometry
model. K4/historyalonecannotexplainfirstK1harm. Noresidualcopy/collapseclaim.

AUCCESSdecompositiongeometryvsbase:commonsolvedspeed+.014872,gained+.000708,
lost-.007408=sum+.008172. Gainsmostlyfasterexistingsolvabletasks,notnewcoverage.
Newevidencesuggestsbasefallback/gatingcontrolandfreshsceneconfirmationbefore
large representation redesign. Taskcentricminimumremainsauthorized/unlaunched;
retainfallbackandfirstupdatesafety/usefulnesscontrols. ThisanalysisdidNOTlaunch
newfallbackdeploymentorconfirmatorytest. Usercurrentrequestwasanalysis.

# Completed: adapted-probe closed-loop comparison

Authoritative results docs/ADAPTED_PROBE_CLOSED_LOOP_RESULTS_20261002.md.
Registration docs/ADAPTED_PROBE_CLOSED_LOOP_RUN.md.
Run results/adapted_probe_closed_loop_20261001/ local/sharedoprime.
34280smoke/34281audit/34282all22evaluations/34283fullsummary COMPLETE.
SUMMARY.json complete=true,trace_audit_passed=true;2200condition/taskpairs,
98441actionselections auditedremote AND independentlyreplayedlocally.
Local2200AUCCESS/aggregates/firstupdateranks recomputed;4423downloadedJSON/NPZ
hashesmatchremote. VERIFICATION.json records scope. Rawpackaction/labelalignment
wascheckedremote;fullscores/tracesarelocal. Priorprogressreportsaresuperseded.

Same100crossD0tasks4templates,1024candidates/task,100attempts,10failurememory
updates,keep4,successstop,stablelog64,batch32. Cachedsimulatorreplay,no newrollouts
ortraining. Frozenstableengine/common byteidentical topreviousstableevaluation.
Allinitialscores exactlyreplay frozenhead/base outputs;emptygate samefirstaction
withinhead;originalbase reproduceshistoricalfirstsuccess. Resumeartifactspreserved.

Complete group means:
                         AUCCESS      S10       S100
baseoriginal1            .454133      .4700     .6200
baseadapted3             .454070      .4600     .6800
latent_only3x3           .452687      .4778     .6289
latent_geometry3x3       .462241      .4700     .6378
Allseedsretained;9combinations sharebase/R0/probeinit,not9independentfullruns.
Geometry-minus-sameheadbase AUCCESS positive9/9,mean+.008172,
range+.005615to+.011860. Geometry-minus-matchedL1 positive9/9,mean+.009555.
ButgeometryS100 lowerthansameheadbase9/9;mean63.78%vs68%. Do notclaimuniform
memorysuperiority orstatisticalsignificance. PureL1vsbaseAUCCESSnegative8/9.

Firstfailureupdates (sameheadcommonfirstfailure;excludeonlythatfailedaction):
probe p0/p1/p2 has58/57/54taskswithsolutions. Latent_only firstupdateAUCdelta
mean-.056705,rankimproves15–21tasks/worsens30–35 perconfiguration.
Geometry firstupdateAUCdelta mean-.027378,improves18–22/worsens28–31.
All9geometryAUCdeltasnegative:lessdamage,notreliablepositive supportutility.
Do notreuseold57denominator orsumoverlappingseedtasksasindependentsamples.

Per-template averagedAUCCESS baseadapted→geometry:
00000 .541887→.547999 (+.006113)
00002 .894625→.913925 (+.019300)
00007 .029791→.008613 (-.021179),S10017.33%→8%
00014 .349976→.378429 (+.028452),S10076%→70.67%
Globalgainnotuniform;memoryK1finetuningtoK2–4deployment shiftscontextnumber
andlaterpolicyhistory. Cannot attributeallclosedloopeffectstoloss/probealone.

Interpretation:geometrysupervisiongivesmodestpairedAUCCESSgainonreusedD0;
finalcoverageandfirstsupportbenefitremainunresolved. Probeadaptationimproves
latebudgetcoveragebutnotaverageAUCCESS. Nextcompacttaskspacepilotmustretain
thesecontrols andsupport-querydiagnostics;lowerlatentlossalone notsuccess.

# Completed results: probe adaptation and geometry supervision (2026-10-01)

User authorized probe adaptation → geometry supervision → task-centric latent
when useful. Both registered experiments COMPLETE and independently audited.
At completion both runs had no remaining jobs. Task-centric pilot remains AUTHORIZED
but NOT implemented/submitted; current action is the registered closed-loop
comparison above before representation changes. No subagent authorization.

Authoritative new report docs/PROBE_GEOMETRY_RESULTS_20261001.md.
Probe run results/probe_adaptation_cross_20261001/:34243–34247 complete;
129scenes/5160rawarrays verified remote. Geometry run
results/memory_geometry_followup_20261001/:34265smoke/34266sixtrainings/
34267sixevaluations/34268summary complete,3096rawarrays verified.
Local independent1704macros from903scene summaries and paired sampling parity;
943downloadedJSON hashes matchshared. LargeNPZ remote only; audit scope recorded
results/PROBE_GEOMETRY_VERIFICATION_20261001.json. Frozen sources preserved.

Cross sameK1/query,73nonempty/57AUC scenes:
                        originalProbeAUC    adaptedProbeAUC
base                    .565401             .710069
originalR0              .505159             .628841
matchedL1finetune        .514331             .641259
L1+geometryfinetune      .556020             .676733
Adapted averages3heads;finetunes average3memoryseeds,adaptedfinetunes all3x3;
not nine independent full runs. Probe true_controlbase/memory .561728/.503834:
extra true-latent probe training alone not samegain. Readout helps both,base more;
memory-baseAUCgap worsens -.060242→-.081228. No memorysuperiorityclaim.
Probe mixsuccess geometry:base32.419→32.131px;R041.232→38.664px.

Geometry vs pairedL1 under originalprobe:successgeometry40.166→34.019px;
adaptedheads37.754→34.558px. OverallL1 .114315→.114966 (WORSE), successfulL1
.131544→.132279(worse),yetAUCimproves. All3memoryseeds and all9adapted-head
combinations show aggregateAUC/geometryimprovement. Calibration28AUC/29scenes
samepositive directions. Stillbelowbasewithsamehead. No newclosedloopAUCCESS.
00007 originalprobeAUCdelta geometry-L1 NEGATIVEall3seeds(-.0314,-.0155,-.0222)
despitegeometry10–14pxbetter; adaptedheadAUCdelta positiveall9,interactionimportant.
OtheroriginalheadtemplateAUCdeltaspositive,strongest00014(+.1106to+.1182).

Probe initial-only trainedhead averageAUC.4243;mixheadinitial-inputablation .5860
vs mixR0 .6288;seed0ablation nearlytiesR0. Does not establish correcttemporal
physicalinformation; ablation changesinputdistribution. Keep shortcutlimits.

Registered geometry continuation criterion PASSED cross+calibration3/3seeds
underoriginal andaverageadaptedheads. Direction-only exploratory criterion,
not significance,notuniformtemplateimprovement orbasewin. Next authorized step:
compactphysicalalignment+embeddingreconstruction pilot,matched controls before
launch;do not conflate changinglatentspacewithprobeadaptation. Preservecommon
externalgeometry/rankingmetrics,train-onlyphysicalsupervision, no futurequery
inputs, and fairbase/readout comparisons. May first make lightweight objective
intervention to isolate task-space weighting; do not label as fullTC-WM replication.

Registrations remain docs/PROBE_ADAPTATION_RUN.md and
MEMORY_GEOMETRY_FOLLOWUP_RUN.md. Newreport and teammate report containlatestfacts.
Same331trainscenes5289queries12templates,29calibration464queries4holdouttemplates,
100cross4unseen-to-fittemplates;historicK1supporthalf notstable1024deploypolicy.
Allcross evidence reusedD0; seeds sharepretrainedbase/R0. Positiveprobeintervention
supportsreadoutlimitation but not unique distribution-shift cause. Singlegeometry
weight/data setting is not general proof; compactspace nextstage untested.

# Completed: attribution stage 0/A

User-authorized phase0/A finished; no later branch launched. No queued/running
jobs on last squeue check2026-09-30. Audit34223, smoke34224,10shards34225 and
summary34226 produced complete outputs; sacct unavailable(accountingdisabled).
Run results/attribution_stage_a_cross_20260930/ local/sharedoprime project.
Report docs/ATTRIBUTION_STAGE_A_RESULTS.md; registration ATTRIBUTION_STAGE_A_RUN.md.
100D0scenes/4templates,73nonemptyK1,62K4,57K1withsolutions. Allrawalignment,
base/K1exactscore replay, all5emptypaths passed. Remote summarizer checked raw
scores/AUC/classL1; local recomputed2160macrovalues and100taskauditflags,
103downloadedfilehashes matchremote. VERIFICATION.json records limits.

Primary73scene overallL1 base.115010→R0.114210; A/Bmodels.113861–.114019.
57scene AUC base.565401→R0.505159; A/B.442402–.453620.
Successful geometryMAE32.419→R041.232; A/B47.890–49.879px.
CRITICAL correction: successfulL1 R0.131602 is LOWER thanbase.131889,
35/57scenes improved. Do not infer successfulL1alwaysworsens or classimbalance
alone explains failure. R062/73overallL1improve;41/57AUCworse;39/57both.
Same57sceneoverallL1base.116264→R0.115240;denominatorsnotexplanation.
ObservedlatentAUC.976895,truegeometry.994159;doesnotexonerateprobeonpredlatent.
Coarse/denseweightedtruegeometryAUC.994084/.994053:no aggregate rescue.
MatchedK4eligible R0K1AUC.438982→K4.439397 vsbase.500947:littlechange.
No causal identification of supporttransfer/readout/loss; reuseddevnotnoveltest.
Later B/C interventions proposed but not launched. No remaining required work.
LargeNPZsremote; summaries+manifest+verificationlocal. Sharedteammatereportupdated.

# Completed background: stable scorer cross

All12 runs34089 and summary34090 completed; no active/pending jobs.
Registration/results: docs/STABLE_SCORER_CROSS_RUN.md; teammate report
MEMORY_EXPERIMENT_REPORT_20260929.md updated with fullcrossresults(section10.1).
Run: results/stable_scorer_cross_20260930/. Shared project oprime:/path/to/phyre-unc.

Legacy→stable AUCCESS, unchanged S@100:
base .4541→.4541,62%;R0 .4420→.4419,61%;
A_s0 .4127→.4119,53%;B_s0 .4042→.4062,52%;
A_s1 .4166→.4173,55%;B_s1 .4099→.4110,55%.
No model's solved-task set changed. First-success counts changed in2/3/2/4/3/5
of100tasks respectively; not a count of intermediate action changes.
Allmemory stillbelowbase,allfinetunesbelowR0,BbelowAatbothfinetuneseeds.
No general method/significance claim; reused100task/4templatecrossdevpanel.
Fixedquery7scene AUC improvements remainvalid, but differenttemplates/pools;
final numerical scoring fix alone doesnot solve crossmemoryfailure.

Verified all1200taskAUCCESS, every action selection fromscores, failonlycontext,
successstop,updatebudget,task/pool alignment, originalbase/R0historicalreplay.
Fullaudit rerun locally;13raw/summary hashesmatchremote. Complete:true.
Per-task traces and score snapshots downloaded. No furtherexperimentstarted.
First-update saved-trace analysis now completed, no newforward/training.
Allmodel/baseinitialscores and firstselectedaction match exactly. Among57tasks
withfirstfailureandavailablesolution, firstupdaterank better/same/worse:
R0 22/8/27;A_s0 17/5/35;B_s0 17/5/35;A_s1 20/6/31;B_s1 20/5/32.
Newstable A_s0/B_s0:00014:273 rank2→106/141,00014:407 rank4→122/145
oncommonremainingqueries. Damage starts beforepolicydivergence; no proof of
physicalcorrectiondirectionoruniquelearningcause. Reproduce via
pipeline/analyze_first_memory_update.py; run/FIRST_UPDATE_ANALYSIS.json.
Reports updatedsection10.2. No remainingrequiredwork.

Earlier deployment pilot and scorer analyses complete. Latest report:
docs/MEMORY_DEPLOY_SCORE_DIAGNOSIS.md; teammate report updated in
MEMORY_EXPERIMENT_REPORT_20260929.md. Shared project oprime:/path/to/phyre-unc.

Jobs33987/33988 finished. Five models,12scenes,70prefixes,1,400nonempty/base rows.
Exact original scores reproduced; independently verified NumPy stable scores
from saved distances, scene-macro AUC and six artifact hashes. Outputs:
results/memory_deploy_score_diagnostic_20260929/{ANALYSIS,VERIFICATION}.json.

On7AUC-defined scenes, old→stable64/log AUC:
base .82442→.83907; R0 .78498→.85406;
A_s0 .78252→.86763; B_s0 .77730→.86240;
A_s1 .77632→.84402; B_s1 .78151→.84739.
All memory means exceed stable base on this tiny conditional fixed-query panel.
This reverses old-scoring ordering, NOT new cross improvement. B-A still mixed
-.00522/+.00336. Two finetune seeds shareR0pretrain, not independent fullseeds.

00012:076 policy AUC .5→.93548, success rank3 vsbase5. But R0 firstprefix
successful minimum surface distance91.77px vsbase5.24 andtrue2.59. Successful
surface MAE base25.97→R031.74, fine-tunes31.85–32.61 across7scenes. Numerics and
geometry problems coexist. Truegeometry/observedlatent AUC1 inall7; wrongscene
stable AUC remains above correct. No proof scene-specific memory use.
R0 float32 sigmoid .84069 vsfloat64/log .85406: preserve precision to sorting.

Prior pilot: 3,555rollouts,51pairedtrain/12valscenes; all4models2000steps,
base/probe frozen;50%legacy replay, actualK/querydraws matched. Original metrics
preserved. Reports keep historical cross/R0–R4 separate from this fixedpanel.
The recommended closed-loop comparison is now authorized/submitted above;
collection/training-score legacy remains acknowledged.

# Completed background: R0–R4 screening

User-authorized 2026-09-28 first wave completed: all five 20K-step models,
64-scene fixed-Q panels, and 100-task cross-f1 closed-loop evaluations.
Jobs 33806 and aggregate 33807 finished; SUMMARY.json complete=true. Additional
smoke 33813 completed both evaluators. No running jobs remain in squeue.
Retrieved raw outputs; remote/local SHA256 hashes match. Independently checked
500 per-task AUCCESS values, task IDs and shortlist solution counts against
cx_clm_base.json; every fixed-Q base/empty score matches exactly.

Cross AUCCESS: matched base .4541; R0 .4420; R1 .4277; R2 .4201;
R3 .4233; R4 .3962. Solved@100: 62/61/58/60/57/51 respectively.
No redesign variant beats R0 or base. Template 00014 solves:
base17, R0 16, R1 13, R2 14, R3 12, R4 6 (of25).
On 29 nonempty-context memory-heldout scenes, all variants have lower mean
ranking AUC relative to base. Correct/wrong-scene contrasts are inconclusive:
all conditional intervals include zero and 25/29 donors share the template.
Success-query latent errors rise in 29/29 scenes for R0–R3 and 28/29 for R4;
failure-query errors fall only in aggregate (R3 worsens in 19/29 scenes).
These are single-seed exploratory reused-dev results, not a final test result.
Architecture/data hypotheses remain hypotheses; do not claim memory is impossible
or ranking objectives universally fail from this wave.

Reports: docs/MEMORY_REDESIGN_RESULTS.md and docs/MEMORY_REDESIGN_RUN.md.
User-requested interpretation review completed; authoritative qualifications:
docs/MEMORY_REDESIGN_ANALYSIS_REVIEW.md. Post-hoc paired task resampling
conditional on four observed templates includes zero for R0-base, R1/R2/R3-R0,
and R4-R3. It excludes seed/novel-template variation and is not a significance
or equivalence claim. R1 changes actual K (2.1043→1.5359) as well as support
distribution; R2 bundles query sampling and loss weighting. `diverse_failure`
actually means uniform failure sampling. Existing memory attention already
conditions on query scene/action; no evidence yet requires architecture change.
Review metrics: results/memory_redesign_20260928/REVIEW_METRICS.json; reproduce
with pipeline/review_memory_redesign_results.py. No new training launched.
Subsequent user-requested failure reanalysis completed; report:
docs/MEMORY_FAILURE_REANALYSIS.md. New offline measurements on the two selected
HISTORICAL gated-checkpoint cases: support needed final-x correction +111.8/
170.5 px, successful-query mean needed correction -128.8/-127.3 px, actual
memory correction +31.3/+28.9 px. All 13/13 and 20/20 successful query
endpoint corrections have negative dot product with the required correction.
These are probe readouts; no new rollout or R0–R4 mechanistic validation.
Across all remaining queries, correction is not one constant vector (common
mean-vector energy fraction .0043/.2326). Reproduction script:
pipeline/reanalyze_memory_transfer.py; output:
results/cross_memory_diagnostic_20260928/transfer_direction_reanalysis.json.
Artifacts: results/memory_redesign_20260928/{SUMMARY,PAIRED_ANALYSIS}.json,
R0–R4 fixed/closed outputs, frozen code, dataset manifest, SHA256 verification.
Remote weights/resume: /mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928/.
Shared project: oprime:/path/to/phyre-unc.
Training union 99,419 unique actions, 1,200 scenes; 64 memory-only holdout scenes
from templates 00001/00004/00012/00022. Frozen base/probe saw these templates.
Next action: none running; report results. Further training is not launched.

# Completed: cross-template memory diagnostic

2026-09-28: user authorized running the fixed-support failure diagnostic.
Objective: explain memory ranking degradation on 00014:273 and 00014:407.
Script: pipeline/diagnose_cross_memory.py. Frozen existing cross-f1 checkpoints;
same query pool across conditions at each K, support is the failed prefix of
the base ranking. Compare base/observed/empty/full/gated/scaled, wrong scenes,
and visual/residual-zero ablations. No training or new simulator rollout.
Checks passed: historical base first-success 3/5 reproduced; gated-empty scores
equal base; saved-array ranks/AUC/latent means independently recomputed; all
downloaded remote output hashes matched.
Slurm array 33762 (indices 0/1) completed / exit 0, 4m41s / 4m46s, oprime.ib.
Output: results/cross_memory_diagnostic_20260928/{00014_273,00014_407}/.
Logs: logs/cx_mem_diag_33762_{0,1}.out on oprime.
Result: K=1 gated memory rank 2→350 / 4→141; average latent L1 improves,
but successful-query L1 worsens for 13/13 and 19/20 candidates. All successful
predicted endpoints move right (+31.27/+28.87px), away from actual left-side
solutions. More fixed failed context does not repair ranking; real-latent
probe AUC=1.0 on both exact shortlists. Selected dev cases, not general proof.
Report: docs/CROSS_MEMORY_DIAGNOSTIC.md; raw arrays/metrics/plots in output.
Next action: none authorized beyond completed diagnostic. Blockers: none.
Background: docs/CROSS_MEMORY_FAILURE_ANALYSIS.md.

# Historical PHYRE installation

Status: idle; installation completed and verified 2026-09-18.
Project: oprime:/path/to/phyre-unc; conda environment phyre.
Source: VAORA 4995736af63206e60504a3d3452bf7371f11fb76; PHYRE 0.2.2/Python 3.10.21.
Verified: 50 Python tests passed / 2 skipped; 40 C++ passed / 2 disabled;
pip check clean; 8,337 tasks; standard splits plus VAORA custom IDs 1–3;
one-ball and two-ball simulation/rendering; demo HTTP/assets passed.
Compatibility: Thrift C++ 0.11 built locally without SSL to avoid incompatible
conda OpenSSL pin; NumPy np.float alias fixed; missing frontend asset symlink
and generated JS supplied; conda activation sets PYTHONNOUSERSITE=1.
README.md contains activation, demo, reproduction and scope details.
Random single-ball examples completed and downloaded 2026-09-19:
examples/single-ball-random-20260919.zip and extracted directory, local and UNC.
Five distinct randomly selected templates, seed 20260919, first valid random
placement per task; no outcome filtering. All five NOT_SOLVED. Each 256x256
GIF is 20 FPS / 16.7 s. At user request, reduced extracted frames to exactly
10 evenly spaced PNGs per GIF (50 total), including first and last frames.
Deleted the other extracted PNGs and extra initial images, and replaced the
ZIP on both local and UNC copies. Full GIFs preserved.
Manifest records actions, rejected placements, source revision and timing.
Every decoded GIF timeline matches its PNGs; downloaded archive and extracted
file SHA256 checks match UNC. Reproduction: record_random_examples.py, then
thin_recorded_frames.py. Reduced ZIP SHA256:
b0f5e538cc9d2784c5301b7dc6ca3261ea701593c313e5dca2dce15b53657fe0.
No training, GPU jobs or persistent demo server. Next action: none. Blockers: none.

Official ground-truth solution lookup completed 2026-09-19. Official PHYRE
revision d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a evaluation files match
installed VAORA files byte-for-byte for the five example templates. Stored
stable-action counts: 00003:059=4, 00011:706=4, 00024:360=5, 00015:474=5,
00020:087=5. Replayed the first action for each: all solved; 00003:059 returned
UNSTABLY_SOLVED under stability perturbation testing, others STABLY_SOLVED.
Full actions, source URLs/hashes and replay outcomes: ground_truth_solutions.json
(local and UNC). Inspection script: inspect_ground_truth.py.

Successful 00015:474 recording completed and downloaded 2026-09-19:
examples/04_task_00015_474_success.zip and extracted directory, local and UNC.
Replayed first official stored solution; STABLY_SOLVED. Full 11.7 s GIF,
20 FPS, 256x256; exactly 10 evenly spaced PNGs. GIF/PNG timeline and download
checksums verified. Original random example preserved. Reproduction script:
record_success_example.py. No pending work.

Requested near-ball failed push completed and downloaded 2026-09-19:
examples/04_task_00015_474_near_ball_failure.zip plus extracted directory.
NOT_SOLVED; red surface starts 4.6 px from green, contacts about 0.5 s,
green moves at most 6.6 px right and stays in cradle. Same x/radius with
60 px higher drop solves; pre-contact speed 29.4 vs 88.2 px/s supports
insufficient impact interpretation. Full 16.7 s GIF at 20 FPS/256x256;
exactly 10 PNGs. GIF timeline and downloaded checksums verified.
Probe evidence: near_ball_probe.json / probe_near_ball.py. Export script:
record_near_ball_failure.py. Local and UNC artifacts match; no pending work.

Initial scene without red ball exported locally 2026-09-19:
examples/04_task_00015_474_initial.png (256x256). Rendered directly from
VAORA's pinned tasks.bin.lzma task 00015:474 using isolated local PHYRE 0.2.2
in /tmp/phyre-initial-render. No action added or simulation step performed.
Verified absence of red pixels. UNC SSH authenticated but command startup
stalled, so local rendering used. PNG SHA256:
ce4ba837148dd6daa6d0a17b15712fbd29ab560c70818f447ad422e7ab2dd810.
