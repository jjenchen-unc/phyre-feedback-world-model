# Active: success-objective iterative experiments (2026-10-21)

User explicitly set goal: design small experiments, iteratively verify BCE/pairwise,
geometry scorer vs learned success readout, residual utility; provide clear direction
by TODAY Oct21 09:00 America/New_York (confirmed async answer). Goal remains active.
Started ~02:00 EDT. No subagents authorized/used. No slides or repo push.

New isolated root results/success_objective_loop_20261021 local and sunflower shared
/path/to/phyre-unc. Historical code/caches unchanged. Old completion state
archived at newroot/PREVIOUS_STATE.md. Original next_direction study completed,
independent audit440vectors passed; file now synced remote for provenance.

Current stage: registered frozenPLAN/code; pack8GPU36703 completedall75activecached
scenes, observed scoreerror typically1e-14 andworst~4e-8 in initial logs. Calibration
36711 and dependent GPU smoke36712 submitted. No formal training yet.
First sbatch request mem48G rejected by clusterconfig; removed explicit mem requests
(no job had started). All40 sunflower GPUs initiallyidle.
Need verify numerical scorer parity and cache availability, then calibration ->GPU
smoke ->6freshupdater fits ->discovery16/confirmation16 fullpool evaluations ->summary.
Do NOT launch training before smoke passed. Source hashes frozen in PLAN at register;
if bugs fix before formal run, preserve failedattempt provenance and explicitly
version/re-register revised dependencies; no silent edits to frozen experiments.

Round1: one feedback update, same first Q0-failed action from immutable current_probe
cache. 64fit/16discovery/16confirmation scenes as prior next_direction study,16training
templates; all upstream seen; not new cross-template generalization. Active denominators
expected53/12/10. All6 freshupdaters seed7919,128steps, one scene8uniformuntriedqueries
perstep, same schedule. Loss L1future+.1geometry plus optional BCE/pair, realcentered
orzeroinput. Predictor/initialtokens/probes/scorer fixed. Q0 supportprediction cached
only because firstupdate and fixedinitialreader; never stale evolving residuals.
Pack requires existing full observation cache for supportand8queries; NO simulation
or encoder fallback. Predictor supportforward original32batch verified exact vs old
scores. Data pack storesfull32targets, queryinitial, features, supportpred/observed.

Differentiable Torch scorer reproduces frozen log(sum sigmoid((4-distance)/1))) with
same initial-only object queries and surface geometry. FP32geometry FP64logscore;
scoremaxerror<5e-4 gate, chunk order agreement recorded. Deployment evaluation retains
original exact NumPy scorer. Smoke checks real gradients and finite differences;
initial reader/head gradients disabled; futuregeometry input mutation invariance.
No success head added in round1. Fixed positive-affine logit calibration from TRAIN
full1023Q0 pools; same slope in BCE and pair. This cannot change static ranking itself.
Lambda each supervised loss via median initialstate gradientnorm ratio vs prediction
on first<=8FITmixedlabel scenes. Pair0ifqueriesalloneclass; prediction retained.

Adaptive secondround not predetermined winner: afterround1 inspect BOTHpartitions,
paired real-vs-ownbaseline andreal-vs-trainedzero, geometric errors. Ifpromising
replicate4fits baseline/winner xreal/zero underfreshsecondseed. Else inspect query
positivecoverage, score saturation/readout issues; bounded followupmaytrain a small
success readout on frozen prediction geometry. Max12updaterfits total in PLAN, no
largepretraining. Aim finish evidence/analysis before09:00, do not endatlaunchstatus.

Important code-review possibilities beforelaunch: smoke mustcatch memory/gradient or
numeric failures; registration imports GPU-free torch only in environment venv. ar.make_team
uses frozenQ0source, reconstruct freshPrefixPredictor seed; condition hook centered/zero.
Raw data scores originalQ0first should exactlymatch candidatechunk shape32.

SSH -oBatchMode=yes -oGSSAPIAuthentication=no -oControlPath=none -oConnectTimeout=8.
Remote .venv-ml/bin/python; sbatchdefq; max4h/job. No auth.key access.

Update: pack/calibration/GPU smoke completed. SMOKE passed finite-difference relative
error .016, updater gradients nonzero, futuremetadata invariant. Six training jobs
36713_0-5 submitted (128steps). Pack fit53scenes424queries32positives, only20scenes
with bothlabels. Discovery12active/6querypositives, confirmation10active/5positives.
Maximum scorer parity error3.31e-6, allcheckedchunk orders identical. Weights BCE1.5745,
pair.4001. Dependency evaldiscovery/confirmation arrays submitted, IDs pending.

IMPORTANT new analytical finding BEFOREtrainingoutcomeinspection: frozen affine
logitcalibration slope.11795 intercept-2.42209 and physical score upperboundlog32
imply maxsuccessprobability .1178137. Round1 BCE is thus a constrained calibrated
readout, NOT a fair general rejection test ofBCE. SavedCALIBRATION_RANGE_DIAGNOSTIC.json.
Need adaptive followup addressing this beforebestdirectionconclusion: e.g joint
learnable positiveaffine mapping remains ranking-invariant, with calibration-only
control and readerfixed. Or betterjustified fixed mapping with auditedrange. Do NOT
silentlychange frozenround1source orclaimBCEpoor generally ifitloses.

Latest ~02:20 EDT: all6fresh128step trainings36713 complete in47–59s each; initial
weights/reader/queryschedule identical. Actualpairactive50/128steps. Evaluation
36719/36720 failedbeforeanymodelwork due inheritedsys.path resolving wrong train.py.
Originalsource/PLAN untouched; root execution_v2.py explicitimportsowntrain/evaluate,
EXECUTION_FIX_V2.json records hashes. RealGPUcheck36736_0 COMPLETE, residual andzero
eval6arms working. Rest discovery36737array1-7,confirmation36738array0-7; summary
36752 afterok all3 using execution_v2.py summarize. No training rerun needed.

Adaptive round2a readout isolation REGISTERED and collecting8GPU36753; filesunder
root/readout, separate PLAN/hash. Collect fullQ0distance32times all1024, verifyoriginal
score replay exact; query observedlatent andsimulatorgeometryoracle scores diagnostics.
Fit two37feature logisticridge heads (fixed.01 regularization): sameuniform8query
labels vs full1023poollabels, no predictor/probe/updaterchanges. Compareoriginalscore
plusmonotoniccalibrationcontrol. Differentlabelcounts EXPLICIT, no fairmethodclaim
forfullpoolvs8query. No nearfilters/sceneIDs/actions infeatures. Readout fit dependency
submission in execsession80624; inspectID. Originalscore logitsceiling analytical
trigger, not resultcherrypick.

Adaptive round2b code PREPARED ONLY localroot/calibration_update/experiment.py (not
registered/submitted). Twofresh BCEreal/zero updaterfits withjointlearnedpositiveaffine
calibration (exp(logslope),bias), initialsameoldcalibration, same128step schedule,
updater optimizerunchanged; scalarAdam.03cosine separate. Inferenceoriginalgeometry
scorer unchanged, scalarpositiveaffinecannotalterrank directly. Budget8fits inclthese,
leaves4forsecondseed replicationifpromising. Need firstinspectround1results, register
separatePLAN, GPUtrain/eval. Goalstillactive, ~6h leftto09:00.

Update ~02:30EDT: round1 summary completeaudited by own aggregator. Discovery12 top8:
Q05,predreal3,predzero4,BCEreal4,BCEzero5,pairreal3,pairzero6. Confirmation10:
Q01,predreal2,predzero1,BCEreal5,BCEzero6,pairreal7,pairzero6. Top1confirmation
predreal1,BCEreal4,pairreal4. Geometryandlatenterrors increasewithsuccesssupervision
(e.g L1~.10->.15). Pair passesvsownpred baselinebothpartitions butfailsreal-vs-zero
in discovery. No overallwinner orresidualcontentclaim yet.

Round2a readoutcollect36753complete; firstfit36761failedassert exactpositiveaffine
rankpreservation becauseFP64rounding mergesnear-ties. FixedONLYscalarcontrolrank
implementationtoretainoriginalrawscores via experiment_v2.py/FIT_V2.json; original
PLAN/code untouched; fittedweightsunchanged, MODELS_before_fit_v2 preserved. Fitv2
36780complete. Fullpoolreadout train54219examples vsquery8=424; top8 discovery
Q05/12,query8=5/12,fullpool5/12;confirmationQ01/10,query8=1/10,fullpool3/10.
Oraclequerydiagnostic (notfullpool): mixed-labelscenesFIT20/disc4/conf4;AUC predicted
.635/.536/.845 vsobservedlatent .993/.964/1.0 vstruegeometry .993/1/1. This suggests
currentreadout canrank trueinformation; smallqueries donotproveglobalbottleneck.

Round2b jointcalibration REGISTERED,2fits36762complete;eval36764/36765complete.
BCElearnedcalmaxprob .135(real)/.141(zero), slopes .0648/.0746: remainslow.
Top1/3/5/8disc12:real0/2/2/3,zero0/2/3/4;conf10:real3/3/5/6,zero2/3/4/4.
Doesnotreliablyresolvecrosspartition instability; noBCEgeneralrejection.

Round3 REGISTERED replication/run.py plan;4freshseed7920 fits36781 launched:
predictionreal/zero,pairreal/zero. Same fixedQ0/initialtokens, dataset,128steps; new
updaterinitialization andsceneorder. Originaltrainmodule explicitbinding avoids
collision; evalsubset4arms viaev.ARMS, outputs originalroot/evaluation/seed7920.
Budgettotal12updaterfits nowallocated(6+2+4). Evaldisc/conf submittedin session218?
LATEST submit exec session notyetcollected; inspectoutputs then JOBS.

Round3 cross_transfer/experiment.py REGISTERED fixed20testscenes4per5unseentraining
templates, RNG20261021 independentoutcome; original100panel alreadyusedhistorically.
Pack8GPU36785 launched; originalQ0full1024+distancefeatures replayexact, supportfull
observationfromstrictcacheonly. Datareceipt/cachehashes. Evalall6seed7919+4seed7920
+2learnedcalupdaters +Q0/classifier/query8/fullpoolreadout on samefirstQ0failure,
12updaters? actually12 totaltrainedvariants, 16scoremethods including4static.
Transfer evalsubmitted dependency36781:36785:36762. Need monitorIDs/jobs/cachemisses.
All staticreadouts fromfrozenQ0; no newfittingontest. Onefeedbackconditionaltopk not
fullmultiupdateS@k. Checkclassifierbaselineaftercommonfailedfirst separately from
itsowninitialpolicy. Goalstillactive: finish replication+cross evidence, independent
rawaudits andclearrecommendation by09:00; optionaladditionalanalysisnoextratrainfits.

Latest ~02:40EDT: all12updater fits complete. Seed7920 replication eval36793/36794
complete. Pairreal top8disc6/12 vsownprediction7/12,zero4/12;conf4/10 vsownprediction
3/10,zero2/10. Priorlargegain doesnotfullyrepeat. Sourceweights frozen.

Cross20 pack36785/eval36795complete (19firstfailures,1firstsuccess). Independent
analysis_audit.py (localCPU, no importsfromtrain/eval) verifiesstoredrawscores.
Cross20 top8 among19: Q02;predictionreal3bothseeds;BCEreal0;pairreal0(seed7919),
2(seed7920);zero counterparts1BCE,0/1pair; fullpoolreadout2,query8readout3.
AUC Q0.626, predictionreal.557/.601, BCEreal.349, pairreal.311/.420. No successful
supervisiontransfer demonstrated; positive training-template results insufficient.

Evaluation-only extension to existing100crossscenes REGISTERED cross_transfer/
EXTENSION_PLAN.json andextension.py. Remaining80 jobs36817pack32GPU,36849eval32GPU
dependency. Reuses20records andall16methods (12updaters+4static); NO newfits or
model selection. Extensionreceipts addextension_planhash, parentPLAN/code unchanged.
New independentanalysis supports cross20 pluscross100 ifEXTENSION_PLAN copiedlocal;
need resync latestJSON/rawNPZ tolocal. No newly generatedobservationsperstrictcache.

Additional no-training mechanismtest REGISTERING update_amplitude/experiment.py.
Onfixedcross20,6successupdaterarms BCEreal/zero seed7919,pairreal/zero bothseeds;
state=initial+alpha*(updated-initial),alpha .1/.25/.5. Verifyalpha0/1 reference
chunkexact; storestateupdate norm andfull1024scores. ExistingQ0/originalupdates
provideendpoints. Distinguishesamplitudefromdirection; restorationtowardQ0alone
NOTpositivefeedbackevidence. Eval8GPU submissionexecsessionpending (see latesttool);
~8freeGPUswhilecross80uses32. Noadditionaltrainingbeyond12fits.

Historicaloraclefull1024reference located results/oracle_small_1024_20261017/
SUMMARY.json (cross10,oraclelatentS1.8,S9=1;truegeoS1=1;Q0S1=0,S9=.2). Hasrecovery
numericalaudit; do NOTciteascurrentmatchedexperimentuntilcheckinghead/poolprovenance.
Currentnight oraclequerycontrolledsubstudy alreadyvalid, butonly8validationmixed
scenes, notfull1024.

Completionremaining: finish100eval andamplitudecausaldiagnostic, inspectalljobstates;
independentlyauditrawscores/pairedtemplatecounts andCI, distinguishprobabilitymapping
constraints, querypositivecoverage, geometrydegradation andcrossgeneralization.
Produceclearbestnextdirectionwithlimits ratherthanclaimBCEorpaireverdominates.
Keepgoalactiveuntiltheseandfinalevidence/reportdone; noearlycomplete. Deadline09:00.
