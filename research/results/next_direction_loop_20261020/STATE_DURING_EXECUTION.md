# Active: iterative next-direction analysis (2026-10-20)

User explicitly requested a new result-driven experimental loop to decide nextstep.
Three bounded rounds implemented under results/next_direction_loop_20261020.
Frozen PLAN.json; no predictor/updater training, new simulations/encoder/100kselection,
slides or repo push. Existing current_probe_adaptation caches/pools reused immutably.
No subagents requested or used.

Jobs36662round1 COMPLETE;36663all8shortheadtraining smokes COMPLETE;36664audit PASSED;
36665GPU ranking smoke COMPLETE (old full1024scores exact, futuregeometryinput mutation
invariant, unchanged predictor; samefirstsupport acrossresidual/zero).
36666eightformalheads COMPLETE (38–41s actualtraining each, all2000steps).36667trainaudit
PASSED (identicalinitialization; matchedcontext/targetdraws).36668discovery8shards
RUNNING on8GPUs node005 at lastcheck. Then
36668discovery8shards ->36669adaptivechoice ->36670confirmation8shards ->36671final.
No new encoder services needed. All jobs under user jjenchen on sunflower.

Round1 cachedmatchedhistory analysis usesFIT64 only toselect memory variant.53firstfailure
scenes. Allmethods samefirstaction/support, maskit alsofromQ0 before rankingremaining1023.
L1raw top8=22/53 vszero25/53 vsQ018/53; L1center23 vszero25;Smoothraw16 vszero17;
Smoothcenter18 vszero17 vsQ018. SelectedSmoothcenter byregistered residual-minus-zero
thenminusQ0 top8 rule. Only+1scene overzero; not an overallperformancewinner orrobust
contentbenefit. Actualtrialmemory stateafteronefailure; staticnext8ranking,NOTS@9.
Detailed pairedfit counts saved ROUND1_PAIRED_ANALYSIS.json.

Round2 arms each2000steps/batch128,AdamW3e-5wd.05cosineclip1, same originalcrossadapted
rankinghead initialization: Q0-domain50/50 observed/predicted, selectedmemory-domain50/50,
selectedmemory predicted-only, memory50/50 taskobjects-only (dynamicGREEN/BLUE vsall).
Two target-sampling seeds20261021/20261022; NOT twoindependentworldmodels/updaters/headinits.
Memoryarms identicalscene/state/query/time draws; mixed/predicted-only alsoidenticalobject
labels. Object-only changes sampledobject only, retains allscenes/actions/times; noNearfilter.
Memorytraining53scenes133cachedstates1064queryinstances65positiveinstances. Truth/predicted
inputs exactly cachedFP32 representations; currentcheckpoint residual/history previously
collected. Onlyprobe gradients/parameters update.

Discovery16 andconfirmation16, one scene eachper16template, drawn fromprevious32adaptation
validation (alreadyexamined; not untouchedtest). No evaluationdata forheadtraining.
Full1024scores onQ0 andchosenmemory+same-losszero afterFIRSTfailedtrial. Deepertrainingstates
remain1/3/5. Mainheadquality metricsstatictop1/3/5/8, AUC/geometryauxiliary.
Headgeometry reportedallobjects/taskobjects/closest-of8sampledtime (nottruecontinuous
contacttime). Oldandpreviousmixedscore arrays exactreplayed whereavailable.
Round2 chooseeligiblearm withtop8>=oldbothseeds,mean>0,andtop1>=oldboth; rankbyworstseed
thenmeantop8/top1/AUC. Ifnone, bestdiagnosticarmstilltested,notlabelledpromoted.
ConfirmationtestsselectedarmANDmemory_mixed comparator, Q0specialized separate; zero
controlsamefirstsupport. Finalrawranking metrics independentlyrecomputed bydecide.py.

Important finalreview limits (ROUND1_PAIRED_ANALYSIS.json): specializationchanges effective
per-domain exposure; predicted-only doublespredictedpresentations atsame2000stepbudget.
Do notinferpurecapacity/anchorcausality. AutomatedFINAL_SUMMARY recommendationis a screening
heuristic; independentlycheckmemory-vs-Q0-domain ANDoldQ0, not justvsoldmemoryandzero;
pairedscene countsandtwo-partition stability before recommending largertraining. One-scene
wins require cautious follow-up. Closedloopnotpartofthisscreen; nonewS@1/3/5/9.

Next: monitor36666through36671, inspectfailureifany; preserve frozen source/PLAN onedits.
Aftercompletion independently audit/downloadrawscores, analyzeallcomparisons, decidebestnext
experiment andreportuser. Userauthorized autonomous iteration; don'tstopatmerelaunchstatus
if results reveal a meaningful additional necessarycheck. Root/watch.py monitorscompletion/errors andwillnotifythisthread; WATCH_STATUS.json
islatestpoll. A same-thread backgroundmonitor isbeingstarted afterthisentry. Never
interpret queuednotification as completion without checking remote FINAL_SUMMARY.

Previous completedprobeadaptation state archived root/PREVIOUS_STATE.md. Previousresults
current_probe_adaptation_20261020/COMPLETION_AUDIT.json: mixedhead geometry .7216->.7077
butAUC .7326->.7197;geometrygainQ0only, all6memoryvariants worsegeometry; no closedloop.
Remote /path/to/phyre-unc, .venv-ml/bin/python, sunflower/oprime sharedfilesystem.
SSH BatchMode=yes GSSAPIAuthentication=no ControlPath=none ConnectTimeout=8.
Never copy/read auth.key. OriginalclassifierA5000 numericalmismatchirrelevanthere: allpools
reused canonicalA6000 scores, predictor/probeA5000 reference replay required.
