> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Stable scorer cross closed-loop registration

Registered 2026-09-30 UTC, before evaluation. User authorized the matched full
closed-loop comparison after the fixed-query scorer diagnostic.

## Question and primary comparison

Does the recovered fixed-query ordering improve actual sequential selection
on the existing cross-f1 development panel? Primary comparisons are each of
five memory checkpoints against base, all using stable scoring. Also compare
stable versus legacy scoring within each checkpoint. A/B contrasts use the
same scoring mode and paired fine-tuning seed. No model selection or training.

Models: base; original R0; A_s0, B_s0, A_s1, B_s1 from the completed deployment
pilot. Every model receives both legacy and stable evaluation: twelve runs.
The four pilot checkpoints share the original R0 initialization. Their two
seeds replicate fine-tuning only. Historical R1–R4 are outside this narrow run.

## Fixed protocol

- Same 100 cross-f1 development tasks, four templates, and saved 1,024-candidate
  shortlist per task as the original closed-loop evaluations.
- Maximum 100 attempts, stop on first actual cached simulator success. For
  memory, update after a failed attempt at most ten times, retaining four
  most recent failures. Freeze memory thereafter, as in the old evaluator.
- Same frozen checkpoints, initial latents, bf16 model forwards, fp32 probe,
  NumPy surface geometry, batch32, empty gate, and earliest-index tie-breaking.
- Legacy score uses the original float32 tanh soft-contact sum. Stable score
  uses the SAME predicted distances, converted to float64, then
  `logsumexp(logsigmoid((tau-distance)/scale))`. Keep float64 through sorting.
  Log is monotonic: this ranks the original mathematical sum without losing
  tiny tails. It changes neither model parameters nor predicted trajectories
  for a fixed context. Subsequent contexts may differ because selection differs.
- Query prediction uses initial features only. True future latents/outcomes
  enter memory only for already attempted failures. Full saved outcomes are
  used for evaluator bookkeeping, never preselecting an action by its label.
- Reuse the historical shortlist and encoding route. No new shortlist
  generation, collection or simulator rerun. This isolates downstream scoring;
  it does not remove old-scorer effects on training data or shortlist coverage.

Metrics: AUCCESS, Success@1/10/100, per-task first-success attempt, per-template
results available from raw rows. This is reused development data, not an
untouched test or a new estimate over independent template draws.

## Checks and execution

Frozen source is copied into `results/stable_scorer_cross_20260930/code/`.
Each completed task stores resumable output, checkpoint/source/metadata hashes,
both score channels at every update, selected action and context traces.
Resuming requires identical arguments and provenance.

Preflight checks: one failure-sensitive cross scene (00014:273), base/R0,
both scoring modes, at most one update. Check empty-memory equality, original
base first-success reproduction, log-vs-direct stable ordering, and finite
scores. Full twelve-run array depends on successful preflight.

Final audit replays every selected action from score snapshots, verifies no
repeated actions, failure-only context, success stopping, update budget,
task/shortlist identity and AUCCESS. Legacy base and R0 must reproduce their
historical per-task first-success counts. Aggregate is published complete only
after every run and audit passes. Original result files remain unchanged.

New evaluator: `pipeline/closed_loop_stable_scores.py`, derived from the frozen
original closed-loop evaluator. Scorer diagnostic used a Torch geometry
implementation; this comparison retains the original NumPy closed-loop route
in BOTH arms rather than silently changing two components.

Shared outputs: `oprime:/path/to/phyre-unc/results/stable_scorer_cross_20260930/`.

## Submitted jobs

- Preflight34088 PASSED: historical base first-success3 reproduced; empty
  identity, log/direct ordering and paired score-channel checks passed.
- Twelve-run evaluation array34089 (indices0–11, concurrency capped at2),
  started after successful preflight; indices0/1 running at launch verification. Mapping: model order base,initial_R0,
  A_s0,B_s0,A_s1,B_s1; even index legacy, odd index stable_log.
- Summary/audit34090 depends on all twelve evaluations succeeding.
- All runs checkpoint after each completed task and can resume with identical
  arguments/provenance. Frozen source SHA256 manifest is in the run directory.

## 完整閉環結果（2026-09-30 已完成）

12/12 組與彙整核對均完成。目前沒有執行或排隊中的實驗。所有 1,200 筆逐題結果、逐次 action 選擇、context 更新與停止條件，已在遠端及本機通過核對；舊 scorer 的 base／R0 逐題成功次數重現歷史結果。13 個結果檔案 SHA256 一致。

| 模型 | 舊 scorer AUCCESS | 穩定 scorer AUCCESS | Success@100（兩者相同） |
|---|---:|---:|---:|
| base | 0.4541 | 0.4541 | 62% |
| initial_R0 | 0.4420 | 0.4419 | 61% |
| A_s0 | 0.4127 | 0.4119 | 53% |
| B_s0 | 0.4042 | 0.4062 | 52% |
| A_s1 | 0.4166 | 0.4173 | 55% |
| B_s1 | 0.4099 | 0.4110 | 55% |

穩定 scorer 沒有改變任何模型的已解題集合，也沒有改變 Success@100。各模型的首次成功次數只在 2–5/100 題改變；AUCCESS 最大增加約 0.0020。這裡的「改變題數」不是 action trace 有變的題數，不能據此說中間決策完全相同。

使用穩定 scorer 時，全部 memory 模型仍低於 base。A/B 微調都低於原始 R0；B 在兩個微調 seeds 都低於 A（差值約 −0.0057／−0.0062）。這是這批固定模型與重複使用的開發資料的結果，不是一般方法優劣或統計顯著性的判定。

需要區分兩個結論：數值缺陷確實存在，且曾影響固定 query 的 AUC；但只修正最後的 scorer，沒有解決本次 cross 解題退步。先前 AUC 診斷的 memory 保留模板為 00001/00004/00012/00022，這次 cross 是 00000/00002/00007/00014，候選數與閉環過程也不同；不能把兩者當成同一批題目上的矛盾結果。

保留原本小型固定 query 的結果，不以本次 cross 負結果抹去它。後續可用已保存的 traces 檢查分數改變是否影響前排候選，以及 memory 修正在哪次更新開始傷害成功 action；本次沒有再啟動訓練或其他新實驗。


## 第一次 memory 更新的失效位置（事後 trace 分析）

使用已保存的穩定 scorer 分數，檢查第一次失敗前後的排序。先確認各模型與 base 的初始分數完全相同，選到的第一個 action 也相同，再從兩邊比較的 query 中排除這個已嘗試失敗。因此，下面的變化發生在共同 context 下，尚未混入後續政策分歧。

100 題中，有 57 題第一次失敗且候選池內有解；另外的第一步成功題與無解候選池不進入此分析。

| 模型 | 成功候選排名改善 | 不變 | 變差 |
|---|---:|---:|---:|
| 原始 R0 | 22 | 8 | 27 |
| A seed 0 | 17 | 5 | 35 |
| B seed 0 | 17 | 5 | 35 |
| A seed 1 | 20 | 6 | 31 |
| B seed 1 | 20 | 5 | 32 |

例如 `00014:273`，排除第一個失敗後，base 的第一個成功候選排第 2；A seed0 更新後排第106，B seed0 排第141。`00014:407` 是第4→122／145。這些數字來自新微調模型與穩定 scorer，不是重用舊模型案例的排名。A_s0／B_s0 最後在這兩題都未能於100次內解出，而 base 原本能解。

目前可以定位的失效是：非空 context 引起的修正在第一次更新就會損害一些成功候選的相對排序。這些案例不需要先經過多輪錯誤累積才開始退步；但後續更新仍可能放大或修復，第一次排名下降本身不等於最後必定失敗。

它為何被學出仍未隔離。結合其他實驗，support 經驗對 query 的適用範圍學得不夠好，以及 latent loss 未保護決策重要部分，是值得優先查的解釋。此處保存的是分數，不能據此宣稱新模型把所有 query 往同一個物理方向修正、已證明 residual 複製，或資料不平衡就是唯一原因。

分析不新增訓練或推論。重現：`pipeline/analyze_first_memory_update.py`；完整逐題結果：`results/stable_scorer_cross_20260930/FIRST_UPDATE_ANALYSIS.json`。
