> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# 適配 probe 與幾何監督 memory：閉環對照

2026-10-01。使用者明確授權「跑」。本輪先驗證解題收益，再接續已授權的
task-centric latent 最小實驗；不是先根據較高離線 AUC 宣稱新方法更有效。

## 問題與比較

固定候選池，適配 probe 帶來的排序改善是否提高 AUCCESS？在相同 probe
下，memory 是否比 base 有額外收益？幾何監督是否比同預算的純 L1 有用？

| 組別 | 設定數 |
|---|---:|
| Base＋原 probe | 1 |
| Base＋predicted_mix 適配 probe | 3 probe seeds |
| 純 L1 續訓 memory＋適配 probe | 3 memory × 3 probe seeds＝9 |
| L1＋幾何監督 memory＋適配 probe | 3 memory × 3 probe seeds＝9 |

共 22 組，全部保留。不挑最好的 seed，不只用同編號 seed 配對。
兩種 memory 使用上一輪已凍結的 final checkpoints；不新增訓練。
適配 heads 來自原 R0/base 預測混合資料，沒有對新 memory 再適配。

主要比較：各 memory 對同一 probe 的 base；相同 memory/probe seed 下，
幾何監督對純 L1；各適配 base 對原 probe base。既有原 R0 的 offline
結果仍保留，但此次四組比較不加入額外 R0 閉環組。

## 固定執行規則

- 同一 cross fold1 的 100 場景、四模板、每題既有 1,024 個候選。
- 每題最多嘗試 100 個不同 action，第一次真實成功立即停止。
- Memory 只在失敗後更新，最多 10 次，保留最近 4 個已嘗試的失敗。
  第 10 次更新後固定 context，直到成功或用完預算。
- 所有權重凍結。相同 bf16 predictor／fp32 probe、NumPy 幾何、batch32，
  float64 穩定 log scorer 與最早 index 的 tie-breaking。
- 空 memory 必須等同同 probe 的 base。因此同 head 所有 memory 模型
  在每題第一次必須選相同 action，Success@1 必須一致。

這是**既有候選與已保存 simulator 結果上的閉環重播**：模型自行選擇下一
個 action，只有執行失敗之後，其真實 trajectory 才能進入 memory。
沒有重新收集 rollout、重新產生 shortlist，或拿 query 真實未來預先選 action。
Future labels 只供評估器判斷成功停止與離線診斷。

不同 probe 可能從第一個 action 就分歧；同 head 的不同 memory 則在第一
次失敗前完全配對，之後各自使用自己實際累積的歷史。不能把不同 head
下各自遇到的失敗當相同 support 的因果對照。

本輪 memory 續訓只使用 K1，閉環沿用既有 keep4 規則，後續可收到 2～4
條失敗經驗。因此第一步更新是最直接的 K1 對照；多次更新的結果同時包含
context 數量與政策歷史的轉移，不能全部歸因為 loss 或 probe 的單一效果。

## 指標與第一次更新的診斷

報告 AUCCESS、Success@1/10/100、每題首次成功次數、各模板與各 seed。
組別平均之外保留範圍和所有 22 組結果。九個 memory/probe 組合共享預訓練
模型，不是九次獨立完整訓練；不以 1,024 個 action 當独立統計樣本。

保存每次更新的全部候選分數、context、所選 action 與停止原因。第一次
更新比較先排除已嘗試的第一個失敗，再比較 base 與 memory 的成功候選
排名、AUC。全失敗候選池保留，AUC／成功排名未定義；另報無 memory
可用的第一步成功題。各 probe 的分母可能不同，不強制沿用之前的 57 題。

這仍是已重複分析的 D0 開發資料。閉環提升可以驗證這些固定模型在此候選
池的解題收益，不能直接變成新模板或新場景的確認性泛化結論。

## 工程檢查與執行

沿用已核對的 `closed_loop_stable_scores.py`，不改其選擇／更新邏輯。
新 wrapper `adapted_probe_closed_loop.py` 只負責固定設定、啟動與完整核對。

Smoke 使用固定兩題 00000:000、00014:273，每題最多一次 memory 更新。
四種 base/head 加六種 memory（各 memory seed 配相同編號 probe）共十組。
從上一輪既有分數確認這兩題在四種 head 下第一步都失敗，以確實執行更新；
選題只用於工程覆蓋，不篩選正式 100 題或方法。

核對要求：

- 原 base 每題首次成功次數精確重現歷史 stable scorer 結果。
- 每組初始全部分數精確重現上一輪同 head 的 base 分數，包括每個 memory
  checkpoint 的空 context。
- 從原始分數重播每次 argmax，確認不重複 action、同分規則、失敗限定
  context、最多十次更新、keep4、成功立即停止。
- Action／task／候選池／標籤對齊，重算每题 AUCCESS 及全部 aggregate。
- Manifest、checkpoint、metadata、程式 hashes 固定；各題輸出可恢復，
  只有相同參數與 provenance 才允許接續。

Slurm 順序：十組 smoke（最多2GPU）→CPU smoke 核對→22組正式閉環
（最多2GPU）→CPU完整核對／彙整。使用 afterok，上游失敗不啟動下游。
新 run：`results/adapted_probe_closed_loop_20261001/`，凍結程式在 `code/`。
Job IDs 在 `jobs.json`，當前狀態在專案 STATE.md。

## 後續 task-centric latent

已授權但本輪不混入 representation 改動。這輪完成後，先查幾何 memory
相對同 head base 的額外 AUCCESS，以及第一次更新是否仍傷害排序。
再把 compact physical alignment＋reconstruction 的最小實驗具體化，
保留本輪 base＋適配 probe 與幾何監督 memory 為基準。若結果不支持 memory
額外收益，需同時分析 support–query 適用關係，不以換 latent 空間作為
唯一失敗解釋。進一步工作已在使用者授權內，不需重問許可。

最小介入的準備稿：[TASK_CENTRIC_MINIMAL_PLAN.md](TASK_CENTRIC_MINIMAL_PLAN.md)。
該文件不是已凍結的訓練設定，representation 工作尚未提交。

## 已提交工作

CPU 前置核對已通過。34280 是十組 GPU smoke（同時最多兩組），34281
負責 smoke 的完整軌跡核對。22 組正式閉環評估 34282 與彙整 34283 已用
`afterok` 接續，只有前一步成功才會啟動。尚未發布新科學結果。

## Smoke 已通過、正式評估已開始

34280／34281 完成，20 組設定×場景、1,124 次選擇全部核對通過；原 base
歷史首次成功次數與所有 head 的初始分數精確重現。正式 34282 的前兩組
已執行，其餘依最多兩張 GPU 的上限排隊；34283 等全部評估通過後彙整。
目前只有工程驗證，尚無本輪正式 AUCCESS 結果。

## 完成狀態（2026-10-02）

34282 全部 22 組及 34283 彙整完成，遠端與本機完整核對通過。前文的
執行中狀態是歷史記錄；本輪已無執行或排隊工作。結果與限制：
[ADAPTED_PROBE_CLOSED_LOOP_RESULTS_20261002.md](ADAPTED_PROBE_CLOSED_LOOP_RESULTS_20261002.md)。
