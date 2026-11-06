> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# 適配 probe 與幾何監督 memory：完整閉環結果

2026-10-02。22/22 組評估完成，完整核對通過。目前本輪沒有執行／排隊工作。
這份結果取代先前的中途進度；task-centric latent 尚未實作或啟動。

## 結果回答什麼？

相同 cross fold1 的 100 題、每題 1,024 個既有候選，依序最多嘗試 100 次，
成功即停。Memory 只讀已嘗試失敗的 trajectory，最多更新十次、keep4。
這是保存 simulator 結果上的閉環重播，不是新增 rollout 或新的 holdout。

| 方法 | 設定數 | 平均 AUCCESS ↑ | Success@10 ↑ | Success@100 ↑ |
|---|---:|---:|---:|---:|
| Base＋原 probe | 1 | 0.4541 | 47.0% | 62.0% |
| Base＋適配 probe | 3 | 0.4541 | 46.0% | **68.0%** |
| 純 L1 memory＋適配 probe | 9 | 0.4527 | **47.8%** | 62.9% |
| 幾何監督 memory＋適配 probe | 9 | **0.4622** | 47.0% | 63.8% |

適配 probe 的 base 平均全部三個 probe seeds；memory 平均全部 3×3
memory/probe seed 組合，不挑最好者。組合共享預訓練 base、R0 與原 head，
不是九次獨立完整訓練。百分比先在 100 題計算，再對設定平均。

## 1. 幾何監督 memory 的 AUCCESS 有小幅正收益

相同 probe 配對，幾何監督 memory 相對 base：

- 平均 AUCCESS **+0.008172**，即約 +0.82 個百分點。
- 全部九個配對 AUCCESS 都為正，範圍 **+0.005615～+0.011860**。
- 相對同 memory/probe seed 的純 L1，九個配對也全為正，平均 **+0.009555**。

所以在這批固定 D0 場景和候選池上，可以支持「幾何 supervision 比純 L1
有用，且提供小幅 AUCCESS 收益」。不能把九個共享模型的組合當獨立樣本
聲稱統計顯著，或直接外推未見場景／模板。

## 2. 不能說 memory 已全面贏 base

幾何監督 memory 的平均 Success@100 **63.8% 仍低於 base＋適配 probe 的
68.0%**，九個相同 probe 配對也全部較低。因此這是解題效率與最後可解
題目數之間的取捨；較高 AUCCESS 不代表在所有指標都更好。

只做 probe adaptation，平均 AUCCESS 幾乎不變（0.454133→0.454070），
但 Success@100 從 62% 到 68%。先前已查到，多數新增成功發生在第十次
以後。相反地，memory 可能讓一些題更早成功，同時失去其他晚期可解題。
是否值得部署應依要優先的嘗試預算與指標判斷，不能只報單一贏的數字。

## 3. 第一次失敗經驗仍常讓排序變差

同 probe 的所有模型初始分數與第一個 action 精確相同。第一次失敗後，
排除該 action，再比較同一批剩餘候選；此時尚未混入後續選擇歷史的分歧。

| 第一次更新的診斷 | 純 L1 memory | 幾何監督 memory |
|---|---:|---:|
| 平均 AUC 變化，相對未更新 base | -0.0567 | -0.0274 |
| 第一個成功候選排名改善的題數，各設定範圍 | 15～21 | 18～22 |
| 第一個成功候選排名退步的題數，各設定範圍 | 30～35 | 28～31 |

各 probe 的可比較題數為 58／57／54，因為第一個選擇不同。AUC 變化的
平均先按各設定的有效場景計算，再平均九個設定。不可沿用舊固定 query
實驗的 57 題分母，也不可把九組題数加總當獨立題目。

幾何 loss 減輕第一次更新的損害，但所有九組的平均 AUC 變化仍是負的，
成功候選排名退步的題目也多於改善。閉環 AUCCESS 小幅上升，並不等於
support 已能穩定幫助 query；少數題的早期收益、整體候選排序與後續更新
是不同面向，仍需分開追查。這些分數不證明 residual 被複製或某種物理
修正方向，也不把 loss、probe 或 support transfer 說成唯一原因。

## 4. 模板效果不均勻

每模板各 25 題，先對全部 seeds 平均：

| 模板 | Base＋適配 AUCCESS | 幾何 memory AUCCESS | 差值 |
|---|---:|---:|---:|
| 00000 | 0.5419 | 0.5480 | +0.0061 |
| 00002 | 0.8946 | 0.9139 | +0.0193 |
| 00007 | 0.0298 | 0.0086 | **-0.0212** |
| 00014 | 0.3500 | 0.3784 | +0.0285 |

00007 仍是明顯退步模板，其平均 Success@100 由 17.3% 降至 8.0%；00014
雖然 AUCCESS 改善，Success@100 仍由 76.0% 降至 70.7%。整體平均不能
掩蓋这些異質性。新 memory 續訓使用 K1，但閉環後續使用最多四次失敗；
後續結果同時包含 context 數量與政策歷史轉移，不能只歸因為訓練 loss。

## 結論與已授權的下一步

目前最嚴謹的結論是：**幾何監督在這批 cross 上提供一致但小幅的配對
AUCCESS 改善，尚未改善到最後成功率或穩定的第一次 support 效用。**

Task-centric latent 的最小實驗仍值得做，但成功標準要同時保留同 readout
base、現有幾何 loss、AUCCESS／Success@100 與第一次更新診斷。不能只要
latent loss 更低、offline AUC 更高就宣稱成功，也不以換空間代替檢查
support–query 修正適用關係。準備稿見 [TASK_CENTRIC_MINIMAL_PLAN.md](TASK_CENTRIC_MINIMAL_PLAN.md)；
該分支已授權但尚未實作／提交，不能報成正在訓練。

## 驗證與檔案

- Slurm 34282 的 22 組與 34283 彙整全部成功。
- 遠端完整核對 2,200 組設定×場景、98,441 次選擇：候選／action／label
  對齊、初始分數、無重複選擇、失敗限定 context、更新預算、keep4、成功
  停止與 AUCCESS。原 base 精確重現歷史結果。
- 本機再次從完整原始分數／trace 重播同樣 98,441 次選擇，重算 2,200
  個 AUCCESS、aggregate 與第一次更新排名；4,423 個下載檔案 SHA256
  和共享伺服器一致。原 pack 的 action／label 對齊由遠端完整核對執行。
- 輸出 `results/adapted_probe_closed_loop_20261001/` 中的 `SUMMARY.json`、
  `RESULTS.md`、`VERIFICATION.json` 與 `evaluation/*.json.tasks/`。
- 執行設定：[ADAPTED_PROBE_CLOSED_LOOP_RUN.md](ADAPTED_PROBE_CLOSED_LOOP_RUN.md)。
