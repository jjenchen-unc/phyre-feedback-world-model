> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# 閉環評估進度：base 組已完成，memory 尚未跑齊

2026-10-02。22 組中 6 組完成並通過獨立核對；2 組執行、14 組等待 array
並行上限。完整彙整 34283 尚未開始。這是中途進度，不是整輪最終結論。

## Base 的完整三個 probe seeds

同一 100 題、每題 1,024 個候選、最多 100 次嘗試，使用已保存 simulator
結果的閉環重播。每個 head 一律保留，沒有挑最好者。

| 模型／probe | AUCCESS | Success@10 | Success@100 |
|---|---:|---:|---:|
| Base＋原 probe | 0.4541 | 47% | 62% |
| Base＋適配 probe seed 0 | 0.4469 | 47% | 65% |
| Base＋適配 probe seed 1 | 0.4566 | 46% | 69% |
| Base＋適配 probe seed 2 | 0.4587 | 45% | 70% |
| 適配 probe 三 seeds 平均 | 0.4541 | 46% | 68% |

精確平均 AUCCESS 是 0.4540699，原 probe 是 0.4541330；兩者差約 -0.000063，
不能把四位小數相同誤稱為逐題結果完全相同。

可支持的觀察：適配 probe 提高 100 次內的成功率，但沒有帶來三 seed 平均
AUCCESS 的提升，前十次成功率也未增加。相對原 probe，各 seed 新解出的
題目數為 4／9／10，其中 3／8／9 題在第十次之後才成功；同時各失去
1／2／2 題原本可解的題目。因此收益多反映在較後面的嘗試，而不是一致
改善最前排候選。AUCCESS 較重視早期成功，與 Success@100 衡量的重點不同。

這不推翻先前 offline AUC 提升：AUC 衡量成功與失敗候選的整體配對順序，
不等同第一個成功 action 的排名。先前主要 AUC 使用原 base 有失敗 memory
且候選池有解的 57 題，本次閉環使用全部 100 題；不能把兩個 aggregate
視為完全相同分母、只改一個指標的對照。

## 已完成的兩個 memory 組合：僅局部結果

| Memory 設定 | AUCCESS | 同 probe 的 base | S@100 | 同 probe base S@100 |
|---|---:|---:|---:|---:|
| 純 L1 memory seed 0＋probe seed 0 | 0.4442 | 0.4469 | 61% | 65% |
| 純 L1 memory seed 0＋probe seed 1 | 0.4546 | 0.4566 | 65% | 69% |

兩者目前都未贏過同 probe 的 base。但這只占 18 個 memory/probe 組合中的
2 組，幾何監督 memory 尚無完整結果，不能用它們判定所有 memory 或幾何
監督的閉環效果。不能將未完成場景與已完成的完整 100 題平均混在一起。

## 執行與核對

- 執行中：34282_6（latent_only_m0_p2）、34282_7（latent_only_m1_p0）。
- 等待並行名額：34282_8–21；最多兩個 GPU 工作。
- 34283 在全部正式設定成功完成後自動進行完整核對。
- 本次已完成六組的 600 題、26,663 次選擇，從原始分數逐次重播；確認
  無重複 action、失敗限定 context、keep4／十次更新、成功停止、action／
  label／task 對齊與初始分數精確重現。原 base 重現歷史结果。
- 本機下載的 1,206 個已完成結果／trace／NPZ SHA256 全部與遠端一致，
  並另外重算 600 個逐題 AUCCESS。沒有修改正在執行的凍結程式或設定。

核對：`results/adapted_probe_closed_loop_20261001/PROGRESS_AUDIT_20261002.json`
與 `PROGRESS_LOCAL_VERIFICATION_20261002.json`。
獨立進度核對程式：`pipeline/audit_closed_loop_progress.py`。

下一步先讓既定 22 組完成，檢查幾何監督是否改善早期成功，以及第一次
memory 更新的收益；task-centric latent 已授權，但仍未啟動。
