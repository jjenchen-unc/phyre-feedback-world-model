> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Memory redesign first-wave results

Complete: True

Single seed, memory-only internal holdout, reused cross-f1 development panel; not final test evidence.

| Variant | Training | Cross dev AUCCESS | S@10 | S@100 |
|---|---|---:|---:|---:|
| R0 | True | 0.442 | 0.46 | 0.61 |
| R1 | True | 0.4277 | 0.45 | 0.58 |
| R2 | True | 0.4201 | 0.41 | 0.6 |
| R3 | True | 0.4233 | 0.44 | 0.57 |
| R4 | True | 0.3962 | 0.42 | 0.51 |
