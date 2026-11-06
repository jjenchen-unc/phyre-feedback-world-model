> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Stable scorer closed-loop comparison

All traces audited; original base/R0 reproduced. Reused development set, fixed historical shortlist.

| Model | Legacy AUCCESS | Stable AUCCESS | Stable S@100 |
|---|---:|---:|---:|
| base | 0.4541 | 0.4541 | 62% |
| initial_R0 | 0.4420 | 0.4419 | 61% |
| A_s0 | 0.4127 | 0.4119 | 53% |
| B_s0 | 0.4042 | 0.4062 | 52% |
| A_s1 | 0.4166 | 0.4173 | 55% |
| B_s1 | 0.4099 | 0.4110 | 55% |
