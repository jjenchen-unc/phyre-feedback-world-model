> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Milestone memory results

Same Q0 start, legacy terminal-state loss, one seed; 100 scenes per panel, 9 attempts.

| Panel | Method | S@1 | S@3 | S@5 | S@9 | Gained/lost vs Q0 at 9 |
|---|---|---:|---:|---:|---:|---|
| cross | base | 2% | 9% | 11% | 17% | — |
| cross | q0 | 1% | 6% | 12% | 19% | — |
| cross | residual | 1% | 8% | 13% | 17% | 2/4 |
| cross | zero | 1% | 7% | 14% | 18% | 3/4 |
| cross | tta | 1% | 10% | 15% | 18% | 7/8 |
| within | base | 23% | 30% | 39% | 50% | — |
| within | q0 | 20% | 33% | 43% | 54% | — |
| within | residual | 20% | 39% | 43% | 48% | 1/7 |
| within | zero | 20% | 37% | 50% | 52% | 3/5 |
| within | tta | 20% | 34% | 43% | 53% | 9/10 |

All raw action choices, initial scores, success stopping and paired counts audited.

Full timing, first-update ranking diagnostics and task IDs are in SUMMARY.json. This compares methods on the existing shared shortlist; no universal or multi-seed claim.
