> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Adapted-probe closed-loop results

22 fixed configurations on reused cross D0, cached simulator closed-loop replay, historical shortlist; no new training, rollout collection, or fresh holdout.

| Condition | AUCCESS | S@10 | S@100 |
|---|---:|---:|---:|
| base_original | 0.45413 | 47.0% | 62.0% |
| base_p0 | 0.44686 | 47.0% | 65.0% |
| base_p1 | 0.45663 | 46.0% | 69.0% |
| base_p2 | 0.45871 | 45.0% | 70.0% |
| latent_only_m0_p0 | 0.44420 | 48.0% | 61.0% |
| latent_only_m0_p1 | 0.45455 | 48.0% | 65.0% |
| latent_only_m0_p2 | 0.45724 | 47.0% | 64.0% |
| latent_only_m1_p0 | 0.44260 | 47.0% | 60.0% |
| latent_only_m1_p1 | 0.45507 | 48.0% | 65.0% |
| latent_only_m1_p2 | 0.45730 | 47.0% | 61.0% |
| latent_only_m2_p0 | 0.44597 | 48.0% | 61.0% |
| latent_only_m2_p1 | 0.45904 | 49.0% | 65.0% |
| latent_only_m2_p2 | 0.45822 | 48.0% | 64.0% |
| latent_geometry_m0_p0 | 0.45248 | 47.0% | 63.0% |
| latent_geometry_m0_p1 | 0.46537 | 48.0% | 65.0% |
| latent_geometry_m0_p2 | 0.46529 | 48.0% | 62.0% |
| latent_geometry_m1_p0 | 0.45526 | 46.0% | 63.0% |
| latent_geometry_m1_p1 | 0.46527 | 47.0% | 66.0% |
| latent_geometry_m1_p2 | 0.47057 | 47.0% | 64.0% |
| latent_geometry_m2_p0 | 0.45476 | 46.0% | 62.0% |
| latent_geometry_m2_p1 | 0.46478 | 47.0% | 65.0% |
| latent_geometry_m2_p2 | 0.46640 | 47.0% | 64.0% |

All 22 conditions retained. Nine memory/probe combinations share pretrained models; not nine independent full-training seeds.
First-update diagnostics exclude each head-specific first failed action; denominator may differ across probe seeds.
Cached simulator trajectories are revealed only after failed attempts. No new rollout collection or new holdout test.
