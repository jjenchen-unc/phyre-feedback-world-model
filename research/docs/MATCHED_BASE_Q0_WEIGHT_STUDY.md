> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Base 與 Q0：相同 loss 的從頭訓練比較

2026-10-14 更新：24 組訓練與評估全部完成，配對與結果核對通過。參見[完整結果與組會紀錄](MATCHED_BASE_Q0_RESULTS_20261014.md)。

使用者於 2026-10-12 指定：base 與 Q0 都從隨機 predictor 權重開始，先讓兩者的 loss 一致，再比較幾何 loss 權重對 performance 的影響。這不是在舊 predictor 上增加訓練，也不是訓練 memory updater。

## 要回答的問題

1. 相同資料、初始 backbone、loss、訓練步數下，加入可學習 initial tokens 是否改善 ranking？
2. 幾何監督的權重會如何影響 latent 誤差、幾何誤差和成功率？
3. Base 與 Q0 對幾何監督權重的反應是否不同？

## 配對設定

| 項目 | Base | Q0 |
|---|---|---|
| Predictor 初始化 | 隨機 backbone | 相同 seed 的相同隨機 backbone，另加隨機 tokens |
| 架構 | 原 predictor，沒有 prefix tokens | 原 predictor self-attention 加入 32 個 prefix tokens |
| 訓練參數 | 全部 predictor 參數 | 全部 predictor 參數與 initial tokens |
| Loss | latent L1 + λ × geometry | 完全相同的函數 |
| 評估 | 固定 predictor 排序 | 固定 predictor＋initial tokens 排序，不更新 tokens |

Encoder、PCA、geometry probe 與評估 scorer 沿用並凍結。Predictor 的 output projection 依原初始化方式設為零；因此第一步的 token gradient 為零是預期行為，output projection 開始學習後才有 token gradient。對照不載入原 pretrained predictor 的 state_dict，只讀取原設定的維度；每個 seed 的 base/Q0 backbone 初始化 SHA 必須相同。

幾何 loss 沿用 Q0 的原始實作：每條 query 固定取 8 個未來時間點；經凍結 probe 讀取動態物體位置與長條的 doubled-angle 表示，再對照真實幾何。先對每個 query 內的動態物體／時間平均，再對 queries 平均。兩個模型的時間取樣、normalization 和權重完全一致。不是早期 train_predictor.py 中僅有 XY 的 auxiliary loss。

## 實驗組合與預算

- 官方 cross fold1、within fold0。
- λ = 0、0.03、0.1、0.3。
- λ = 0.1：base 與 Q0 各 3 seeds。
- 其餘三個 λ：base 與 Q0 各先跑 seed0，作為權重趨勢探索。
- 總計 24 次訓練，原先最多 4 張 GPU，2026-10-14 經使用者同意提高為 6 張；各 lane 先跑 λ0.1，再跑其他權重。
- 每次使用完整訓練資料、8 epochs、batch8、AdamW、lr3e-4、weight decay0.05、gradient clip1、OneCycleLR 3% warmup。
- Cross：52,276 條軌跡，52,272 次 optimizer updates，418,176 次 trajectory exposure。
- Within：52,349 條軌跡，52,344 次 optimizer updates，418,752 次 trajectory exposure。
- 同一 seed 的所有模型／權重共用相同資料順序，固定 epoch draw SHA。每 epoch 捨去不足一個 batch 的尾端。
- 使用固定最終 checkpoint，不依評估結果選最佳 epoch。

## 評估與驗證

同 split 沿用已驗證的官方100個場景及相同1,024個候選 actions、實際模擬成功 labels，scorer 不變（cross 為 predicted_mix_s0，within 為原 within probe）。主要 S@9，另報 S@1/3/5/20 與可定義場景的 ranking AUC。因模型不在線更新，一次固定排序即可得到各嘗試預算的結果。

每個 epoch 另對固定400條 dev 軌跡量測 latent L1 與相同定義的 geometry loss，所有模型共用相同 trajectories 與時間取樣。這400條不是正式ranking候選池，因此只用來觀察prediction誤差趨勢，不把它當作逐action的ranking機制證明。訓練log會同時顯示原始L1、geometry，以及 λ×geometry/L1 的比例。

正式訓練前做 GPU smoke：配對backbone初始化、scratch零輸出投影、相同初始loss、Q0 loss精確重現舊函數、probe固定、8次實際optimizer updates，以及 Q0 移除 activation checkpointing 前後的loss與更新後權重精確一致。通過後才用不重算的版本正式訓練。沿用原本的可微分geometry loss與評估scorer；本輪未套用新加速 scorer；其後續驗證紀錄見 [updater 加速驗證](UPDATER_ACCELERATION_VALIDATION.md)。

λ0.1有三seeds；其他λ只有一seed，不能據此宣稱普遍最佳權重。三seeds共用100題，不能當作300個獨立場景。

## 產物

- 程式：`pipeline/matched_base_q0.py`；正式執行使用結果目錄內凍結的副本。
- 結果：`results/matched_base_q0_20261012/`。
- `PLAN.json`：初始化方式、資料、共同loss、權重、seeds、步數與來源SHA。
- `SMOKE_{cross,within}.json`：工程驗證。
- `models/{panel}/{base,q0}_g{weight}_s{seed}/`：初始狀態、進度、各epoch誤差、可恢復checkpoint與完成紀錄。
- `evaluation/{panel}/{name}/`：逐場景scores、成功排名、AUC及完整性檢查。
- `SUMMARY.json`：24個訓練及評估全部完成後才產生。

GPU smoke 35024／35025 均通過。正式工作35026–35049及彙總35050均已完成；最後一組評估於2026-10-14 07:55 UTC完成。SUMMARY.json 的 complete 與 paired_training_verified 均為 true。原始排程見 jobs.json，後續六張 GPU 的調整見下節。


## 2026-10-14 排程調整

使用者同意將同時使用的 GPU 上限由 4 張提高至 6 張。提前釋放兩個獨立的
Q0 λ＝0.3／seed0 工作：cross 35037、within 35049，解除它們對 λ＝0.03
工作完成的等待。模型、資料、loss、訓練步數與凍結程式均未變更。
彙整工作 35050 改為明確等待尚未完成的全部六組
35031／35036／35037／35043／35048／35049 成功，避免解除序列後漏等。
排程變更紀錄：`results/matched_base_q0_20261012/schedule_expansion_20261014.json`。
實際啟動仍由 Slurm 資源與優先序決定；最新狀態見 STATE.md。


## 實測耗時更新

首批 λ0.1／seed0 的 cross／within、base／Q0 四組訓練與評估均已完成。Base 完整8epochs約3.34–3.36小時，Q0約3.61小時；每組100題固定排序評估約15–17分鐘。這是首批實測；後續執行速度變慢，原先全程23–25小時與剩餘17–19小時的估計已失效，不能作為目前完成時間。

若後續固定λ0.1、兩個protocol各補兩個fold、每個fold兩模型各3seeds，需要額外24組，資料備妥後約23–26小時。新fold訓練packs目前尚未備妥，以上不含資料收集／編碼及排隊。此擴展僅估時，尚未提交。若選其他λ，現有fold還需補seed1/2；不能把單seed的sweep視為已有三seeds。
