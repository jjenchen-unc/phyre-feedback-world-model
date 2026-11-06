> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# 從頭訓練 memory：L1＋幾何監督

2026-10-05 UTC（本地 10-04）。使用者授權依既有「加入幾何監督改善 ranking」的結果重新訓練 memory。這輪建立新的起點，不載入 R0 或任何舊 memory 權重。先固定一套抽樣，其他 support/query 組合留待此基準完成後研究。

## 目標與資料角色

Base world model、原 probe 和評估用的三個 adapted probes 固定。新 memory 利用同場景已執行的失敗 action，修正其他 action 的預測。Support 提供 action、失敗標籤、真實 latent 與相對 base 的 residual；query 只提供初始 latent 與 action。Query 真實未來僅用於 loss／評估，不作記憶輸入。

每步均勻抽一個訓練場景，從其 support 區均勻抽一條真實失敗，再從 query 區均勻、不依成功標籤抽最多八個 action。這些 query 共用該條 support。Support 抽樣使用独立 RNG。K 固定 1；query 可以成功或失敗。

## 資料範圍

沿用已核對的 task/action 去重與群組分區，探索／support／query 目標比例 20/40/40。資料源是既有 cluster 與 s16r16 pack union。這輪只要求有失敗 support 和 query，**不讀取 N/H/F 作篩選或抽樣**，不使用 5／20 px 門檻決定訓練資格。

| 集合 | 場景 | 模板 | Query | 成功 query | 可用失敗 support |
|---|---:|---:|---:|---:|---:|
| 訓練 | 1,200 | 12，每個 100 題 | 29,349 | 12,702 | 16,916 |
| Memory 保留資料 | 400 | 4，每個 100 題 | 10,193 | 4,457 | 5,573 |

四個保留模板曾被 base／原 probe 看過，沒有提供新 memory 訓練梯度，不能稱為完整系統的未見模板。訓練與保留集合的 simulator 標籤、task 身分、support/query action 和群組隔離、來源 SHA 在準備階段重新核對。

資料來源 inventory 含歷史 Near 欄位，但新 manifest 只複製全部真實失敗、query 和 split；沒有要求場景具備 Near。也沒有因 query 缺少成功候選而排除場景。

## 模型與訓練

| 項目 | 固定設定 |
|---|---|
| 初始化 | 新建 MemoryPredictor，完全不載入舊 memory state |
| Base | 既有 cross-f1 PCA128 world model，固定 |
| Memory | residual，16 個記憶時間點、10×10 pooling、per-attempt encoder |
| 初始修正 | 各層 memory 輸出投影的 weight/bias 為 0；有 support 時也須精確等於 base |
| 空 support | 既有 empty gate；訓練後仍精確等於 base |
| Loss | 未來 31 幀 latent L1 ＋ 0.1 × 幾何 loss |
| 幾何 loss | 沿用原 probe 的動態物件 XY L1／256×10，加 bar 的 doubled-angle L1；每 query 固定八個未來監督時點 |
| Seeds | 0、1、2，各自重新初始化 memory，共享固定 base |
| 預算 | 每 seed 20,000 steps，每步最多八個 query |
| Optimizer | AdamW，max LR 3e-4、weight decay .05、clip 1；OneCycle 3% warmup |
| 保存 | 0、500、2,000、5,000、10,000、20,000 steps；另每 250 步保存 optimizer／RNG 供同一新訓練作業中斷恢復 |
| 最終比較 | 固定第 20,000 步，不在 cross 選最好 checkpoint |

學習率／schedule 沿用先前從零訓練的預算設定，沒有用續訓的 2,000 步假定已收斂。20,000 步也不構成收斂保證。與舊 R0 系列的資料、初始化及預算不同，因此新舊數值差不能唯一歸因於初始化或幾何 loss。本輪沒有新增純 L1 arm，不能單憑本輪重做一次 loss 因果主張。

## 評估與學習曲線

每個保存時點在固定、看效果前選定的小面板評估：訓練模板每模板兩題，共 24 題；memory 保留模板每模板兩題，共八題；cross 每模板兩題，共八題。每題至多 64 個 query，依固定 hash 選取，不依成功標籤補足。訓練場景的 query 是已見資料，只用來判斷是否學會，不當成泛化成績。

曲線同時保存原 head 與主要 adapted seed 0 的 base／memory 分數、成功／失敗 latent 誤差、位置／表面距離誤差和 ranking。每次從 NPZ 重算 AUC 與幾何／latent 指標。小面板分母、候選數與完整評估不同，不能直接把兩者均值接成同一曲線。Cross 曲線只供開發診斷，不選 checkpoint、不中途依表現停訓。

固定最後一步另做三個 seeds 的完整評估：

- 同一組 24 個已見訓練場景、保留模板每模板八題共 32 題，使用各題完整 query 區與固定隨機失敗 support。
- 全部既有 cross 100 題、每題 1,024 候選，固定 base＋adapted seed 0 選出第一次 action。若失敗，提供這條 support，對其餘 1,023 個 action 評分。若第一次成功，保留空 memory 檢查，與首次失敗收益分母分開。
- 全部新生成 within 100 題，待既有資料編碼／base cache 完成後使用相同自然首次失敗程序。這些題涵蓋新 memory 12 個訓練模板中的十個，沒有因 N/H/F 缺失排除任何題。

完整評估保留四個 heads；主要 head 仍固定 adapted seed 0。Base 必須使用相同 head／query；cross 的 base 分數、位置與距離逐元素重現既有 cache。原 probe 提供訓練梯度，adapted heads 固定且未適配新 memory；讀出差異仍是限制。

所有「先失敗再修正」收益另報有效場景數；AUC 沒有正負類時為未定義，不填 0.5。彙整按場景配對、先平均三 seed 的差值，再作模板分層的場景 bootstrap；區間條件於這些模板及已訓練模型。保留每 seed、每模板結果。

## 執行與邊界

Run：`results/memory_fresh_geometry_20261005/`。程式 `pipeline/fresh_memory_geometry.py`；執行用 run/code 凍結副本及 SHA256SUMS。Checkpoint 存在 run/models/seed_*/checkpoints/。Job receipts 見 run/jobs.json，當前狀態見 STATE.md。

工程 smoke 跑八個 updates，確認非空記憶初始 identity、空記憶 identity、零投影能取得梯度、第一次更新後 encoder 能取得梯度、幾何梯度確實回傳且不監督初始幀、base/probe 不更新、未來 query 幾何不影響 readout，以及保存指標可由原始陣列重算。Smoke 不以 ranking 好壞決定是否訓練。

三個訓練作業最多同時用兩張 GPU；訓練曲線在各自作業內完成。正式 cross 評估、獨立彙整接續成功的訓練。Within 評估另等待原 within 資料準備成功。原有實驗作業保留。

本輪是固定候選的 prediction／ranking 實驗，不是閉環 AUCCESS。Cross 是反覆使用的 D0；within 是未曝光的同模板新生成場景，不是官方固定 within fold。沒有宣稱成功、已收斂或已解決 support–query 適用性。

## 提交紀錄

| 作業 | Slurm ID |
|---|---:|
| 八步工程 smoke | 34453，已通過 |
| 完整候選／四 heads／抽樣核對 | 34454，已通過 |
| 三個 seeds 從頭訓練 | 34455，array 0–2，最多兩張 GPU；seed 0、1 已啟動，seed 2 排隊 |
| 最終 train／calibration／cross 評估 | 34456 |
| 原始陣列重算與彙整 | 34457 |
| 接入已註冊 within 新場景資料 | 34458，等待來源準備作業 34413 |
| 三 seeds 的 within 評估 | 34459，等待訓練與 within 接入 |
| Within 原始陣列重算與彙整 | 34460 |

最初 smoke 的三題各四 query 恰好都是單一類，因此 AUC 為未定義；它已驗證其他指標與身份／梯度檢查。獨立核對另選一個有正負 query 的完整 1,024 候選場景，只為覆蓋數值 AUC、四 heads、checkpoint 重新載入與歷史 base replay 的工程路徑，不用結果好壞決定訓練。

34454 已完成：60,000 個預定 episode 全部通過隔離與標籤檢查，各 seed 都涵蓋 1,200 題；載入 smoke checkpoint 後的完整候選評估核對八份條件陣列，四個 heads 精確重現歷史 base，AUC 另用排序／searchsorted 方法獨立重算。正式訓練已接續啟動；這些是工程證據，尚無新模型效果結論。
