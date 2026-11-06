> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Probe 適配之後：memory 的幾何監督對照

2026-10-01。使用者已授權 probe adaptation 完成後繼續後面的實驗。
這份設定在檢視完整 probe-adaptation 彙整與本輪結果之前固定。

## 要回答的問題

在相同資料、模型與訓練預算下，要求 memory 修正保留正確物件幾何，
是否比只降低 latent L1 更能改善 action 排序？

這是固定候選的離線歸因實驗，不是新 AUCCESS 或閉環成功率實驗。
也不是 TC-WM 的重現：沒有更換 latent space，沒有 proprioception 輸入。

## 唯一的訓練介入

| Arm | Loss |
|---|---|
| latent_only | 未來 31 個時間點的平均 latent L1 |
| latent_geometry | 同一 latent L1 ＋ 0.1 × 幾何 loss |

幾何 loss 沿用 probe 的位置與方向監督：XY L1 / 256 × 10，bar 加上
doubled-angle L1。每 query 使用既有固定 8 個未來取樣點，所有可移動物件
等權，再對 query 等權。這個 pilot 沒有直接監督 surface distance 或接觸
持續時間；它們是獨立評估指標，不能把位置 loss 稱為完整物理 supervision。

提供梯度的是固定原 probe。Base 與 probe 權重完全不更新，梯度經 probe
回傳到 memory。兩 arm 都計算幾何項，控制組係數為 0，避免額外 probe
前向路徑成為不同的實作。固定係數 0.1，不用 cross 結果調參。

每 arm 使用 seeds 0/1/2，都從相同 R0 初始化，各 2,000 步，每步同一場景
最多 8 個不重複 query。AdamW lr 3e-5、weight decay .05、cosine、clip 1。
比較相同最後一步 checkpoint，沒有 best-on-cross 選擇。這些是微調抽樣／
dropout seeds，不是三次獨立預訓練 base/R0。每 200 步保存可恢復訓練狀態。

## 資料與公平性

完整沿用 [probe 適配設定](PROBE_ADAPTATION_RUN.md) 的資料 IDs：331 訓練場景、
5,289 query，來自 R0 的 12 個 training templates。每步對場景均勻抽樣。
K1 是舊 support 半池中 base 的第一個真實失敗；support/query 不重疊。
同 seed 的兩 arm 抽樣序列必須有相同 SHA256，第一步兩項 loss 必須完全相等。
所有 pack metadata、checkpoint、source、manifest hashes 固定。

這個資料配方與先前 R0 原始訓練不同，因此要用本輪 latent_only 續訓對照
來歸因 loss，不能只拿 latent_geometry 對 R0。原始 R0 與 base 另作參考。
兩 arm 使用同一真實幾何資料；只有 treatment 使用其訓練梯度。

29 個 calibration 場景／464 query 不訓練，不選 checkpoint；其模板是
memory/適配保留模板，但 base 與原 probe 曾看過，不能當整個系統的新模板。
Cross 沿用 100 場景、同一 1,024 候選、K1 與共同 query mask；主要 73 個
非空 memory 場景，AUC 與成功誤差使用其中可定義的 57 個場景。
這批 cross 已影響研究方向，是重用 D0 開發面板，不是確認性測試。

## 評估怎麼避免只討好訓練用 head

每個新 memory checkpoint 都用四個固定 readout：原 probe，以及已完成
適配的 predicted_mix seeds 0/1/2。不依適配結果挑最好 head，不再訓練 head。
適配 heads 未接受本輪新 memory 的輸出訓練；它們是額外的固定讀出檢查，
不等於真正獨立的物理驗證，仍共享架構與原 probe 初始化。

逐 query 保存位置、表面距離、穩定 log score、latent L1/L2。報告 AUC、
hit@10、成功／全體幾何誤差、latent 誤差；按場景平均，再按模板和 seed 分列。
同 head 下比較幾何 arm 對 L1 arm、原 R0、base。不同 head 的 latent L1
必須相同，base 分數必須精確重現 probe 適配階段，未來 target 不得進入讀出。

## 執行順序與檢查

1. Probe adaptation 完整彙整成功並通過 5,160 組 raw-array 核對。
2. 工程 smoke：同 seed 兩 arm 各 3 步，兩個預先指定 cross 場景讀取全部
   四個 head；核對幾何梯度非零、base/probe 無梯度且權重未變、抽樣相同、
   空 memory identity、base 精確重現、所有 head 的未來資訊隔離，以及原始
   陣列重新計算的全部場景指標。
3. 六組正式訓練，最多兩個 GPU 工作同時執行。
4. 六組正式離線評估（各 100 cross＋29 calibration），最多兩個 GPU 工作。
5. CPU 重新計算全部原始陣列指標與配對效果，才發布正式彙整。

所有步驟使用 Slurm afterok；上游失敗時不執行下游。原 probe 適配的凍結
程式與輸出不修改。新的 run 是 `results/memory_geometry_followup_20261001/`。
程式 `pipeline/memory_geometry_followup.py`，實際執行使用 run/code 的凍結副本。
Job IDs 與狀態記錄於 run/jobs.json 及專案 STATE.md。

## 接下來的 task-centric latent

使用者已授權符合結果時繼續，不需要再次問許可；但架構實驗要在此輪檢查
完成後才具體定案，不能把未執行的設計當作已排隊。

先固定探索性接續門檻：cross 主面板與 calibration 各自至少 2/3 memory
seeds，在原 probe 下同時 AUC 上升、成功 query surface MAE 下降；且同 seed
在三個 adapted heads 的平均效果也要同方向。兩面板達標的 seed 不必相同。
這只是方向一致的研究排序規則，沒有最小效果量或顯著性保证。彙整仍會保留
各模板與每個 head 的效果，避免平均掩蓋退步。

達標：進一步設計物理 alignment＋embedding reconstruction 的 compact
representation，保留相同資料與相符的 base／memory 對照，先做小型 smoke。
更換 latent space 後，以共同幾何、ranking 與後續閉環指标比較；不可直接
比較不同空間的 latent loss 大小。

未達標：先分析是否只有訓練用 head 改善、是否幾何改善但 ranking 不變、
是否 calibration 能轉移而 cross 不能，再決定表示方法是否有足夠理由。
此時不自動擴大成昂貴架構掃描，也不能把單一係數／資料分布的負結果當成
物理監督無效。這個分支同样在授權範圍內，應接續分析並更新 STATE。

## 完成狀態（2026-10-01 更新）

本輪訓練、評估與原始陣列彙整皆完成且通過核對。前文提交／執行中狀態為
歷史紀錄；目前沒有本輪執行或排隊工作。結果與可支持的解釋：
[PROBE_GEOMETRY_RESULTS_20261001.md](PROBE_GEOMETRY_RESULTS_20261001.md)。
