> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Memory 失敗原因：重新分層分析

本分析使用既有程式、R0–R4 結果，以及兩個歷史 checkpoint 的固定
support 診斷，沒有新增訓練或模型推論。歷史模型的局部機制不能直接
套到新模型；R0–R4 的單 seed 比較也不能確立一般方法優劣。

目前最值得檢驗的問題是：**support 上觀察到的誤差，是否被轉移到
不適用的 query action？** 資料分布、目標函數與修正強度都可能影響
這件事，但尚未分離其因果貢獻。

## 已確認的失效位置

- 舊 ungated 模型在空 memory 也會改動預測；新 gated 模型已排除這條
  路徑，所有固定 query 的 base/empty 分數相同。不能再用空 memory
  偏移解釋新模型的退步。
- 兩個選定的舊模型 cross 案例，一個失敗 support 就足以損害剩餘
  query 的排序；所以多步閉環不是這兩題首次退步的必要條件。
- 真實 query latents 經相同 probe 在這兩題的排序 AUC 都是 1.0。
  這排除「真實影片本身完全讀不出成功」，不排除 predicted latent
  與 probe 的失配。
- R0–R4 在該 cross panel 都未超過 base；R0 很接近 base。大部分
  新方法之間的成對差異仍不確定，不能把它們當成同一種嚴重失敗。

## 假說一：support 誤差的適用範圍學錯了

模型需要估計的是 query 的誤差 `e(q) = z_true(q) − z_base(q)`。
觀察到 support 的 `e(s)`，並不意味著 `e(q)` 同方向或同大小。
Residual 是該次 action 的模型預測誤差，並非可直接共用的物理參數。

重新從保存的 arrays 計算歷史 gated 模型的終點修正，得到：

| 綠球終點 x 的修正需求／實際修正 | 00014:273 | 00014:407 |
|---|---:|---:|
| 第一個失敗 support：真實位置 − base 讀出位置 | +111.8 px | +170.5 px |
| 成功 query：平均真實位置 − base 讀出位置 | −128.8 px | −127.3 px |
| 成功 query：加入 memory 後的平均位置變化 | +31.3 px | +28.9 px |

兩題全部 13/13、20/20 個成功 query，其二維實際修正與所需修正的
內積都為負。這支持「對 support 可能合理的修正方向，在這些成功
actions 上不適用」的局部解釋；沒有證明模型真的複製了 support residual。
位置都是 probe 的讀出，不是解碼影片或新的物理模擬。

不能進一步稱為「所有 query 都收到相同偏移」：對全部未嘗試
candidates，平均修正向量僅占修正平方能量的 0.43%／23.26%，其餘
是隨 query 改變的部分。現有模型本來就有 scene/action-conditioned
attention。問題可能在於學到的條件關係錯誤，而非完全沒有條件化。

同模板錯場景與正確場景產生的 query 終點修正，整體 cosine 為
0.938／0.955。這只是對這組 donor 的敏感性證據；共享 query、相似
場景與讀出機制也可能造成相似，不能據此證明 memory 被忽略。

**如何辨別：** 在固定 query 上更換多個 support，依實際碰撞／接觸
結果與所需修正方向分組，測量修正是否有適當選擇性。Action 座標
距離本身不等於物理作用相似度。需要擴及未預先挑選的 cross 場景。

## 假說二：訓練沒有涵蓋部署時的 support–query 關係

不能只看 support 是否都是失敗。還要看它們如何被選到、與 query
有什麼關係，以及哪些作用結果在訓練裡被充分覆蓋。

R0–R4 訓練場景的 support pool 中位數只有 32 actions（範圍 8–32）；
實際 cross 從 1024 actions 排序。Policy traces 來自凍結 base／舊
memory，最多前四次嘗試，沒有用新 R 模型重新收集。部署則可更新
十次、保留最近四次失敗。候選池、選擇策略與後期 context 組合都不同。
較大的 pool 也可能讓策略選到訓練小 pool 未涵蓋的極端高分錯誤。

R1 對抽樣配方做了部分修正，但同時改變實際 K；它的負結果不能
排除分布失配，也不能證明 failure-only 訓練本身有害。

**如何辨別：** 固定實際 K、query 與 loss，比較 random-failure、
舊 policy 與目前模型的大候選池 traces；先量測支援／查詢的幾何
關係及誤差結構差異，再將分布差異和表現差異連起來。

## 假說三：被最佳化的改善，未必是決策需要的改善

舊模型的兩個案例直接顯示：平均 latent L1 下降，成功候選 L1、
probe 幾何誤差與排序同時惡化。這確立該處的預測目標與決策指標
不一致，尚未證明這是訓練失效的唯一或主要原因。

R2 的類別平衡只能調整成功／失敗的總權重；它沒有指定哪些時間、
空間位置或 latent 維度對決策重要。R4 加入的是固定 probe 的 soft
score 排序。數值一致與梯度存在，只驗證目標實作；並不驗證 scorer
在被最佳化後的 predicted latents 上仍可靠。

因此仍需區分 latent 預測本身不好、probe 在預測分布上失準，以及
score 與真正成功條件不一致。尚無證據說 R4 已發生 scorer exploitation，
也不能由單一權重／溫度的結果否定 ranking loss。

**如何辨別：** 同一批固定 queries 同時比較 latent 誤差、幾何／接觸
誤差、success score margin 與 first-success rank；檢查它們的變化
是否一致。訓練集、memory holdout、真正 cross 要分開呈現。

## 假說四：學習或泛化本身尚未成立

我們還沒有對新模型證明：在匹配的 training episodes 上，正確 memory
相對 base／錯誤 memory 有穩定利益。20K steps 與有限梯度不等於收斂，
也不等於學會有效利用 evidence。若訓練場景都沒有利益，應先查最佳化、
表示能力與目標；若只有已見場景有效，再把焦點轉向泛化。

Memory 編碼使用固定時間／空間位置、時間取樣與空間 pooling；可能
依賴模板外觀捷徑或丟失關鍵動態，但尚未量測。沒有證據支持直接
加大容量、斷言忘記物理規則，或把明確 retrieval 當成已證實的解法。

## 假說五：不合適的修正被過度施加，閉環再放大損失

舊 gated 模型將 scale 從 1 降到 .25，AUCCESS 從 .4065 回升到
.4502，但仍未超過 base .4541。這說明減少該模型介入可以減輕損失，
不能辨別「方向正確但太大」與「方向錯誤、縮小只是接近 base」。
兩個診斷案例的成功 query 修正方向有明確不合適的證據。

目前 gate 只處理空 context，沒有顯式學習非空 context 的可靠性。
但 attention／adapter 仍可能學會忽略訊息；不能說架構強迫非零修正。
新的 action 選擇又會改變後續 evidence，可能放大早期錯誤；完整逐次
分數與 context trace 尚未保存，還不能宣稱已觀察到自我強化循環。

**如何辨別：** 固定 context 的縮放曲線與每次更新的 score trace，
分別測試方向／幅度問題與閉環累積問題，不把兩者混為同一原因。

## 優先順序與結論

先確認新模型在 train／memory holdout／cross 的固定 episodes 上
各自能否利用正確 evidence；再測量 support 誤差到 query 誤差的
適用範圍；接著用實際 K 相同的資料消融，區分分布與目標問題。
Gate、retrieval、容量調整都是後續候選，不是本次已確立的處方。

最有根據的局部失敗描述是「修正對成功 actions 不適用，破壞原本
仍有用的相對排序」。它為何被學出、在多少模板上成立，以及新
R0–R4 是否沿同一條路徑失敗，仍需上述對照。

新增量測可由 `pipeline/reanalyze_memory_transfer.py` 重現，輸出為
`results/cross_memory_diagnostic_20260928/transfer_direction_reanalysis.json`。
統計／控制限制見 `MEMORY_REDESIGN_ANALYSIS_REVIEW.md`。未啟動新實驗。
