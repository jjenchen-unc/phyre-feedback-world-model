> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Probe 適配驗證：執行設定

2026-10-01。使用者已授權驗證「memory 預測的 latent 是否需要不同讀出」。
本輪只更新 probe，固定 cross fold1 base、原始 R0 memory 與 scorer；沒有新增
world-model/memory 訓練、simulator rollout 或閉環評估。

## 問題與可支持的結論

同一批凍結 latent，替換 probe 能否同時改善成功 action 的幾何誤差與 ranking？
若能，支持 readout 是可修復的限制之一；不能唯一證明 distribution shift，
也不能保證 memory latent 已具備正確物理語義。若不能，可能是資料量、適配
分布、模型能力或優化設定不足；不能判定 latent 已不可恢復地丟失資訊。

## 固定比較

所有 head 從同一原 probe 初始化，三個訓練 arms，各3個抽樣 seed（0/1/2）。
原 probe 不更新，另列為 reference。

| Arm | Probe 輸入 | Supervision |
|---|---|---|
| true_control | 真實 rollout 的 latent | 真實物件位置、bar方向 |
| predicted_mix | 同一批 query 的 base/R0 預測 latent，各50% | 同上 |
| initial_only | 只給相同 query 的初始 latent | 同上，仍以未來幾何為目標 |

每步128個 query/time/object 樣本、2000步、AdamW lr3e-5/weight decay.05，
cosine schedule、gradient clip1。三組使用完全相同的目標抽樣序列，保存並
核對 draw SHA256；mixed輸入每步嚴格平分base與R0。Loss沿用原probe：
XY L1/256×10，加bar的double-angle向量L1；不用成功標籤或ranking loss。
所有組使用最後一步checkpoint，不從cross或calibration挑最好者。

這是同query/time/object的真實latent續訓對照，不宣稱重現原probe全部訓練分布。
三個seed共享預訓練head/base/R0，量到的是probe適配抽樣的變異。

## 資料隔離與 context

從R0原有12個training templates中，每模板最多32個有base首次失敗context
的場景，共331場景、5289 query。每場景從事先分開的query pool隨機取最多16
條，不按query成功與否篩選；每query固定抽8個未來時間點（1..31）。先對場景
均勻抽樣，再抽該場景query/time/dynamic object。8點只用於適配快取，評估仍用
完整32點。三組共用樣本、真實幾何目標與初始物件描述。

Context為原資料中base在support半池遇到的第一個真實失敗；與query完全分開，
actual K固定1。這些是舊scorer、support半池的歷史trace，**不等同**穩定scorer
在1024候選上的部署分布。負結果須保留此適配分布限制。

Calibration使用R0原先保留的4個training templates中29個非空context場景，
共464 query；不參與適配、不選checkpoint，僅檢查適配是否轉移到保留場景。
Base與原probe曾看過這些模板，因此這是適配保留集，不是完整系統的新模板測試。

Cross直接重用階段A的100場景、每題1024候選、相同K1 context及query mask。
主要分析73個非空K1場景（57個有解可計算AUC），另報all100。4個cross模板
不出現在任何probe適配training/calibration模板中。這批cross已被反覆分析，
屬D0開發資料；本輪不得當作新的確認性測試。

## 評估與解釋

每個probe都讀base、R0、真實latent與initial-only四種輸入。保持原有batch32、
bf16 predictor、fp32 probe及NumPy幾何；最後穩定log scorer不變。
Geometry讀出只用初始物件描述；query未來僅作監督或明確標記的真實latent控制。

主要對比：同一R0 latent，mixed-probe相對原probe與true-control的AUC及成功
query表面距離MAE變化；報告每seed/每template。再看memory與base的差距是否
縮小，避免把所有predicted latent共同的readout改善說成memory特有問題。
Initial-only trained head與mixed head的initial-only推論消融並列：前者檢查
初始畫面捷徑，後者是輸入消融，可能有OOD影響，不單独作因果證明。原probe
沒有時間輸入，initial-only head也沒有新增時間輸入，會產生靜態幾何讀出。

Latent L1在所有probe間必須完全相同。原probe對cross的base/R0/true讀出必須
精確重現階段A的distance、XY、穩定score與L1，否則不發布科學結果。核對
空memory identity、future geometry輸入隔離、非零有限probe梯度、凍結模型hash、
逐query原始幾何→score→AUC與class L1/geometry平均。保存全部逐query幾何及分數。

## 執行順序與產物

凍結manifest/source → 工程smoke → 完整training cache → 9組probe →
100 cross＋29calibration場景評估 → 原始陣列核對與彙整。
Smoke使用事前固定2個training場景、每arm3步，以及00000/00014各第一個cross
場景；工程檢查不選方法。最多2個GPU工作同時執行。

Run：`results/probe_adaptation_cross_20261001/`；實作：
`pipeline/probe_adaptation.py`、`pipeline/summarize_probe_adaptation.py`。
Source/manifest/checkpoint/資料metadata hashes固定；快取另存SHA256。
Job ID與狀態將記錄在run的`jobs.json`及專案`STATE.md`。

## 已提交工作

Smoke34243已通過：2個cross場景、32組條件陣列核對完成，原probe精確重現、
未來幾何輸入隔離、各arm抽樣一致與非零有限梯度均通過。完整快取34244正在
執行；9組probe訓練34245、評估34246與彙整34247已設定afterok依賴，前一步
通過才會開始。這是工程檢查通過，尚無正式probe適配結果。

## 後續授權與進度更新（2026-10-01）

完整 cache 與九組 probe 訓練已完成，評估最後兩個分片仍在執行，正式彙整
34247 尚未完成。使用者授權此輪完成後繼續 loss／representation 實驗。
[幾何監督對照](MEMORY_GEOMETRY_FOLLOWUP_RUN.md) 已排入 afterok 接續鏈：
34265 smoke →34266 訓練 →34267 評估 →34268 彙整；不改本輪固定設定。

## 完成狀態（2026-10-01 更新）

本輪訓練、評估與原始陣列彙整皆完成且通過核對。前文提交／執行中狀態為
歷史紀錄；目前沒有本輪執行或排隊工作。結果與可支持的解釋：
[PROBE_GEOMETRY_RESULTS_20261001.md](PROBE_GEOMETRY_RESULTS_20261001.md)。
