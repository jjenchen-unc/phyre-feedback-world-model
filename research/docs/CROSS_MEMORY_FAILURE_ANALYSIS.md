> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Cross-template memory failure analysis

分析日期：2026-09-28。範圍：cross-template fold 1 的 100 個 dev tasks、4 個模板；既有結果與程式／checkpoint／資料 metadata 的唯讀分析。未重訓、未啟動新 rollout，也未重新執行模型推論。

最新的分層機制分析與假說限制見
[MEMORY_FAILURE_REANALYSIS.md](MEMORY_FAILURE_REANALYSIS.md)；下文為歷史
分析，不代表 R0–R4 重訓後已分離出因果原因。

後續：使用者授權的固定 context 推論診斷已完成，見
[CROSS_MEMORY_DIAGNOSTIC.md](CROSS_MEMORY_DIAGNOSTIC.md)。下文保留推論前的
證據與假說；§6 的兩題診斷已執行。新證據確認平均 latent loss 改善但成功
候選誤差與排名惡化，並量測到成功候選的預測終點被往失敗側修正。

## 結論與證據強度

已確認：memory 對 cross 的傷害集中於 00014；空記憶 adapter 的預設修正會造成退步，但把空記憶修正關閉，收到 feedback 後仍會退步。降低非空記憶的修正強度可以追回大部分損失。

最有根據的機制假說：在成功鄰域、成功／失敗混合 context 上學到的 latent 修正，對未見模板上的模型自行選擇的失敗序列泛化不足；沒有 task-ranking objective 或非空 context 的可靠性 gate 限制錯誤修正。這些結構與分布差異已確認，因果貢獻尚未透過匹配消融分離。

## 1. 從逐題資料重算

所有方法在相同 100 tasks 上比較，shortlist 每題 1024 actions，最多 100 attempts、10 次 updates，memory 保留最近 4 個失敗。84/100 tasks 的 shortlist 有解。

| 方法 | AUCCESS | S@100 | 00014 AUCCESS | 00014 解出數 /25 |
|---|---:|---:|---:|---:|
| base | 0.4541 | 62% | 0.3430 | 17 |
| memory no feedback | 0.3632 | 47% | 0.0659 | 4 |
| memory feedback | 0.3937 | 49% | 0.0868 | 5 |
| 同一 checkpoint，推論時 gate empty | 0.4073 | 49% | 0.1334 | 5 |
| 重訓 gate-empty checkpoint | 0.4065 | 50% | 0.1401 | 5 |
| gate checkpoint，scale 0.5 | 0.4293 | 60% | 0.2356 | 14 |
| gate checkpoint，scale 0.25 | 0.4502 | 60% | 0.3225 | 15 |
| TTA 20 steps | 0.5075 | 67% | 0.4384 | 19 |

資料：[results/cx_clm_*.json](../results)。Aggregate、唯一 task IDs 與每題 AUCCESS 公式已核對。base/nofb/fb/fb_gateinf/fb_gate/fb_gate_scale0.25/tta20 的本機與 oprime 檔案 SHA256 相同。

原 memory feedback 比 base 少解的 13 題中，12 題屬於 00014、1 題屬於 00000，沒有新增解出的題目。00014 的 AUCCESS 差值對全體貢獻 -0.0641，整體差值 -0.0604；其他模板的淨效果略為抵銷這個損失。

| 模板 | shortlist 有解 /25 | base 解出 | memory feedback 解出 | TTA 解出 |
|---|---:|---:|---:|---:|
| 00000 | 25 | 20 | 19 | 21 |
| 00002 | 25 | 25 | 25 | 25 |
| 00007 | 14 | 0 | 0 | 2 |
| 00014 | 20 | 17 | 5 | 19 |

00007 是所有方法的共同難點且存在 coverage 限制，不能拿它的零分解釋 memory 相對 base 的下降。00014 的 pool 有 20 題有解，base 可解 17 題，memory 的退步主要發生在已有可用解的候選排序。

例：00014:273 的 shortlist 有 13 個解；base 第 3 次成功，gate-empty feedback 版本到 100 次仍未成功。00014:407 有 20 個解；base 第 5 次成功，gate-empty feedback 同樣未成功。它們證明問題不是候選不存在，但沒有 attempt trace，無法指出第幾次更新開始排錯。

## 2. 空記憶的預設偏移是確定機制，但不是完整原因

[common.py](../pipeline/common.py) 的 `encode_memory` 永遠加入可訓練 null token；未啟用 `gate_empty` 時，空 context 仍執行 cross-attention adapter。只剩單一有效 null key 時，attention 的 value 對所有 query 一樣，形成每層共享的 learned additive offset；經過後續 nonlinear blocks 後仍可改變不同 candidates 的相對排序。

這解釋為什麼 no-feedback 不是 base：0.4541 → 0.3632。它是模型定義允許的行為，不是讀不到 feedback 的程式錯誤。

同一原 checkpoint 在推論時強制 gate empty（`fb_gateinf`）能把結果改善到 0.4073，屬較乾淨的空記憶消融；另行訓練的 gate checkpoint 為 0.4065。兩者仍低於 base，且 00014 仍只解 5/25，因此非空 context 的修正也有問題。

`gate_empty` 僅檢查 context 是否存在，並不是 uncertainty/relevance gate；context 非空時修正照常全量施加。

## 3. 訓練與閉環 context 的分布明顯不同

直接讀取 oprime checkpoint 與 pack metadata：

- checkpoint 使用 `cross_f1_train_clusters_pca128`、正確的 cross fold 1 base checkpoint，並非誤接 within-template base。
- 1599 個 training tasks、16 個 training templates、51161 個 rollouts，成功比例 42.4229%。四個 dev templates 與 train templates 不重疊。
- residual mode，16 個 memory time steps、10×10 spatial grid、per-attempt encoder；凍結 base，20K training steps、batch 8、kmax 4、p-empty 0.1。沒有覆蓋 base 權重的 checkpoint keys。
- [make_rollouts.py](../pipeline/make_rollouts.py) 的 cluster anchors 是成功 action，或最靠近成功 action 的失敗 action；鄰域半徑預設 0.08。這是 outcome-informed training sampling，不代表 test leakage。
- [train_memory.py](../pipeline/train_memory.py) 在同一 task 隨機抽不重疊 context/query，沒有模擬模型逐次選出失敗的閉環過程。
- [closed_loop_memory.py](../pipeline/closed_loop_memory.py) 成功立刻 break，只把失敗寫入 context；所以實際讀到的 context 全是失敗。

依實際 train pack 計算，先均勻抽 task，再不放回抽 K 個 context，其全失敗機率：

| K | Training 全失敗機率 | 閉環 context |
|---|---:|---:|
| 1 | 57.58% | 100% |
| 2 | 32.68% | 100% |
| 3 | 18.32% | 100% |
| 4 | 10.20% | 100% |

training 中不是完全沒見過全失敗；落差是其頻率、action 選擇機制、模板和幾何關係同時改變。部署失敗也不是均勻 random negatives，而是模型認為值得嘗試卻失敗的 actions。沒有保存完整選擇 trace，因此這次未量測 support/query 的 action 距離或 residual 相關性。

cross shortlist 全部 actions 的成功比例：00000 12.629%、00002 12.312%、00007 0.316%、00014 1.566%。這是候選分布，不能當成模型實際選擇 actions 的成功比例。

## 4. 訓練目標沒有保證 ranking 改善

`train_memory.py:72` 只最小化 query 的平均 latent L1；沒有成功排序、接觸條件或偏離 base 的懲罰，也沒有在未見模板的閉環 ranking 上選 checkpoint 的程式邏輯。原 checkpoint training L1 從首 1000 steps 的 0.09572 降到末 1000 steps 的 0.09173，只證明 training objective 降低，不能證明 cross prediction 或 decisions 改善。

adapter 的 nonempty correction scale=1→0.5→0.25 時，全體 AUCCESS=0.4065→0.4293→0.4502。這是同一 gate checkpoint 的縮放實驗，支持「完整修正在這組資料上傷害排序，減少介入可追回」；不能推出存在某個正 scale 一定超過 base，也不能把 hidden-state scale 當成最終 latent 輸出的線性混合。

00014 source (`oprime:/path/to/phyre-unc/VAORA/phyre/data/task_scripts/main/task00014.py`) 的目標是讓綠球接觸對側地板，中間有從上方垂下的垂直隔板。是否進入正確側、通過隔板下方等離散結果可能對接觸排序非常敏感。這是依任務幾何提出的假說；沒有 adapted trajectory/probe 誤差紀錄，不能宣稱已看到 memory 預測穿牆、平均化軌跡或錯誤接觸。

## 5. 能排除與尚未分離的因素

- 未見錯用 within checkpoint 或 backbone 被覆寫；本機／遠端三份核心程式 SHA256 一致。
- 原 memory feedback 相對自身 no-feedback：0.3632→0.3937，16 題更早、4 題更晚、80 題相同，沒有已解題變未解，新增解出 2 題。不能說 memory 完全不會利用 feedback；但不同選擇軌跡使此差值不是可精確拆解的因果比例。
- Cross probe 的 observed-latent diagnostic AUC=0.9801、同題 AUC=0.98，green median error=1.52px（[probe 結果](../results/probe_cross_f1_pca128.json)）。這不支持 scorer 對真實 cross 影片全面失效，但不能排除它讀 adapted latents 時的失配；diagnostic 候選集也不同於 shortlist。
- TTA 改善到 0.5075，00014 解出 19/25，說明 feedback 在此設定具有可利用價值；但 TTA 更新 backbone、累積最多 10 個失敗，而 memory 只保留 4 個、凍結 backbone。不能把差異全歸於 gradient adaptation 優於 memory architecture。
- 目前結果沒有每次更新的候選分數、實際 action ID、latent/geometry 誤差或 attention。無法區分「物理預測變壞」、「預測 L1 變好但 contact ranking 變壞」、「probe 對 adapted latent 不適用」的比例。

## 6. 最小下一個診斷（尚未執行）

固定 00014 的同一批 support failures 和同一批 query candidates，分別計算 base、empty、feedback、gate／scaled feedback 的：latent L1、green trajectory／接觸誤差、success rank，以及 predicted score 改變。先用 `00014:273`、`00014:407` 等已有明顯逆轉的題目，避免比較各方法不同的閉環軌跡。

同時比較正確 support、其他場景的 support、只保留 action＋failure outcome 的 support。這可以區分針對物理 residual 的有效轉移、單純先驗偏移和失敗標籤效應。確定失效路徑後，再考慮 failure-only／on-policy context training 或 decision-aware loss，而不是直接加大 memory。

以上分析使用 dev 結果提出假說，後續調參效果必須在未使用的新場景／fold 上驗證。
