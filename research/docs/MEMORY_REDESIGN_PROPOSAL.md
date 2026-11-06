> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Memory 方法與訓練資料重設計提案

2026-09-28。狀態：設計建議，尚未實作或啟動新訓練。依據
[固定 context 診斷](CROSS_MEMORY_DIAGNOSTIC.md)；兩題已知 dev 失敗不能視為
所有 cross templates 的成因證明。研究重心仍是
[attempt 資訊的提取、轉移與效益](INFORMATION_FRAMING.md)。

## 1. 要改變的學習問題

教模型「這次失敗對哪些未試 action 有用」，而非把場景中所有 actions
往相同失敗方向修正。需要同一 context 下，成功與失敗 queries 都有足夠
監督，以及同一固定 query pool 下不同 context 的比較。

已確認的是平均 L1 與成功候選品質背離、空 context 造成偏移、非空修正
仍傷害排序。Failure-only distribution shift、學到 template prior、缺乏
relevance 都是待驗證機制，不能先假設某個新 gate 已能解決。

## 2. 先分開四個變因

1. Context 來源：random mixed outcomes vs policy-selected failures。
2. Query 分布與權重：自然／原本抽樣 vs 成功、hard failures、邊界、全域。
3. Objective：原始 latent L1 vs class-balanced prediction vs decision loss。
4. Architecture：既有 gated bank vs 顯式 candidate-dependent correction。

避免一次替換四者而無法歸因。保留 frozen base、encoder、probe、candidate
pool 與資料量／訓練步數記錄，先測資料和 objective 的作用。

## 3. Context 配方（建議起點，非經驗最優比例）

主閉環成功即停止，因此主要 training supports 應全為失敗；成功影片
仍是必要的 query targets，不應從整份訓練資料刪除。

| Episodes 比例 | Context 來源 | 目的 |
|---|---|---|
| 60% | 實際策略的失敗前綴，先一半 frozen base、一半當前 memory | 覆蓋模型自己選錯的分布 |
| 30% | 同場景較多樣的失敗；兼含局部相近與跨 interaction 的組合 | 避免只學某個 policy 的單一路徑 |
| 10% | 空 context | 檢查部署起點，保持 base identity |

K=1/2/4 起步；在每個 K 分別比較一個 interaction 的重複證據與不同
interaction 的互補證據，不預設後者一定較好。Empty gate 若為 hard
identity，空 context 不會產生 adapter 的 prediction gradient，不能
宣稱它因此學到更多；可以少量保留作不變量檢查，避免浪費大比例更新。

Policy trace 必須先依 score 選 action，再揭露其真實結果；成功就停止。
不能先排除成功候選再選「policy failure」。離線從已知失敗做分層抽樣
可以用於 diversity 組，但須與實際 policy 前綴區分。原始 attempt 均記帳。

先以現有 checkpoint 收集，再在一輪訓練後以新 policy 補一輪；保留部分
舊 traces，避免只追逐最近 policy。這借用 policy-induced distribution
的資料聚合動機，不是聲稱此演算法等同 DAgger 或享有其理論保證。
[Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)

## 4. Query 組合：context 是失敗，不代表 query 也應全失敗

建議每個 scene/context 先選 8 個 disjoint queries，分成：

| Query 數量 | 類型 | 要學的區別 |
|---|---|---|
| 2 | 多樣的成功 actions | 保護稀少的解，避免將失敗修正套到所有 actions |
| 2 | Frozen base 高分但實際失敗 | 修正會誤導規劃的 false positives |
| 2 | 成功鄰域的失敗 actions | 小動作差異可能改變 outcome，鄰近不等於可直接複製修正 |
| 2 | 從其餘合法 candidates 均勻抽取 | 保留全域覆蓋，避免僅學成功邊界 |

最後一組可能含成功，故此表不是 1:1 class balance；loss 另以 class
normalization 防止失敗例支配。此配方是訓練抽樣，不是評估分布。
同一 scene 無成功 action 時保留 negative-only episode，不虛構正例、
不丟棄難題，省略該 episode 的 pairwise ranking term。類別不足時記錄
fallback，從其餘合法、不重複的 query 補足。

對 query 的 outcome-informed 抽樣只發生在 training supervisor；query
真實影片／標籤不得流入 context 或部署的 relevance gate。Q 與支援候選
reservoir 事先分開，同一批 Q 在 C、C+A、C+B、C+A+B 比較中保持不變。
初始畫面、template/task ID 捷徑和重複 action IDs 都要檢查。

Interaction 類型（碰撞、支撐解除、目標接觸、無有效互動）可由 simulator
作離線分層標籤；除非新增並評估可部署提取器，不能把 query 真實 contact
graph 當成模型輸入。大 residual 不一定是最有價值的 evidence。

## 5. Objective：分開 fidelity、decision 和 adaptation utility

第一個 loss 改動：

`L_bal = 0.5 * mean(L_pred on successes) + 0.5 * mean(L_pred on failures)`

單一 class 的 episode 用其有效類別，不以缺失類別除零。先以此確認平均
L1 對稀少成功例的遮蔽是否可改善；不一開始引入一堆 loss。

下一步才增加固定 scorer 的成功／失敗 pairwise ranking：

`L_rank = mean softplus(-(s_success - s_failure) / temperature)`

兩個同樣失敗的 action 沒有已知真實先後，不能任意賦予 success ranking。
Scorer 保持凍結，但梯度可透過可微 readout 回到 predictor；必須保留
latent fidelity 與獨立幾何／readout 評估，以防只迎合 scorer。
現有 `rule_score(surface_dist(...))` 是 NumPy pipeline，不可直接假設
它能反傳。Implementation 需提供 numerically checked Torch 等價的 soft
scorer；hard longest-contact 規則不作梯度路徑。

以 `L_bal + lambda_rank * L_rank` 起步，在隔離 validation 上選係數。
必要時才測 correction-size regularization 或相對 base 的 utility target；
不能強迫每個 query 都必須比 base 好，也不能把 gate 當成保證不退步。
Decision-aware model loss 的動機與
[Farahmand et al., 2017](https://proceedings.mlr.press/v54/farahmand17a.html)
相關；這裡的 PHYRE pairwise loss 是我們的設計，未獲該文結果直接驗證。

## 6. Architecture：先保留 bank，再讓修正有明確適用範圍

第一階段沿用已存在的 query-conditioned attention memory，加上已驗證的
hard empty gate。現有架構本來就有 candidate queries，不能把「增加
cross-attention」當成新解法。

若資料／loss 的改動仍不足，再測：

`Z_hat(q,C) = Z_base(q) + g(q,C) * Delta(q,C)`

- Empty context 時修正嚴格為 0；no-op value 固定為 0，避免 learned
  null token 本身攜帶無經驗的偏移。
- 每個 query 自己決定修正強度，而不是整個 scene 共用固定 scale。
- Relevance input 可包含 action 差異、初始幾何、base 預測的 motion／
  interaction features 與 support 的觀測；禁止 query 真實未來資訊。
- 不把 action 距離當硬 gate：近 actions 可以跨越成功邊界，遠 actions
  也可能共享互動機制。幾何／機制訊息是待驗證的輔助表示。
- Gate 從 disjoint queries 的 adaptation utility 學習，可用離線 base
  與 corrected errors 作訓練 target、stop-gradient 與獨立校準；不能用
  test query label 決定開關。需監測全部關閉的退化解。
- 若改成最終 latent residual，須另行訓練／驗證，不等同現有 hidden-layer
  adapter 的 `mem_scale`。

Query-specific context reading 有
[Attentive Neural Processes](https://arxiv.org/abs/1901.05761) 的一般依據；
該文不保證本任務的 relevance 或 gate 正確。先不加入 recurrent 壓縮，
避免把 evidence 提取與資訊壓縮的失敗混在一起。

## 7. Representation 與 residual 的對照

在相同 episodes、容量與計算預算下，分別訓練 action/outcome、加 observed
video、再加 base residual 三種輸入。原本的 input-zero intervention
不足以判斷 trained action-only baseline；不把對 OOD 零輸入的表現當作
資訊價值結論。

同一 query/context 改變 evidence 的 controls 應包括 wrong scene、
action–video association shuffle、duplicate evidence、additional distinct
failure。分別報 prediction utility 與 decision utility；正確 evidence
若沒有穩定優於錯配 evidence，不能主張已學到可轉移的物理資訊。

## 8. Cross 驗證不能只用 memory 的同模板 holdout

Base 對自身 training scenes 的 residual 可能與 OOD residual 不同；
本次沒有量測這個差異，屬需要防範的 meta-training mismatch。

先在原 training templates 內建立 pseudo-cross validation，held-out
templates 不參與該內部 base 或 memory 的訓練。若只對 memory 留模板，
而 base 已看過它們，必須稱為 memory-only holdout，不能說已模擬完整 cross。
資源允許才作多折 cross-fitting，以 out-of-fold base prediction 建立
memory training episodes；須處理不同 base checkpoint 的介面與誤差分布。

00014:273/407 僅留作固定 failure regression cases，不能繼續作主要選模
依據。現有 dev 已被診斷與挑選，最終結果需在凍結方法後使用尚未用過的
test scenes/templates 或預留 fold。報告範圍與調參使用情況。

## 9. 最小實驗順序

所有新 variants 先共用 gated bank，固定 base、probe、pool 和計算預算。

| Run | Context | Query sampler / loss | 回答 |
|---|---|---|---|
| R0 | 原 mixed | 原 sampler / L1 | Gated reference |
| R1 | Failure-context 配方 | 與 R0 相同 | Context 分布單獨有沒有用 |
| R2 | 與 R0 相同 | 新 query 配方 / balanced prediction | Query 覆蓋＋重權能否保護成功例 |
| R3 | Failure-context 配方 | 與 R2 相同 | 兩種資料改動是否互補 |
| R4 | 與 R3 相同 | 再加 ranking loss | Decision supervision 的額外效果 |

R2 一起改 query sampler 與 class weighting，是有意視為「query coverage
package」；如果有效而需論文歸因，再補 sampler-only / weighting-only。
顯式 gate、輸入表示對照與 recurrent 不加入這第一輪，避免擴成大型
全因子實驗。配方比例為事前起點，先固定一輪，不在 dev 上大量 sweep。

Fixed-Q 指標：class-conditional L1、物體／接觸誤差、同題 AUC、first
success rank、相對 base 的正／負 transfer 分布、correct vs shuffled
evidence utility。Full closed-loop 才報 AUCCESS、S@K、實際互動／耗時。
最終候選需要多個 training seeds 與多個未見 templates 的配對比較。
TTA 對照使用相同失敗 observations 和 keep budget，避免先前 memory
keep4 與 TTA 保留最多10筆的混淆。

若新增 supervision 只提升 ranking、没有改善可驗證的幾何 prediction，
應如實定位成 decision-useful representation adaptation。若 action/outcome
baseline 同樣好，則沒有證據支持影片或 residual 是必要的。
