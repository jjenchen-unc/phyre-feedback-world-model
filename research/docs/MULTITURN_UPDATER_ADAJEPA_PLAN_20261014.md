> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# 多輪 memory 更新與 AdaJEPA 比較設計

2026-10-14。設計已整理，尚未提交本輪訓練。使用官方 test 場景評估；Cross 與 Within 各固定一組 initial configuration，再跨 fold 比較更新方法。

## 這輪要回答的問題

1. 從同一個 Q0 出發，學到的 updater 是否比不更新更好？
2. 收益來自 residual 的觀測資訊，還是 action 與失敗歷史就足夠？
3. 學到的 token 更新，是否比直接以觀測誤差做 test-time gradient update 更有效率或更準？
4. 更新一次與持續多輪更新的差距是什麼？結果在不同 fold 是否同方向？

## 各 protocol 的起始設定

| Protocol | Initial configuration | 既有 checkpoint | 選擇依據 |
|---|---|---|---|
| Cross | Q0，初始訓練 λ＝0，seed 0 | Cross fold 1 的 `q0_g0_s0` | 既有 Q0 權重比較中，S@9＝19%、AUC＝0.7245 最高 |
| Within | Q0，初始訓練 λ＝0.03，seed 0 | Within fold 0 的 `q0_g0.03_s0` | 既有 Q0 權重比較中，S@9＝54%、S@20＝62%、AUC＝0.8851 最高 |

這些依據來自先前 seed 0 與既有候選池，代表選擇起點的理由，並非新 fold 的保證。初始 λ 固定，不在每個 fold 各自挑最好值。

同時保留相同 fold、seed、λ、訓練資料與 8 epochs 的無 tokens base。既有 Cross fold 1 λ＝0 與 Within fold 0 λ＝0.03 的 base／Q0 都可直接重用。

## Fold 與 seed 範圍

第一輪 Cross 用 fold 1、0、2，Within 用 fold 0、1、2，共六個 panel。先各一個 initial seed 與 updater seed，固定為 0，優先檢查不同 fold 的方向，而不是先在同一個 fold 擴大 seeds。

先做已有資產的 Cross fold 1 和 Within fold 0，通過工程檢查後補其餘四個 panel。新 fold 必須以它自己的訓練資料重新訓練 base／Q0 與 updater。不能把 fold 1 的 checkpoint 直接用在 fold 0 test，便稱為 fold 0 訓練實驗。

截至本次資料盤點，完整初始訓練 packs 主要只有 Cross fold 1、Within fold 0；其餘 fold 的資料、初始模型與 support/query manifest 需要準備。Encoder／PCA／probe／scorer 的資料來源也需逐項記錄。新 fold 的 probe／scorer 只用對應 fold 的訓練資料建置，不直接挪用可能見過其 test tasks 的 head。同一 panel 全部方法共享且固定同一個 scorer；新 fold 的訓練配方須在啟動前寫入 manifest。

單 seed 的跨 fold 結果先作方向驗證。後續如補 seed 1、2，應成對補齊所有主比較方法，不只替看起來較好的方法增加 seeds。

## 六個主要比較方法

| 方法 | 起始模型 | Offline 新訓練 | 每次失敗後如何更新 |
|---|---|---|---|
| Base 固定 | 配對 base | 無 | 不更新，只排除已嘗試 action |
| Q0 固定 | 選定 Q0 | 無 | 不更新，只排除已嘗試 action |
| Q0＋residual updater | 同一 Q0 | 訓練 updater | 以當下 residual 更新 tokens |
| Q0＋zero updater | 同一 Q0 | 獨立訓練 updater，residual 固定為零 | 使用 action、失敗／更新歷史更新 tokens |
| Q0＋AdaJEPA-style | 同一 Q0 | 無額外 updater 訓練 | 用已觀察軌跡更新 predictor 部分權重 |
| Base＋AdaJEPA-style | 配對 base | 無額外 updater 訓練 | 同樣做 predictor 的 test-time gradient update |

Q0＋AdaJEPA-style 是核心對照，讓兩種更新方法有相同起始預測與第一個 action。Base＋AdaJEPA-style 另回答「不使用 memory tokens，單靠 test-time adaptation 是否已經足夠」。

這裡的「without residual」明確指 zero residual，不是改成輸入 raw observation。Zero updater 仍接收 action，並透過連續 state 保留失敗歷史；不能把它等同於沒有任何資訊或沒有更新。

Real／zero 兩個 updater 從相同初始 updater 權重、相同 Q0、相同 episode／query 抽樣開始；因各自使用當下 scorer 選 support，訓練後的 support 路徑可以不同。這是完整方法比較，不是只移除一個輸入的純因果實驗。

## Updater 的訓練設定

凍結 predictor 和 initial tokens，只訓練 updater。這樣在沒有任何 support 時，兩個 updater 分支都必須完全等於自己的 Q0，排除額外訓練改變初始模型的影響。

| 項目 | 固定設定 |
|---|---|
| Offline loss | Query latent L1 ＋ 0.03 × geometry loss |
| λ 的含義 | 這是 updater 的 λ，獨立於 initial predictor 的 λ |
| 預算 | 10,000 個訓練 slots，每 slot 2 個 episodes，共 20,000 episodes |
| Query batch | 每個 episode 最多 8 個未作為 support 的 query actions |
| 最大更新數 | 每個 episode 均勻抽取 1、2、4、8、19 之一 |
| Loss 施加位置 | 每次實際更新後都計算同一組 query loss；先在 episode 內平均，再平均兩個 episodes |
| Optimizer | AdamW，lr 3e-4，weight decay 0.05，clip 1，OneCycle 3% warmup |
| 無 support episode | 保留紀錄，不強迫選失敗 action；無 updater gradient，不重抽來湊數 |
| 反向傳播 | 完整跨輪 BPTT；action argmax 不反傳 |

Updater λ＝0.03 是這輪刻意固定的起始假設：保留較小的幾何監督，避免直接沿用先前 0.1。它不是由本輪 updater 結果選出的最佳值。Real 和 zero 兩分支使用完全相同的 loss。

每個 training episode 的流程如下：

1. 以固定 manifest 抽一個訓練場景，從它的 query 分組取最多 8 個 actions。Support 候選與 query 依 action／生成群組分開，完全相同 action 不可同時出現在兩邊；不使用 Near 成功門檻篩選。
2. Predictor 以當下 tokens 預測 support 候選，由部署時同一個 scorer 排名，選出尚未嘗試的最高分 action。候選池包含自然出現的成功與失敗 actions，不強迫失敗，也不讀取未選 action 的結果。
3. 若成功就停止；若失敗且未用完 update cap，讀取該 action 的觀測 latent。
4. Real residual＝觀測 latent－包含當下 tokens 的預測；zero 分支把這個張量設為零。更新 tokens 後，計算 query loss，重新排名後再選下一個 action。
5. 達到 cap、成功或候選用完時結束。保留實際更新數的分布；例如抽到 19 並不代表一定執行 19 次。

更新到 19 次是為了讓 20 次嘗試評估具有對應的訓練深度，避免只訓練到 8 次卻把第 9–19 次更新當成同分布結果。這也增加計算量，因此需先量測真實 episode throughput。若深度 19 的自然覆蓋很低，先回報覆蓋不足，不能默默改成只挑失敗 actions。

訓練候選數可能少於測試的 1,024 個，必須記錄每個 fold 的候選數分布與被選 support 排名，不能宣稱使用相同 scorer 就已完全消除 train／deploy distribution 差異。

## AdaJEPA 的比較方式

延用現有 predictor-only PHYRE 移植，名稱標為 **AdaJEPA-style**。[參考官方 pinned 程式](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/planning/adajepa.py)；本實驗不宣稱完整重現原論文的 encoder adaptation 或 MPC benchmark。

本輪設定如下：更新最後一個 predictor Transformer block 與 final LayerNorm，固定 encoder、PCA、scorer、其他 predictor layers、輸出 linear；Q0 分支另外固定 initial tokens，且不使用 learned updater。每次失敗後 replay 最近 5 個已觀察 trials，以未來 latent L1 更新權重。同場景內權重累積，每次 adaptation call 重建 Adam；換場景時還原權重並清空 replay。

每個 trial 都從自己的初始場景與 action 預測，不把 reset 之間的軌跡拼成一段影片。更新只使用已執行 trial 的觀測，不能用未嘗試 query 的真實未來。

因起始 checkpoint 已改變，不直接假設舊 AdaJEPA learning rate 最佳。先在 anchor panels 各固定 25 個 test 場景比較 lr＝3e-5／3e-4／1e-3、每輪 1／3 steps；base 與 Q0 各有相同的六組搜尋預算。依 S@9、S@20、較少 steps、較小 lr 的順序選定，再於同 protocol 其餘 folds 固定使用。每個搜尋結果保留，不在每個新 fold 重新挑最好設定。

Learned updater 的 offline query loss 含幾何監督，AdaJEPA-style 使用 online observed-trial latent L1；兩者也有不同的 offline 訓練成本。因此方法間結果回答的是整套策略的效果，不單獨證明「tokens 更新比 gradient update 更好」。需同時報告 offline 訓練時間、online 更新時間與成功率。

## Test 評估規則

每個 fold 固定 100 個官方 test 場景，按模板平衡、以固定 hash 選擇，不依成功率挑題。最多嘗試 20 個不同 actions，成功就停止。所有 support trials 都算進這 20 次；第一次 action 前不允許任何更新。

主比較使用相同的 1,024 個有效候選 actions，從固定 action cache 按固定順序取樣，只用 validity 篩選，不依任何模型的分數製作 shortlist。這讓 base、Q0 與更新方法不因候選池由某個參賽模型挑選而先天受益。必須保留每題候選 IDs、順序、實際 labels 與成功候選覆蓋率；任何方法選擇前都不能讀取未選 actions 的 labels。

既有 anchor panels 的歷史候選池由 base shortlist 產生，保留為另外標示的銜接評估。本輪主要候選池重新建立後，所有靜態 baseline 都要重跑；上面的 19%／54% 是歷史選擇依據，不能直接搬作新主表的 baseline。

主要指標 S@9；同時報 S@1／3／5／20、每輪累積成功曲線、每次失敗後更新耗時、整題耗時與實際更新數。先報每 fold 的配對差值，再報三 folds 的等權平均；不把不同 fold／seed 重複出現的場景當作獨立樣本增加可信度。

## 不增加訓練的歸因檢查

1. **同一個 real updater 在測試時把 residual 設為零。** 與 real 輸入比較可檢查模型是否依賴 residual；但這可能是輸入分布改變，所以必須與獨立訓練的 zero updater 一起看。
2. **Real updater 只在第一次失敗後更新一次，之後凍結 tokens。** 和持續更新比較，直接回答第二輪以後是否有額外收益。已試 action 仍持續排除。
3. **固定共同 support history 的離線診斷。** 用 Q0 的失敗試驗序列作共同 prefix，讓各方法讀相同歷史，對共同未嘗試候選評估 latent／geometry／排名的變化。這個診斷與各自閉環選 action 的主評估分開。

各方法從第二輪開始會選到不同 actions，存活到第 k 輪的場景也可能不同。因此不能直接比較各自剩餘場景的平均 AUC，便說某 updater 改善或破壞 ranking。需要共同 support/query 診斷，且不把它等同於實際閉環成功率。

## 工程檢查與加速

- Q0、兩個 updater 分支及 Q0＋AdaJEPA 在第一次 action 前的 scores 與選擇必須相同；base 與 base＋AdaJEPA 亦同。
- Zero updater 的輸出不可隨被遮掉的 residual 值改變；兩種 learned updater 訓練後的 Q0／backbone hash 不變。
- 未選 action 的 observation 不可進入線上更新。每輪排除已試 actions，成功後停止，每個新場景重設 tokens／適配權重。
- 只快取固定觀測、encoder 特徵與候選初始輸入；當下 residual 與候選 prediction 在 tokens／權重改變後重新計算。
- 候選推論使用已驗證的 B32 加速 scorer。不同場景可分批排程，單一場景的輪次仍維持因果順序。
- 原本加速驗證只涵蓋特定訓練歷史與最多 8 updates；新 checkpoint、19 updates、多狀態 query loss、Q0 AdaJEPA 的補驗現已通過，見 [補驗與成本報告](MULTITURN_SPEEDUP_COST_20261014.md)。CUDA 預設數值模式會造成原版對原版也分岔，使用既有受控設定檢查實作，不宣稱所有長程訓練都 bit-exact。

## 工作量與執行順序

主矩陣為 2 protocols × 3 folds × 6 方法＝36 組評估，即 3,600 個 scene-method episodes。每個 panel 需兩個 initial models 與兩個 learned updaters，共 12 個 initial models、12 個 updater trainings；其中四個 initial models 已完成可重用，因此新增 **8 次 initial training＋12 次 updater training＝20 次訓練**。這不含新 fold 的 encoder 資料製作、probe／scorer 訓練。

AdaJEPA 設定比較另外是 2 anchor panels × 2 initial architectures × 6 設定 × 25 題＝600 個 episodes。兩個閉環歸因分支若全做，另加 12 組、1,200 個 episodes；共同 support 的離線診斷另外計算。這些都不需要再訓練 updater。

執行順序：先核對 manifests 與 fold-specific 資料來源，準備兩個 anchor panels 的候選池與 smoke；開始 real／zero updater 訓練及 AdaJEPA 設定比較，同時準備新 folds；再平行展開其餘 folds 的配對 initial training、updater training 與 test。最多使用已授權的 6 張 GPU。

實測補驗已完成，詳見 [成本表與適用條件](MULTITURN_SPEEDUP_COST_20261014.md)。新增模型訓練投影約 113.5 GPU-hours；加主評估、TTA 搜尋、線上歸因與 encoding，平均 10–20 trials 情境的小計約 288–438 GPU-hours，未含 new heads、共同 query 離線診斷與排隊。Updater 投影取決於早成功比例，不能把自然短序列的平均值視為深度 19 的固定成本。

## 設計與證據位置

- [機器可讀設計](../results/multiturn_design_20261014/DESIGN.json)
- [初始模型實驗結果](MATCHED_BASE_Q0_RESULTS_20261014.md)
- [Updater 加速驗證](UPDATER_ACCELERATION_VALIDATION.md)
- 現有 learned updater：`pipeline/fast_updater.py`、`pipeline/prefix_policy_predictor.py`。
- 既有 AdaJEPA 移植保留於 `results/adajepa_phyre_20261010/code/tta.py`；新增 `pipeline/multiturn_runtime.py` 的 base／Q0 adapter 已通過長序列與相容性檢查。

本文件是可實作的設計，不是已執行結果；上述工程檢查已完成；新 fold 資料索引／packs 與 scorer 配方、provenance 尚須補齊，才能將 DESIGN 轉成含實際路徑與 hashes 的執行 PLAN。
