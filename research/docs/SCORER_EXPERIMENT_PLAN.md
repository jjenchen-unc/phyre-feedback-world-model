> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# 實驗計畫：world model 怎麼 rank，以及怎麼分辨 scorer 和 predictor 的責任

日期：2026-09-25。來源：與使用者討論 MPC ranking 與 success head 的對話。本文沒有實驗結果。

與既有文件的關係：本文補充 [PROPOSAL_REFRAMED.md](PROPOSAL_REFRAMED.md) §5–7 裡 scorer 的部分。那邊的 proposer / predictor / scorer 分層、split 定義、metrics、scorer-only 對照仍然適用。研究定位以 [INFORMATION_FRAMING.md](INFORMATION_FRAMING.md) 為準。本文新增的是：scorer 的輸入空間要錨定在哪裡、test-trained oracle scorer 能不能用、shortcut 風險，以及 initial-frame baseline。

證據等級標記：**[已驗證]** = 本次用指令查過 source；**[推論]** = 設計判斷；**[未查證]** = 記憶中的文獻或數字，引用前要再確認。

## 0. 決策記錄

2026-09-25 使用者定案：

| 項目 | 決定 |
|---|---|
| 算力預算 | 先做再說，暫不設上限。Candidate pool 先用 5K（§1.1.2a） |
| Split | 16:4:5，也就是 PHYRE 標準的 cross-template fold（16 train / 4 dev / 5 test templates），用 `phyre.get_fold` 取得。Within-template 同樣用標準 fold。跑 3 個 fold（0、1、2）。投影片 slide 11 的 15:5:5 要改 |
| Frame sampling | 全部做成可調參數。預設 encoder stride 8（7.5 fps）；predictor 每隔一個 latent 預測一次；horizon 固定。原本定案「不足的補最後一幀」，2026-09-25 review 發現這樣擋不住洩漏，改成所有 rollout 都跑滿完整模擬，見 §1.2.4 |
| 算力設計 | 原設計在評估階段不可行。2026-09-25 提出重設計，見 §1.3（提案，待確認） |
| Encoder 及格線 | 使用者請我提出，定義見 §2.4.3。可以調整，但要在看到結果之前定案 |

---

## 1. 決策迴圈

PHYRE 的一次 attempt 是：在 reset 後的場景放一顆紅球，動作是 (x, y, r)。MPC 在這裡是 trial-level 的 propose → predict → score → execute 迴圈，不是每個時間步都施加控制。

1. **Candidate pool。** 從 PHYRE 內建的 100k default actions 抽出固定的 candidate pool，所有方法共用。**[已驗證]** `phyre.get_default_100k_cache` 定義在 `VAORA/phyre/src/python/phyre/simulation_cache.py:42`。
2. **Invalid actions。** 紅球跟場景物體重疊時，simulator 回傳 `INVALID_INPUT`。Metrics 不把它算進 attempt 數。**[已驗證]** `action_simulator.py:48`，以及 `metrics.py:434` 的 `maybe_log_attempt`。所以 invalid 的 candidate 可以先濾掉，不影響 AUCCESS。
3. **Proposer（選配）。** 第一輪不訓練 proposer，細節見 §1.1。
4. **Predict。** Predictor 對每個 candidate 滾出一段 predicted latent sequence。
5. **Score。** 用凍結的 scorer 把每段 sequence 變成一個分數，選分數最高的去執行。
6. **Update。** 觀察真實 rollout，算出 residual，更新 memory。**同一批 candidates 重新 predict、重新 score**，再選下一個。重新打分只在 episode 的 shortlist 內進行，而且最多做 K 次，見 §1.3。

設計重點：residual 不直接進入 scorer。經驗只透過改變 predicted latent 來影響排序。所以 ranking 的變化才能歸因到 predictor 的 adaptation。

**不建議用 CEM 當主 planner。[推論]** PHYRE 的成功區域很窄，success 又是 binary，CEM 在沒有 dense 分數時容易 collapse。可以當 ablation。


### 1.1 Proposer 要不要訓練

**結論：主實驗不訓練 proposer，用 random sample 的固定 pool，所有方法共用。[推論]** 「針對 initial scene propose 強 candidates」的模型，其實就是 §5 的 initial-frame classifier。它可以兼任 proposer，但要守 §1.1.3 的限制。

#### 1.1.1 為什麼主實驗用 random 固定 pool

- **Adaptation 需要 coverage。** 失敗之後，正確答案常常在 proposer 原本評分低的地方。Pool 如果只有 proposer 的 top-100，答案可能不在裡面，memory 再強也選不到。Random pool 比較分散，保留 adaptation 的空間。
- **避免 confound。** Proposer 會改變 pool 裡有沒有解。訓練 proposer 之後，結果的變化會分不清是 proposer 還是 predictor 造成的。
- **避免跟 baseline 纏在一起。** 如果用 initial-frame classifier 挑 top-K，再讓本方法重排，本方法的上限就被 classifier 的 recall 卡住，跟它的比較也不再公平。

代價是成本。每次 attempt 後整個 pool 都要重新 rollout、重新打分。10K candidates 乘上長 rollout 可能太貴，所以 pool 大小要由 §1.1.2 的數據決定。

#### 1.1.2 先跑 coverage 檢查（不需要 GPU）

用 simulator 的離線 label，在 train / val tasks 上計算：

| Pool | 指標 |
|---|---|
| Random 1K | 至少含一個解的 tasks 比例 |
| Random 10K | 同上 |
| Classifier top-100 / top-500 | 同上，加上 recall：pool 內所有解有幾成被保留 |

Classifier 要先在 train templates 訓練好，才能算最後一列，所以這一列排在 §7 第 3 步之後。

判斷規則：

- **Random 1K 的 coverage 已經夠高：** 用 1K，成本約為 10K 的十分之一。
- **Random 1K 不夠，但 classifier top-500 的 coverage 接近 random 10K：** 改用兩階段 pool，細節見 §1.1.3。
- **Classifier top-K 的 recall 很低：** 不用它當 proposer。

「夠高」的門檻還沒定。要在看到 coverage 分布之後，跟 wall-clock 預算一起決定。

#### 1.1.2a 結果：random pool 的 coverage（2026-09-25）

**[已驗證]** 在 oprime 用 `coverage_check.py` 跑，查 PHYRE 內建的 100k ball cache，沒有跑模擬，約 1 分鐘。Split 是標準的 `ball_within_template` 與 `ball_cross_template` fold 0，只算 train 和 dev，沒碰 test。完整數字在專案根目錄的 `coverage_check.json`，本機跟 oprime 各一份。

Coverage 的定義是：pool 裡至少有一個成功 action 的 tasks 比例。下表是共用 pool、取 cache 前 N 個 action（PHYRE 慣例）的結果：

| Pool 大小 | Within train | Within dev | Cross train | Cross dev |
|---|---|---|---|---|
| 1K | 0.78 | 0.80 | 0.73 | 0.86 |
| 2K | 0.86 | 0.89 | 0.83 | 0.94 |
| 3K | 0.91 | 0.93 | 0.89 | 0.96 |
| 5K | 0.95 | 0.96 | 0.94 | 0.98 |
| 10K | 0.98 | 0.98 | 0.97 | 1.00 |
| 100K | 1.00 | 1.00 | 1.00 | 1.00 |

補充觀察：

- **Random 1K 不夠。** 大約 14–27% 的 tasks 在 pool 裡根本沒有解，不管 scorer 或 memory 多好都選不到。
- **轉折點在 3K–5K。** 5K 的 coverage 已經有 94–98%，跟 10K 只差 2–4 個百分點，但成本只有一半。
- **5 個 random seed 的平均跟前 N 個差不多。** N ≥ 1K 時差距都在 0.015 以內，所以直接用 PHYRE 的前 N 個慣例就好。但單一 seed 在 dev 上可以偏離 0.04 左右，因為 dev 只有 400 個 tasks。
- **各 task 各自抽 N 個 valid action，coverage 會高 2–5 個百分點。** 這主要是因為 valid action 比較多。共用 pool 裡大約只有 69–79% 是 valid，所以兩種定義不是同等大小的比較。
- **解很稀少。** 100k 裡每個 task 的解數中位數是 236–412 個，大約佔 0.2–0.4%。Train 裡各有 1 個 task 在 100k 裡完全無解。
- **最難的 templates 很一致。** 00020–00024 在 1K 時的 coverage 只有 0.22–0.48，10K 時仍然最低。
- **Cross dev 只有 4 個 templates。** 它的數字特別高，可能只是這 4 個 templates 比較簡單，不能代表 cross-template 的一般情況。之後應該跑多個 fold。

**建議。[推論]** 主實驗先用 5K 共用 pool。如果 wall-clock 撐不住，退到 3K，大約損失 4–5 個百分點的 coverage。Classifier top-K 那一列要等 §7 第 3 步訓練完 classifier 才能補。

#### 1.1.3 如果用 classifier 當 proposer

1. **K 要大，**大約幾百到一千，不要只取 100。另外混入一部分 random candidates，保持多樣性。
2. **Pool 在整個 episode 內固定。**看到 feedback 之後不重新 propose，不然 adaptation 的效果會跟 proposer 混在一起。
3. **報告 proposer 的 recall@K，**讓讀者知道 ranking 的上限在哪。
4. **Proposer 只用 train templates 訓練，**跟 §5 的 classifier 用同一份資料。

#### 1.1.4 延後：生成式 proposer

例如對 (x, y, r) 訓練 diffusion 或其他生成模型。這留到主結果出來之後再做。它回答的是「怎麼更有效率地搜尋」，不是「feedback 能不能讓 predictor adapt」。


### 1.2 Frame sampling 與 horizon（2026-09-25 量測）

**[已驗證]** 在 oprime 用 `temporal_check.py` 跑，約 16 秒。從 `ball_within_template` fold 0 的 train，每個 template 抽 8 個 task，共 200 個。每個 task 從 100k cache 各取一個成功和一個失敗的 valid action，用 stride 1 重新模擬，共 400 條 rollout：199 條成功、201 條失敗。完整逐筆資料在專案根目錄的 `temporal_check.json`。

PHYRE 的時間常數（`simulator/task_utils.h`）：物理模擬每秒 60 步，最多 1000 步，約 16.7 秒。成功條件是持續接觸 180 步，也就是 3 秒。預設 stride 60，等於每秒 1 幀、最多 17 幀。

#### 1.2.1 幀間位移

位移算的是物體中心的移動，加上長條物體旋轉造成的端點移動，單位是 PHYRE 256 px 畫面上的 pixel。「每條最大值」是一條 rollout 裡最快那一刻的位移。表中是 400 條的中位數，括號是 p90。

| Stride | 等效 fps | 16.7 秒幀數 | 每條最大位移 px | 最大位移超過 18 px 的 rollout 比例 |
|---|---|---|---|---|
| 4 | 15 | 250 | 8.6 (10.7) | 0% |
| 6 | 10 | 167 | 12.8 (15.9) | 1% |
| 8 | 7.5 | 125 | 16.9 (21.1) | 39% |
| 10 | 6 | 100 | 21.0 (26.2) | 71% |
| 15 | 4 | 67 | 30.9 (38.7) | 93% |
| 60（預設） | 1 | 17 | 99.2 (130.9) | 98% |

18 px 大約是 224 px 輸入、patch 16 時一個 patch 對應的寬度。

- **預設 1 fps 太粗。** 最快的時候物體一幀跳大約 100 px，也就是 5 個 patch 以上。碰撞跟反彈幾乎一定會被跳過。
- **Stride 6 到 8 是轉折點。** Stride 6 幾乎沒有超過一個 patch 的跳動。Stride 8 剛好是 LeVJEPA 的訓練取樣率，但有 39% 的 rollout 在最快那一刻會超過一個 patch。
- 位移大致跟 stride 成正比，峰值通常出現在自由落體的時候。

#### 1.2.2 Horizon

成功的開始時間是指連續接觸開始的那一刻，等於成功 rollout 的長度減 3 秒。在 199 條成功的 rollout 中：

| Horizon | 在 horizon 內開始接觸的比例 |
|---|---|
| 6 秒 | 24% |
| 8 秒 | 53% |
| 10 秒 | 76% |
| 12 秒 | 89% |
| 13.5 秒 | 100% |

- 中位數是 7.8 秒，最晚是 13.5 秒。各 template 的中位數在 4.1 到 12.2 秒之間，00021、00017、00022、00007 最晚。
- **Horizon 截短會直接漏掉成功。** 截在 10 秒，會有大約四分之一的成功 rollout 看不到接觸開始。
- 失敗的 rollout 一律跑滿 16.7 秒。以每步 0.1 px 當作靜止的門檻，最後一次移動的中位數是 4.6 秒，但 p90 是 14.5 秒，很多失敗的場景一直在緩慢移動。
- 有 1 條 rollout 跟 cache 不一致：`00004:759` 的一個 action，cache 標成功，重新模擬卻是失敗。400 條裡只有 1 條，可能是浮點誤差或接近邊界的情況，暫不處理。

#### 1.2.3 建議（已定案，見 §0）

- **Encoder 取樣用 stride 6 到 8，**也就是 7.5 到 10 fps。這樣位移大多在一個 patch 以內，也接近 LeVJEPA 的訓練取樣率。
- **Horizon 固定為完整的 16.7 秒，所有 rollout 都跑滿完整模擬，不補幀（§1.2.4）。** 截到 13.5 秒雖然涵蓋了這批樣本的所有接觸開始時間，但樣本只有 199 條，而且截短之後就看不到 3 秒的確認期。
- **成本。** 這樣一條 rollout 有 125 到 167 幀。Predictor 可以用比 encoder 更粗的時間解析度：encoder 用 stride 8 逐幀編碼，predictor 每隔一個 latent 預測一次，等效 3.75 fps、約 63 步。Real rollout 的 residual 一樣可以在 stride 8 的 latent 上算。Block-causal encoder 很適合這樣做。
- **Latent 長度（預設設定，encoder B）。** 模擬跑 1,017 步，在 stride 8 下剛好是 128 幀（約 16.95 秒），也就是 8 段 16 幀的 clip，不需要補幀（§1.2.4）。LeVJEPA 是逐幀 tokenize（tubelet 1），每幀 14×14 = 196 個 patch tokens：

  | 項目 | 時間步 | Tokens |
  |---|---|---|
  | 一段 clip | 16 | 3,136 patch tokens 加 1 個 CLS |
  | 整條 encoded rollout | 128 | 25,088 patch tokens 加 8 個 CLS |
  | Predictor 預測的序列（每隔一個 latent） | 64 | 12,544 |

  換成 V-JEPA 2（encoder A）時數字會不同，因為它把 2 幀合成一個 tubelet，一段 16 幀的 clip 只有 8 個時間步。
- **LeVJEPA 只訓練過 16 幀的 context。** 125 幀以上的長序列要分段 encode，或者先驗證長 context 下特徵有沒有變差。

#### 1.2.4 所有 rollout 跑滿完整模擬（2026-09-25 修正）

**問題。** PHYRE 一達成成功條件就停止模擬，所以成功的 rollout 比較短（§9）。原本的修法是用最後一幀補齊，但這樣擋不住洩漏：補出來的畫面會在動作中途突然凍結。失敗的 rollout 很少這樣，所以 head 和 predictor 都能從「突然凍結」學到答案。Predictor 如果用補過幀的資料訓練，還會學會「接觸之後畫面就凍結」，等於把答案寫進了預測目標。

**[已驗證]** 用 `fulllength_check.py` 在 oprime 抽 10 條成功的 rollout，改用不做成功判定的 `simulator.simulate_scene` 跑滿 1000 步：

- 10 條的前段畫面，跟 `simulate_action` 提早結束的版本逐 pixel 完全相同，所以動態沒有改變。
- 其中 7 條在原本停止的時間點之後，場景還在動。補最後一幀會讓這 7 條在動作中途凍結。
- `simulate_scene` 可以跑超過 1000 步，1,024 步的前 1000 步跟 1000 步的版本逐 pixel 相同。

**修正後的資料生成流程：**

1. **Rollout 影片：** 把 action 轉成 user input 加進場景，也就是 `add_user_input_to_scene`，再用 `simulate_scene` 跑 1,017 步，每 8 步取一幀，得到 128 幀。所有 rollout 長度一樣，不論成功或失敗。
2. **Success label：** 另外用標準的 `simulate_action` 取得 PHYRE 判定的結果，也就是 1000 步、持續接觸 3 秒。第 1000 步之後的幀不影響 label。
3. **Label 用重新模擬的結果，不用 cache。** Cache 只用來抽 action。§1.2.2 發現 400 條裡有 1 條 cache 跟重新模擬不一致，用重新模擬的結果才能保證 label 跟影片一致。
4. **Scorer、probe、predictor 的輸入全部用這個固定長度的 rollout。** 不再有任何補幀。

**成本。** 每條 rollout 的模擬從平均幾百步變成固定 1,017 步。§1.2 的量測中，400 條 stride 1 模擬約 16 秒，所以模擬不是瓶頸。Render 和存檔的成本要在大量生成時實測。

### 1.3 算力設計（2026-09-25 提案，待確認）

#### 1.3.1 問題：原設計在評估階段跑不完

原設計的每一次失敗，都要把 5K 個 candidates 重新 predict。AUCCESS 最多算到 100 次 attempt，所以成本是這幾個數字相乘：

| 因子 | 原設計 |
|---|---|
| Test tasks | 500 |
| 重新打分次數 | 最多 100 |
| Candidates | 5,000 |
| 每條 predicted rollout 的 tokens | 64 步 × 196 = 12,544 |

**粗估。[推論]** 假設 predictor 有 5,000 萬參數，GPU 有效算力每秒 50 TFLOPs，attention 成本不算。每個變體、每個 fold、每種 setting 大約要 2.5 億條 rollout，約 1,700 GPU 小時。乘上 5 個變體、3 個 fold、2 種 setting，大約 5 萬 GPU 小時。

根本原因是把兩個問題綁在一起評估：「feedback 有沒有讓預測變好」很便宜，「closed-loop 能不能解題」才貴。

#### 1.3.2 兩層評估

| 層 | 回答的問題 | 做法 | 成本 |
|---|---|---|---|
| **第一層（主實驗）：support–query** | Content 與 Transfer：feedback 有沒有讓未試過的 action 預測得更準、排得更好 | 每個 task 固定一組 context attempts 和 32 個 query actions。比較有無 feedback 時 query 的 latent prediction error，以及 32 個 query 之間的 ranking AUC | 500 tasks × 32 queries × 幾種 context 數，大約幾萬次預測，幾乎免費 |
| **第二層：closed-loop** | Utility：能不能在更少 attempt 內解題 | 見 §1.3.3 | 每個變體、fold、setting 約 256 萬次預測 |

第一層直接對應投影片 slide 2 的前兩個研究問題，以及 slide 12 的「same untried action prediction error」。Memory 變體的大部分比較、ablation 和除錯都在第一層做。只有最後的 decision utility 才需要第二層。

Query actions 要固定而且所有方法共用。每個 task 裡要保證同時有成功和失敗的 query，可以從 cache 抽，不然 ranking AUC 沒有意義。Context attempts 也從 cache 抽，照 PROPOSAL_REFRAMED.md §6 的 support/query 規則，不能跟 query 重疊。

#### 1.3.3 Closed-loop 的四個成本控制

1. **共用 shortlist（M = 512）。** 每個 task 先從 5K pool 選出 top-512，所有方法共用，之後只在裡面排序。Shortlist 可以用 frozen predictor 或 initial-frame classifier 產生。選哪一個，看兩者的 recall@512 誰高，這可以用 simulator label 離線算。因為所有方法共用同一個 shortlist，§1.1.1 擔心的「不同方法的 pool 不同」這個 confound 不會發生。代價是所有方法的上限都被 recall@512 卡住，這個數字要一起報告。
2. **Memory 最多更新 K = 10 次。** 前 10 次失敗後重新打分。第 11 到 100 次 attempt 沿用第 10 次之後的排序，不再重新打分。Success@1/3/5/10 完全反映 adaptation，AUCCESS@100 仍然算得出來，不用額外成本。
3. **排除已試過的 action。** 失敗的 action 從 shortlist 移除。任務一解出來就停，所以簡單的 task 平均用不到 10 次。
4. **同一次打分的 512 個 candidates 共用同一份 memory，一次 batch 算完。**

#### 1.3.4 降低每條 rollout 的成本

| 手段 | 原本 | 提案 | 備註 |
|---|---|---|---|
| 每幀 tokens | 196（14×14） | 49（7×7 池化） | 池化會不會丟位置資訊，放進 §2.4.3 的 encoder 比較一起量 |
| Predictor 模式 | 未定，隱含 autoregressive 64 步 | Chunked 或 direct | PHYRE 的 action 只在 t = 0 施加，不需要逐步滾。Direct 模式也可以只預測 scorer 需要的時間點 |
| Predictor 大小 | 未定 | 小模型，先從幾千萬參數開始 | 可調 |

原設計每條 rollout 是 12,544 個 tokens，提案大約是 3,136 個（64 步 × 49）。Direct 模式若只預測 32 個時間點，是 1,568 個。

#### 1.3.5 新舊對照

| 項目 | 原設計 | 提案 |
|---|---|---|
| 主要研究問題的評估 | 只有 closed-loop | 第一層 support–query |
| 每次重新打分的 candidates | 5,000 | 512 |
| 重新打分次數 | 最多 100 | 最多 10 |
| 每條 rollout 的 tokens | 12,544 | 1,568 到 3,136 |
| Closed-loop 粗估，每個變體、fold、setting | 約 1,700 GPU 小時 | 約 2 GPU 小時 |

提案的粗估用同樣的假設：predictor 5,000 萬參數、每條 rollout 1,568 個 tokens。這些數字只是量級。Predictor 做出來之後，要實測每秒能處理幾條 rollout，再更新這張表。

#### 1.3.6 其他省算力的規則

- **開發只用 fold 0 和 100 個 dev tasks。** 最終結果才跑 3 個 fold 的完整 test set。
- **Latent 儲存。** 完整解析度下一條 rollout 約 51 MB（25,088 tokens × 1024 維 × fp16）。池化到 7×7 之後約 12.8 MB；再把維度投影到 256，約 3.2 MB。要在大量生成訓練資料之前定下來。
- **Encoder 只在兩個地方跑：**編碼訓練資料，以及編碼 closed-loop 裡真的執行過的 attempt。它不在 candidate 打分的迴圈裡，所以不是瓶頸。


### 1.4 第一、二步結果（2026-09-25）

#### 1.4.1 資料管線

**[已驗證]** `pipeline/make_rollouts.py`，依 §1.2.4 的流程生成。每個 task 從 cache 取 2 個成功、2 個失敗的 action。Label 用重新模擬的結果。

- **範圍：** Fold 0、1、2，within 和 cross 各自的 train（全部 1,600 個 task）與 dev（依 template 分層抽 100 個 task）。
- **規模：** 每份 train 約 6,395 條，每份 dev 400 條。Rollout 以 task 和 action 命名，三個 fold 共用，去重後共 9,994 條、749 MB，放在 oprime 的 `data/rollouts/`。
- **驗證：** 所有 rollout 都是 128 幀。每份抽 80 條跟 `simulate_action` 的 stride 8 畫面比對，全部逐 pixel 相同，物體座標差異為 0。每份 train 跟 cache 的 label 不一致 1 到 4 條，dev 0 到 1 條。每份 train 有 4 到 5 個 task 在 cache 裡的成功 action 不到 2 個。

#### 1.4.2 Success readout：學習式 vs 規則

輸入都是 simulator 的物體座標軌跡，不經過 encoder。學習式參照組每幀在物體之間做 self-attention，再接 temporal transformer，3 個 seed，用完整 train 訓練。規則不需要學習，以 green ball 到 goal 的表面距離連續低於門檻的最長幀數當分數；長條物體用點到線段的距離，其他形狀用圓形近似，門檻在 train 上選（3 或 4 px）。

| Setting | Fold | 學習式，完整軌跡 | 學習式，只看第一幀 | 規則，圓形近似 | 規則，線段距離 |
|---|---|---|---|---|---|
| Within | 0 | 0.986 | 0.911 | 0.781 | 0.980 |
| Within | 1 | 0.996 | 0.951 | 0.721 | 0.987 |
| Within | 2 | 0.991 | 0.934 | 0.771 | 0.971 |
| Cross | 0 | 0.751 | 0.693 | 0.980 | 0.978 |
| Cross | 1 | 0.933 | 0.740 | 0.980 | 0.999 |
| Cross | 2 | 0.945 | 0.744 | 0.833 | 0.997 |

數字是 dev AUC。學習式是 3 個 seed 的平均。

- **規則在 6 組上都 ≥ 0.97，**而且完全不依賴 train templates。它目前的弱點是罐形 goal，fold 0 cross 只有 0.931。
- **學習式在 cross 上不穩定。** 三個 fold 分別是 0.75、0.93、0.95。Fold 1 的 3 個 seed 從 0.87 到 0.99。它能不能泛化，很依賴哪些 templates 被拿出來當 dev。
- **在 cross 上，軌跡對學習式很有幫助。** 只看第一幀大約 0.69 到 0.74，看完整軌跡在 fold 1、2 提升到 0.93 以上。
- **把池化架構換成關係式架構沒有差別。** Fold 0 cross 兩者都是 0.75。增加每個 template 的 task 數也沒有差別：400 個 task 是 0.745，1,600 個是 0.751。
- 各 fold 的 cross dev templates：fold 0 是 00000、00008、00011、00018；fold 1 是 00000、00002、00007、00014；fold 2 是 00006、00009、00018、00019。

**含意。[推論]** 主線 scorer 應該優先考慮 §2.2 的 probe 加規則。學習式 success head 從 16 個 templates 學不到穩定的跨 template 判斷，就算輸入是完美的物體座標也一樣。規則的前提是能從 latent 讀出準確的物體位置，這正是第三步位置 probe 要量的。

Script：`pipeline/ref_classifier.py`（`--arch relational`）、`pipeline/rule_scorer.py`。結果：`results/step2_reference_rel*.json`、`results/rule_scorer.json`。



### 1.5 Base predictor 第一版（2026-09-25，fold 0 within）

**[已驗證]** `pipeline/train_predictor.py`：direct 模式，一次 forward 預測 32 個時間點的完整 14×14 LeVJEPA latent，不把預測餵回去。架構是 space-time 分解的 transformer（每層先做同一時間點的空間 attention，再做同一空間位置的時間 attention），6 層、寬度 384、約 1,500 萬參數。輸入是初始畫面的 latent 加 action 向量，輸出相對初始畫面的變化量，L1 loss。訓練資料是 fold 0 within 的多樣資料（6,382 條），20 個 epoch。評估用 `pipeline/eval_pred_scorer.py`，同一個凍結的 within probe（task 校準，規則為軟分數 τ=4、σ=1）分別讀真實 latent 和預測 latent。

| Fold 0 within dev（400 條） | 規則用 simulator 座標 | 讀真實 latent | 讀預測 latent |
|---|---|---|---|
| 端到端 AUC | 0.979 | 0.978 | 0.738 |
| Task 內 AUC | 0.995 | 0.990 | 0.771 |
| Green 位置誤差中位數 | — | 1.4 px | 15.5 px |

- **預測的位置誤差隨時間增加：** 第 1 個時間點（約 0.5 秒）2 px，約 3 秒後到 20 px 左右，之後大致持平在 18 到 23 px。
- **World model 是目前最大的瓶頸。** Probe 和規則讀真實 latent 幾乎完美，換成預測 latent 就掉到 0.74。
- **Latent L1 不是可靠的指標。[推論]** Dev L1 從「複製初始畫面」的 0.283 降到 0.059，但複製 baseline 在第 1 個時間點就有 0.29，之後一直在 0.27 到 0.30 之間交替，物體幾乎還沒動時就這麼大。這表示 L1 大部分來自 encoder 本身的系統性變化（重疊視窗中每一幀的位置不同），跟物理無關，predictor 的改善很可能大多是學會了這種變化。第一層主實驗不能只用 latent L1，要搭配 probe 讀出的位置誤差。
- **修正的 bug：** 第一版輸出的第 0 個時間點沒有換回觀測到的初始 latent。評估時改成直接用真實的第 0 幀，AUC 從 0.7365 變成 0.7382，影響很小。訓練 script 也已修正。
- **讀資料很慢：** 從 `/mnt/meg` 讀 6,782 個 latent 檔花了 47 分鐘，訓練本身約 50 分鐘。
- 限制：只有 fold 0 within、一個 seed、沒有調過超參數。



### 1.6 Predictor 調整第一輪與第一次 closed-loop（2026-09-26）

**規則（2026-09-26 使用者定案）：predictor 的訓練只能用 latent loss。** 任何讓 scorer（probe 或規則）的梯度回傳到 predictor 的訓練目標，都會讓 predictor 和 scorer 綁在一起，違反 §2.4.1 和 §3 的歸因前提。這類目標只能當成獨立的研究條件（對應 PROPOSAL_REFRAMED.md 的「prediction + decision objective」），單獨報告，並用獨立訓練的 probe 評估，不能拿來選 predictor 的設定。

**資料讀取。** Latent 合併成單一大檔（`pipeline/build_pack.py`，放在 `/mnt/meg/jjenchen/phyre-unc/packs/`），用記憶體映射讀取，載入從 47 分鐘變成約 1 秒。

**調整結果（fold 0 within dev）。** 每組只改一個地方。評估用兩個凍結的 probe：主 probe（多樣資料、task 校準），以及獨立 probe（seed 1、只用每個 task 4 條的資料）。

| 設定 | 主 probe AUC | 獨立 probe AUC | 獨立 probe green 誤差 | 類別 |
|---|---|---|---|---|
| Base（寬 384、6 層、20 epoch） | 0.738 | 0.765 | 16.1 px | 純 latent |
| 動態加權 L1（β=4） | 0.749 | 0.774 | 15.4 px | 純 latent |
| 模型加大（寬 512、8 層、30 epoch） | 待補 | 待補 | 待補 | 純 latent |
| Probe 輔助 loss λ=0.1 | 0.850 | 0.787 | 19.6 px | **用到 scorer，獨立條件** |
| Probe 輔助 loss λ=1 | 0.822 | 0.745 | 26.9 px | **用到 scorer，獨立條件** |

- Probe 輔助 loss 的兩組，用主 probe（訓練時用的那個）評估看起來大幅進步，換成獨立 probe 就幾乎沒有進步，位置誤差反而變大：predictor 學會迎合那個特定 probe，而不是更準地預測物理。結果與模型保留，之後依上面的規則調整。
- 動態加權有小幅進步，可能在雜訊範圍內。

**第一次 closed-loop（base predictor，fold 0 within dev 100 個 task）。** `pipeline/render_candidates.py` 畫出每個 task 前 5,000 個 cache action 中有效的候選（每個 task 約 3,600 到 4,700 個），`pipeline/closed_loop_eval.py` 對每個候選做：兩幀編碼（第 0 幀重複兩次，已驗證跟訓練時的 latent 完全一致）→ predictor → 主 probe → 規則打分，依分數排序後查 cache 判斷成功。上限用 `pipeline/truth_rule_candidates.py`：用 simulator 模擬每個候選的真實物體軌跡，再用同樣的規則打分。

| Within fold 0 dev | AUCCESS@100 | Success@1 | Success@10 | 候選池內 AUC |
|---|---|---|---|---|
| 隨機排序 | 0.156 | 1.4% | 11.7% | 0.5 |
| Base predictor 加 probe 加規則 | 0.305 | 9% | 28% | 0.768 |
| 上限：真實軌跡加同樣規則 | 0.910 | 89% | 92% | 0.993 |

- 規則本身幾乎完美，**瓶頸是 predictor**。
- 每個 task 約 41 秒，100 個 task 約 70 分鐘。
- 上限在 template 00024 只有 0.25，00003 為 0.67，00008、00013、00014、00022 為 0.75，其餘大多是 1.00。
- 這些 dev task 也被用來選 predictor 設定，數字會略為樂觀。



### 1.7 Initial scene baseline、資料擴充與下一階段的設計（2026-09-26）

#### 1.7.1 PHYRE DQN 的資料規模

**[已驗證]** `VAORA/phyre/agents/run_experiment.py:44-56` 的 `DQN_BASE_ARGS`：100,000 次更新、每批 64、成功與失敗平衡抽樣、評估時排序 cache 前 10,000 個 action。`train.py` 的 `--max-train-actions` 預設不設，也就是每個 train task 用全部 100k 個 cache action。以 within fold 0 來說，可用樣本約 200 萬個成功、1.15 億個失敗。

#### 1.7.2 Initial scene 機率模型（`pipeline/scene_classifier.py`）

仿照 DQN：輸入是不含紅球的場景（同一個凍結的 LeVJEPA，第 0 幀重複兩次）加 action 向量，兩層 transformer 加 attention pooling 輸出成功機率。每個 task 只需編碼一次場景，label 全部來自 cache。

#### 1.7.3 PCA 壓縮（`pipeline/pca_project.py`）

在多樣資料的 latent 上擬合 PCA（隨機 20 萬個 token）。128 維保留 94.7% 變異，probe 端到端 AUC 0.978（1,024 維為 0.978），green 誤差 1.49 px（1.37 px）。新資料的 latent 一律在 encode 時直接投影到 128 維。

#### 1.7.4 資料擴充

`ball_within_template_f0_train_s16r16`：within fold 0 全部 1,600 個 train task，每個 task 16 個成功（最遠點取樣）加 16 個自然比例隨機 action，共 52,349 條（25,527 成功）。抽驗 660 條與 PHYRE 逐 pixel 相同，label 與 cache 不一致 50 條（0.1%）。

#### 1.7.5 結果（within fold 0 dev，100 個 task，cache 前 5,000 個 action 中的有效候選）

| 方法 | 訓練資料 | AUCCESS@100 | Success@1 | Success@10 | 候選池內 AUC |
|---|---|---|---|---|---|
| 隨機排序 | — | 0.156 | 1.4% | 11.7% | 0.5 |
| Initial scene 模型 | 6,382 條（同 predictor） | 0.207 | 8% | 20% | — |
| Initial scene 模型 | DQN 規模，640 萬次抽樣 | 0.587 / 0.608（兩次訓練） | 35% | 60% | 0.944 |
| Base predictor（寬 384、6 層） | 6,382 條 | 0.305 | 9% | 28% | 0.768 |
| 加大 predictor（寬 512、8 層） | 6,382 條 | 0.378 | 11% | 39% | 0.832 |
| Base predictor，128 維 | 52,349 條 | 0.480 | 18% | 49% | 0.876 |
| 上述 + initial scene 模型，A：shortlist 512 重排 | — | 0.535 | 19% | 55% | — |
| 上述 + initial scene 模型，B：排名平均 | — | 0.624 | 34% | 66% | — |
| 加大 predictor，128 維 | 52,349 條 | 0.483 | 24% | 50% | 0.901 |
| 上述 + initial scene 模型，B：排名平均 | — | 0.616 | 35% | 62% | — |
| 上限：真實軌跡加規則 | — | 0.910 | 89% | 92% | 0.993 |

離線評估（平衡 dev，400 條，獨立 probe）：base predictor 在 52,349 條上是端到端 AUC 0.876、green 誤差 5.4 px（6,382 條時為 0.765、16.1 px）；加大 predictor 在 52,349 條上是 0.876、4.4 px，task 內 AUC 0.899（base 0.884）。

- 同樣資料下，rollout 優於 initial scene 模型（0.305 vs 0.207）。差距主要來自資料量。
- 結合 B 首次超過 initial scene 模型單獨（0.624 vs 0.608），但 initial scene 模型兩次訓練就差 0.02，這個領先在雜訊範圍內，需要多個 seed 才能確認。
- 注意兩種 AUC 不能互比：「候選池內 AUC」是每個 task 在數千個候選上算、再對 task 平均；「端到端 AUC」是平衡 dev 400 條一起算。

#### 1.7.6 討論與決定（2026-09-26）

**Baseline 定案，不再搜尋訓練設定。** 研究問題看的是同一個 base 有無 feedback 的差值，base 再調高不會讓問題更好回答；目前 AUCCESS 0.48、上限 0.91，改善空間充足；在 52,349 條上加大模型的增益已經很小。預設 base 大小加 52,349 條的 predictor，除非加大版的 closed-loop 明顯更好（AUCCESS 高 0.03 以上）。**結果：加大版 0.483 對 base 0.480，差 0.003，未達門檻，baseline 定為 base 大小（`/mnt/meg/jjenchen/phyre-unc/runs/base_s16r16_pca128`）。** 加大版的 Success@1（24% vs 18%）和候選池內 AUC（0.901 vs 0.876）較好，但訓練時間約兩倍。

**Predictor 的訓練目標只有 latent L1。** 已從各模型存下的參數確認：baseline 候選（base_s16r16_pca128、big_s16r16_pca128、base、big、motion）都沒有 probe position loss；只有 aux0.1、aux1 有，已標成獨立研究條件。

**指標設計。** Latent 的絕對 L1 不可靠（大部分來自 encoder 的系統性變化，且對物體移動不敏感：位置誤差從 16.1 降到 11.6 px 時，latent L1 只從 0.0586 變成 0.0563）。但有無 feedback 用同一個 base 時，這部分兩邊相同，**差值**仍然有意義，而且是 proposal slide 12 的指標。主要指標定為：

1. 同一批 query、同一個 base，有無 feedback 的 latent 誤差差值。
2. 只算會動物體附近 token 的 latent 誤差（simulator 位置只用於評估）。
3. Probe 讀出的位置誤差。
4. AUCCESS 與 Success@k。

前兩個是 prediction utility，後兩個是 decision utility，對應 INFORMATION_FRAMING.md 的分開報告。

**Adaptive 版本的對照組。** 主要比較是同一個 base 有無 feedback。另外要有「會 adapt 的 initial scene 模型」：PHYRE 本身有 DQN online（`--dqn-finetune-iterations`，用測試時的嘗試微調分類器），AdaJEPA 的 test-time gradient 也屬於這一類。

**訓練資料是否適合 adaptation。** 使用者的假設：adaptation 的效益會出現在相鄰或因果相似的 action 之間。目前的 s16r16 資料是刻意分散的，量測結果（`pipeline/pair_stats.py`）：

| 每個 task 內的配對（共 832,270 組） | 數量 | 比例 |
|---|---|---|
| Action 距離 < 0.1 | 26,119 | 3.1% |
| 距離 < 0.1 且結果不同 | 1,704 | 平均每個 task 約 1 組 |
| 讓同一組物體動起來 | 680,712 | 81.8%（定義太粗，沒有鑑別力） |

最近鄰居距離中位數 0.15，只有 9% 的 action 在 0.05 內有鄰居。

**提案：鄰近群集資料。** 每個 task 挑錨點（例如 2 個成功、2 個差一點成功的失敗），從 cache 找半徑 0.05 到 0.1 內的鄰居，每群約 8 個並刻意混合成功與失敗。同群互為 context 與 query 是近距離轉移，跨群是遠距離對照，直接對應 RQ2。規模約 5 萬條（生成約半小時、encode 約 3 小時）。因果相似的細緻定義（紅球先撞哪個物體、green 的移動方向、最接近 goal 的時間）先用於分析，不用於挑資料。

**順序。** 第一版 memory 先用現有資料訓練（主要是遠距離轉移），同時生成鄰近群集資料；兩者完成後比較近距離與遠距離的轉移。



### 1.8 第一層評估工具與 baseline 數字（2026-09-26）

**鄰近群集資料。** `make_rollouts.py --clusters 4 --cluster-size 8 --radius 0.08`：每個 task 4 群，錨點是 2 個隨機成功 action 加 2 個「離某個成功 action 最近的失敗 action」；每群再取半徑 0.08 內的 7 個有效鄰居，盡量混合成功與失敗；每條 rollout 記錄群集編號。Dev（within fold 0 dev 的 100 個 task）：3,200 條、400 群、每群 8 條、98% 的群同時有成功和失敗、群內離錨點最遠距離中位數 0.076。Train（1,600 個 task）生成與 encode 進行中。

**Episode（`pipeline/eval_tier1.py`，固定存在 `results/tier1_episodes_within_f0_dev.json`，共 700 個）：**

| 類型 | Context | 數量 |
|---|---|---|
| k1 | 每群各挑 1 條當 context，其餘 31 條為 query | 400 |
| k2 | 隨機 2 群各 1 條 | 100 |
| k4 | 4 群各 1 條（沒有遠距離 query） | 100 |
| scorer_fail | 凍結的 baseline scorer 在該 task 32 條中排最高、但實際失敗的那一條 | 100 |

同群的 query 為近距離，其他群為遠距離。

**兩個設計問題（2026-09-26 與使用者討論）：**

- **下一個 action 應該由 scorer 選嗎？** 實際解題時是（第二層 closed-loop）。第一層刻意固定 context，讓所有方法拿到同樣的 feedback，才能分清是 feedback 用得好還是剛好選到有用的 attempt。依使用者意見加入 scorer_fail：由凍結的 baseline 選出它最有信心卻失敗的 attempt，更貼近真實解題，而且所有方法共用，比較仍然公平。
- **所有方法都要跑同一個 action 序列嗎？** 候選池要一樣（決定上限，PHYRE 標準做法也是共用 10,000 個 action）；第一層的 context 要一樣（控制實驗）；但 closed-loop 裡嘗試的順序由各方法自己的 scorer 決定，這正是要評估的東西。

**Baseline（不給 feedback）的第一層數字。** Baseline 不讀 context，所以近距離和遠距離應該一樣，這裡是之後比較的起點。

| Episode | Query 數 | Latent L1 | 物體附近 latent L1 | Green 誤差中位數 | 排序 AUC（episode 內） |
|---|---|---|---|---|---|
| k1 | 12,400 | 0.1029 | 0.2443 | 23.8 px | 0.610 |
| k2 | 3,000 | 0.1030 | 0.2447 | 23.3 px | 0.614 |
| k4 | 2,800 | 0.1029 | 0.2443 | 24.2 px | 0.609 |
| scorer_fail | 3,100 | 0.1026 | 0.2438 | 22.5 px | 0.640 |

- **群集資料比平衡 dev 難很多。** 同一段程式在平衡 dev 上得到 green 誤差 5.9 px，在群集 dev 上是 23.9 px（`pipeline/check_tier1_code.py` 確認）；probe 讀真實 latent 在兩者都約 1.5 px，所以差距來自 predictor。各 template 從 7 px 到 98 px 不等。**[推論]** 平衡 dev 的隨機失敗多半是場景幾乎不動的簡單情況；群集資料取在解附近，碰撞和互動多。之前的 5.4 px 高估了 predictor 在關鍵情況下的能力，這也是 adaptation 最有空間的地方。
- 排序 AUC 約 0.61：同群的 action 非常相近、結果卻不同，本來就很難區分。

**程式調整。** 共用的模型定義與規則函式抽到 `pipeline/common.py`，`closed_loop_eval.py` 和 `eval_tier1.py` 都從它匯入。



### 1.9 Cross-template 第一段（2026-09-27，fold 1）

`pipeline/run_cross_f1_stage1.sh`：在 fold 1 cross 的 train（16 個 templates）上訓練 initial scene 模型（DQN 規模，144 萬成功、1.13 億失敗），候選池為 fold 1 cross dev 100 個 task 的 cache 前 5,000 個有效 action；上限用真實軌跡加規則（軟分數 τ=4、σ=2，fold 1 依 template 校準的結果）。

| Cross fold 1 dev | AUCCESS@100 | Success@1 | Success@10 | 候選池內 AUC |
|---|---|---|---|---|
| 隨機排序 | 0.274 | — | — | 0.5 |
| Initial scene 模型 | 0.361 | 26% | 37% | 0.757 |
| 上限：真實軌跡加規則 | 0.962 | 96% | 96% | 0.998 |

各 template（上限 / initial scene 模型）：00000 0.97 / 0.49、00002 1.00 / 0.82、00007 1.00 / 0.12、00014 0.88 / 0.01。

- Initial scene 模型領先隨機的幅度，從 within 的 +0.45（0.608 對 0.156）縮到 cross 的 +0.09。它在看過的 templates 上的優勢，主要來自記憶版面。
- 規則的上限在 cross 上仍然很高，world model 在這裡可能有相對優勢。**依 §7 的條件，第 10 項（cross 第二段）應該執行。**
- 限制：只有 fold 1，dev 只有 4 個 templates。

**第二段結果（`pipeline/run_cross_f1_stage2.sh`，2026-09-27）。** 在 fold 1 cross 的 train 上生成 s16r16 資料（1,600 個 task），用 PCA 128 維 encode，訓練 base 大小的 predictor（8 個 epoch）與依 template 校準的 probe（端到端 AUC 0.980，green 誤差 1.52 px；參照組 0.997），再跑 closed-loop。

| Cross fold 1 dev（100 個 task） | AUCCESS@100 | Success@1 | Success@10 | 候選池內 AUC |
|---|---|---|---|---|
| 隨機排序 | 0.274 | — | — | 0.5 |
| Initial scene 模型 | 0.361 | 26% | 37% | 0.757 |
| **World model** | **0.457** | 28% | 47% | 0.644 |
| 結合 A：shortlist 512 重排 | 0.468 | 29% | 46% | — |
| 結合 B：排名平均 | 0.415 | 25% | 43% | — |
| 上限：真實軌跡加規則 | 0.962 | 96% | 96% | 0.998 |

各 template（上限 / initial scene / world model）：00000 0.97 / 0.49 / 0.60、00002 1.00 / 0.82 / 0.88、00007 1.00 / 0.12 / 0.00、00014 0.88 / 0.01 / 0.35。

- **在沒看過的 templates 上，world model 單獨就贏過 initial scene 模型**（0.457 對 0.361），4 個 templates 贏 3 個。Within 上則是輸（0.480 對 0.608）。支持「initial scene 模型靠記憶版面、world model 靠預測物理而泛化較好」的推論。
- World model 的候選池內 AUC 較低（0.644 對 0.757）但 AUCCESS 較高：它對排名最前面的判斷比較準。
- 限制：只有 fold 1、4 個 dev templates、一個 seed。需要在 fold 0、2 上重複。

**資料量相同的比較（2026-09-27）。** Initial scene 模型改用與 world model 完全相同的 (task, action) 配對訓練（`--data-mode matched`，s16r16，約 5.2 萬筆，2 萬次更新，平衡抽樣）：

| AUCCESS@100 | 隨機 | Initial scene，同樣 5 萬筆 | World model，同樣 5 萬條 | Initial scene，DQN 規模 |
|---|---|---|---|---|
| Within fold 0 | 0.156 | 0.523 | 0.480 | 0.608 |
| Cross fold 1 | 0.274 | 0.333 | 0.457 | 0.361 |

各 template（cross，initial scene 同資料 / world model）：00000 0.29 / 0.60、00002 0.78 / 0.88、00007 0.22 / 0.00、00014 0.04 / 0.35。

- Within：6,382 條時 world model 贏（0.305 對 0.207），5 萬條時 initial scene 模型反超（0.523 對 0.480）。在看過的 templates 上，資料夠多時直接記住「版面加 action 對應到成功」很有效。
- Cross：同樣資料下 world model 明顯贏（0.457 對 0.333），也贏過多用了 1,000 倍以上樣本的 DQN 規模版本（0.361）。
- **[推論]** initial scene 模型的優勢來自記憶看過的版面，world model 的優勢是泛化到新 templates。仍然只有 fold 1、4 個 dev templates、一個 seed。



### 1.10 Memory 第一版（2026-09-27，within fold 0）

**設計（`pipeline/common.py` 的 `MemoryPredictor`、`pipeline/train_memory.py`）：**

- Backbone 是凍結的 baseline predictor（base 大小、52,349 條、128 維）。
- 每個 context attempt 的觀測 latent 取 8 個時間點（約每 2.1 秒）、空間池化成 7×7，一個 attempt 392 個 token；加上時間、空間、action、成功與否的 embedding；最多 4 個 attempt，加一個固定的空白 token，送進 2 層 transformer。
- 6 個 backbone block 之後各插入一層 cross-attention，讓 query 的所有 token 讀 memory；輸出層零初始化。已驗證沒訓練時輸出與 baseline 完全相同（最大差異 0）。
- 可訓練參數 466 萬。訓練：s16r16 資料，每筆樣本同一個 task 抽 1 到 4 條 context（10% 不給）與 1 條 query，只用 query 的 latent L1，2 萬步、每批 8，約 68 分鐘。
- 模式：`full`（觀測 latent 加 action 加結果）、`outcome`、`action`、`residual`（觀測 latent 加 baseline 對該 context 的預測誤差，baseline 凍結所以可以即時計算）。

**第一版（`full` 模式）結果：幾乎沒有效果。**

| k1 episode（12,400 個 query） | Latent L1 | 物體附近 latent L1 | Green 誤差 | 排序 AUC |
|---|---|---|---|---|
| Baseline | 0.1029 | 0.2443 | 23.82 px | 0.610 |
| Memory 模型，不給 context | 0.1030 | 0.2444 | 23.58 px | 0.612 |
| Memory 模型，給 context | 0.1029 | 0.2443 | 23.06 px | 0.610 |

其他 episode 類型、近距離與遠距離都一樣：latent 誤差到小數點後四位不變，位置誤差少約 0.5 px（其中一部分在不給 context 時就有），排序 AUC 不變。訓練 loss 只從 0.078 降到 0.077。

**診斷：context 裡有沒有可以學的資訊？（`pipeline/residual_corr.py`）** 對 dev 群集資料的每條 rollout 算 baseline 的預測誤差（真實 latent 減預測），比較配對之間的 cosine 相似度：

| 配對 | Baseline 誤差的 cosine |
|---|---|
| 同 task、近距離（同群） | 0.303 |
| 同 task、遠距離（不同群） | 0.210 |
| 不同 task | 0.009 |

- Baseline 在同一個場景的誤差有明顯關聯，近距離更強，支持使用者的假設。遠距離也有 0.21，代表有一部分是場景共通的偏差。
- 誤差大小約為物體移動量（`z_t − z_0`）的 15%。
- **[推論]** 第一版只看到觀測 latent，沒有看到 baseline 對 context 的預測，要自己推算「錯在哪裡」很難。所以下一步直接餵 residual，也就是 proposal slide 9 原本的設計。
**`residual` 模式結果（`runs/mem_v1_residual`，s16r16 資料，2 萬步，約 3.1 小時）。** 訓練 loss 從 0.0781 降到 0.0764。用 `pipeline/paired_test.py` 做 task 層級的配對檢定（給 context 減不給 context，同一個模型；先在每個 task 內平均，再對 100 個 task 算標準誤）：

| Episode | Latent L1 | z | 物體附近 latent L1 | z | Green 誤差 | z |
|---|---|---|---|---|---|---|
| k1 全部 | −0.00070 | −10.9 | −0.00023 | −4.1 | −0.18 px | −2.6 |
| k1 近距離 | −0.00086 | −12.3 | −0.00028 | −5.2 | −0.21 px | −2.9 |
| k1 遠距離 | −0.00066 | −10.3 | −0.00021 | −3.7 | −0.17 px | −2.4 |
| k2 全部 | −0.00089 | −12.1 | −0.00028 | −5.1 | −0.08 px | −1.5 |
| k4 全部 | −0.00102 | −13.4 | −0.00034 | −5.9 | −0.12 px | −1.6 |
| scorer_fail 近距離 | −0.00078 | −10.6 | −0.00027 | −4.0 | −0.27 px | −2.3 |
| scorer_fail 遠距離 | −0.00056 | −8.3 | −0.00013 | −2.3 | −0.03 px | −0.4 |

- **RQ1：** residual feedback 有真實效果，latent 誤差的改善非常顯著；`full` 模式（只給觀測 latent）則沒有。不給 context 時的 memory 模型與 baseline 相同（0.1030），所以改善來自 feedback，而不是額外的參數或訓練。
- **RQ2：** 近距離的改善在每一種 episode 都大於遠距離，scorer_fail 最明顯。
- **RQ3：** 1、2、4 次嘗試依序改善 0.0007、0.0009、0.0010。
- **但幅度只有約 0.7% 到 1%**，診斷估計的上限約 5%；排序 AUC 不變，還看不出對決策的幫助。

**用鄰近群集資料訓練的 `residual` 版本（`runs/mem_v1_residual_clusters`，51,161 條，2 萬步，約 4.5 小時）。** 訓練 loss 從 0.0956 降到 0.0916（群集資料本身比較難，數值不能跟 s16r16 直接比）。

| k1 episode（12,400 個 query） | Latent L1 | 物體附近 latent L1 | Green 誤差 | 排序 AUC |
|---|---|---|---|---|
| Baseline | 0.1029 | 0.2443 | 23.8 px | 0.610 |
| 群集訓練的 memory，不給 context | 0.1012 | 0.2449 | 24.9 px | 0.621 |
| 群集訓練的 memory，給 context | 0.0984 | 0.2393 | 20.7 px | 0.636 |

配對檢定（給 context 減不給 context，同一個模型，task 層級）：

| Feedback 的效果 | s16r16 訓練 | 群集訓練 |
|---|---|---|
| Latent L1（k1 全部） | −0.0007（z=−10.9） | −0.0028（z=−9.3） |
| 物體附近 latent L1 | −0.0002（z=−4.1） | −0.0056（z=−7.8） |
| Green 誤差 | −0.18 px（z=−2.6） | −1.7 px（z=−1.4，各 task 變異大） |

- 用與評估同分布、包含近距離配對的資料訓練，feedback 的效果放大 4 到 25 倍，物體附近約 2.3%。排序 AUC 第一次隨 feedback 改善（0.621 → 0.636）。
- **注意：** 群集訓練的模型在不給 context 時就與 baseline 不同（latent L1 較低、位置誤差較高），因為 memory 的空白 token 學到了針對這種分布的修正。所以 feedback 的效果必須用同一個模型給與不給 context 的差值衡量，不能直接跟 baseline 比。
- 近距離與遠距離的差距消失（k2 甚至遠距離較大）：這個模型學到的主要是場景共通的修正。

**解凍與高解析度（2026-09-27）。** 兩者都以 `residual` 為基準、用 s16r16 資料訓練，每組只改一個地方。`paired_test.py` 已加入排序 AUC 的逐 episode 配對檢定。Feedback 的效果（同一模型給與不給 context，k1 全部）：

| 版本 | 訓練資料 | Latent L1 | 物體附近 latent L1 | Green 誤差 | 排序 AUC |
|---|---|---|---|---|---|
| Residual | s16r16 | −0.0007（z=−10.9） | −0.0002（z=−4.1） | −0.18 px（z=−2.6） | +0.001（z=+0.6） |
| Residual + 解凍（backbone lr 3e-5） | s16r16 | −0.0007（z=−11.8） | −0.0002（z=−5.4） | −0.13 px（z=−2.6） | −0.001（z=−0.8） |
| Residual + 高解析度（16 個時間點、10×10、逐 attempt 編碼） | s16r16 | −0.0013（z=−13.6） | −0.0005（z=−7.7） | −0.16 px（z=−1.8） | −0.002（z=−1.6） |
| Residual | 群集 | −0.0028（z=−9.3） | −0.0056（z=−7.8） | −1.66 px（z=−1.4） | +0.015（z=+1.2） |

- **解凍沒有幫助**，效果與凍結版本相同；不給 context 時的輸出也與 baseline 相同，backbone 在低學習率下幾乎沒變。
- **高解析度讓效果約加倍**，且近距離再度明顯大於遠距離（latent 近 −0.0016、遠 −0.0012；k4 −0.0017）。
- **排序 AUC 都還不顯著**，群集版的 +0.015 各 task 變異很大。

**高解析度加群集資料（`runs/mem_v3_residual_hires_clusters`，slurm job 33526，約 3.6 小時）。** 第一次在 oprime 上直接跑時，因實驗室成員在同一張 GPU 開始訓練而記憶體不足，改用 slurm 重跑。

| Feedback 的效果（k1 全部） | Latent L1 | 物體附近 latent L1 | Green 誤差 | 排序 AUC |
|---|---|---|---|---|
| Residual，群集，低解析度 | −0.0028（z=−9.3） | −0.0056（z=−7.8） | −1.66 px（z=−1.4） | +0.015（z=+1.2） |
| Residual，群集，高解析度 | −0.0031（z=−10.4） | −0.0058（z=−7.8） | −1.62 px（z=−1.3） | +0.017（z=+1.3） |

- 用群集資料訓練後，高解析度只再多約 10%。
- **小結：** feedback 對預測的改善穩定且顯著（latent 相關 z 約 −8 到 −10），但第一層每個 task 只有 32 條 rollout，對排序的改善（約 +0.017）還看不出統計上的差異。下一步用第二層 closed-loop 直接測決策效用（§7 第 9 項）。



### 1.11 第二層 closed-loop 加 memory（2026-09-27，within fold 0）

**設定（`pipeline/closed_loop_memory.py`）：** 每個 dev task 用 baseline 分數選出前 512 名候選當共用 shortlist（`data/shortlist_within_f0_dev_top512.jsonl`；有解的 94 個 task 中 95.7% 在 shortlist 內仍有解）。為這 51,200 個 action 預先生成完整 rollout 並 encode（slurm）。迴圈：選分數最高、未試過的候選；失敗就在 feedback 模式下把它的 rollout 加進 memory（保留最近 4 次，因為 memory 訓練時最多 4 個 context），重新為 shortlist 打分；最多更新 10 次，之後排序固定，直到第 100 次嘗試。Baseline 重算的排序與 shortlist 原本排序的 Spearman 相關為 0.87 到 0.997（smoke test 為 0.997）。

| Within fold 0 dev（100 個 task） | AUCCESS@100 | S@1 | S@2 | S@3 | S@5 | S@10 |
|---|---|---|---|---|---|---|
| Baseline | 0.479 | 18% | 23% | 32% | 38% | 48% |
| 高解析度群集 memory，不給 feedback | 0.497 | 22% | 25% | 32% | 38% | 50% |
| 高解析度群集 memory，加 feedback | 0.520 | 22% | 33% | 36% | 46% | 55% |
| 低解析度群集 memory，不給 feedback | 0.500 | 22% | 27% | 34% | 39% | 49% |
| 低解析度群集 memory，加 feedback | 0.527 | 22% | 37% | 39% | 47% | 52% |

逐 task 配對（`pipeline/clm_compare.py`）：

| 比較 | ΔAUCCESS | z | 更早 / 更晚 / 相同 |
|---|---|---|---|
| 高解析度：加 feedback vs 不給 | +0.023 | +1.34 | 29 / 22 / 49 |
| 低解析度：加 feedback vs 不給 | +0.027 | +1.51 | 31 / 20 / 49 |
| 低解析度加 feedback vs baseline | +0.048 | +2.05 | 37 / 27 / 36 |
| 高解析度加 feedback vs baseline | +0.041 | +1.92 | 40 / 22 / 38 |

- Feedback 的效果在兩個 memory 模型上都是正的（約 +0.025），從第 2 次嘗試開始出現（S@2 +8 到 +10 個百分點），但 100 個 task 下單獨看不顯著（z 1.3 到 1.5）。
- 與 baseline 相比約 +0.04 到 +0.05（z 約 2），其中約 0.018 來自 memory 模型本身（不給 feedback 時就較高）。
- 0.527 與同資料的 initial scene 模型（0.523）持平，仍低於 DQN 規模（0.608）。
- 要達到顯著：增加 task 數（within fold 0 完整 400 個 dev task、多個 fold）或多個 seed。
- Cross-template 的 memory 與 closed-loop（shortlist 1,024，因為 512 只保留 71% 的解）已送出 slurm（job 33580 到 33599）。

**Adaptive 對照組（2026-09-27，同一份 512 shortlist）。** AdaJEPA 式 test-time gradient（`closed_loop_memory.py --tta-steps 20`，每個 task 複製一份 baseline predictor，每次失敗後對所有失敗 rollout 的 latent L1 做 20 步 Adam，lr 1e-4）；DQN online 式（`pipeline/closed_loop_dqn_online.py`，每個 task 複製一份 initial scene 模型，每次失敗後以該失敗樣本加 32 筆 train cache 平衡回放樣本微調 10 步，lr 1e-4）。最多 10 次更新。

| 方法 | 固定排序 | 加 feedback | Adapt 效果 | z | 更早 / 更晚 |
|---|---|---|---|---|---|
| World model，test-time gradient | 0.479 | 0.500 | +0.021 | +0.78 | 24 / 32 |
| World model，memory（低解析度群集） | 0.500 | 0.527 | +0.027 | +1.51 | 31 / 20 |
| World model，memory（高解析度群集） | 0.497 | 0.520 | +0.023 | +1.34 | 29 / 22 |
| Initial scene，同樣資料，DQN online | 0.539 | 0.559 | +0.020 | +0.72 | 35 / 25 |
| Initial scene，DQN 規模，DQN online | 0.626 | 0.592 | −0.034 | −1.43 | 22 / 29 |

- Memory 的 adapt 效果最大且最穩定；test-time gradient 平均有進步，但更早解出的 task 比更晚的少。
- Initial scene 模型也能受益，但不穩定：強模型用少數失敗樣本微調反而退步。
- Within 上 initial scene 模型仍然最好（同資料 online 0.559、DQN 規模固定 0.626），world model 最好的是 0.527。
- Initial scene 模型限制在 world model 的 shortlist 內反而更好（同資料 0.523 → 0.539），world model 的 shortlist 本身有用。
- **[推論]** Within 上 adaptation 約帶來 +0.02 到 +0.03，不足以追上記憶版面的 initial scene 模型；真正的機會在 cross。


**擴大到 within fold 0 全部 400 個 dev task（2026-09-28）。** 新增 300 個 task 的 shortlist（前 512 名，解保留比例 96.2%）、153,600 條 rollout；所有方法在這 300 個 task 上重跑（slurm job 33676 到 33691），再用 `pipeline/merge_400.py` 與原本 100 個 task 合併。

| Within fold 0 dev（400 個 task） | AUCCESS@100 | S@1 | S@2 | S@5 | S@10 |
|---|---|---|---|---|---|
| Baseline world model | 0.479 | 18% | 25% | 34% | 50% |
| + test-time gradient | 0.512 | 18% | 27% | 41% | 54% |
| Memory（低解析度群集），不給 feedback | 0.499 | 20% | 28% | 40% | 50% |
| Memory（低解析度群集），加 feedback | 0.536 | 20% | 34% | 49% | 57% |
| Memory（高解析度群集），不給 feedback | 0.499 | 20% | 27% | 40% | 50% |
| Memory（高解析度群集），加 feedback | 0.530 | 20% | 32% | 47% | 57% |
| Initial scene，同樣資料，固定 | 0.560 | 25% | 32% | 45% | 55% |
| Initial scene，同樣資料，DQN online | 0.546 | 25% | 35% | 47% | 57% |
| Initial scene，DQN 規模，固定 | 0.649 | 37% | 45% | 60% | 67% |
| Initial scene，DQN 規模，DQN online | 0.608 | 37% | 45% | 56% | 63% |

| 配對比較（400 個 task） | ΔAUCCESS | z | 更早 / 更晚 |
|---|---|---|---|
| Memory（低）：加 feedback vs 不給 | +0.038 | +4.12 | 138 / 74 |
| Memory（高）：加 feedback vs 不給 | +0.030 | +3.36 | 130 / 85 |
| Memory（低）加 feedback vs baseline | +0.057 | +5.08 | 160 / 97 |
| Test-time gradient vs baseline | +0.033 | +2.73 | 120 / 108 |
| Memory（低）加 feedback vs test-time gradient | +0.024 | +1.75 | 145 / 115 |
| Initial scene 同資料：online vs 固定 | −0.015 | −1.16 | 122 / 123 |
| Initial scene DQN 規模：online vs 固定 | −0.041 | −3.32 | 86 / 124 |
| Memory（低）加 feedback vs initial scene 同資料 online | −0.010 | −0.52 | 167 / 150 |

- **Feedback 的效果在 within 上明確顯著**（+0.030 到 +0.038，z 3.4 到 4.1）。100 個 task 時 z 只有 1.3 到 1.5，主要是 task 數不足。
- **Memory 是唯一穩定受益於 feedback 的方法。** Test-time gradient 也有進步（+0.033 vs baseline），但比 memory 小，更早與更晚解出的 task 幾乎各半；initial scene 模型的 online 微調反而退步，DQN 規模顯著退步。
- **跟 initial scene 模型比：** memory 加 feedback（0.536）與同資料 initial scene 模型的 online 版本（0.546）統計上打平，但仍低於同資料固定排序（0.560）與 DQN 規模（0.649）。

### 1.12 Cross-template 加 memory（2026-09-28，fold 1）

**設定：** 在 fold 1 cross 的 train task 上生成鄰近群集資料（`ball_cross_template_f1_train_clusters`），以 cross 的 base predictor 為基礎訓練 residual 加高解析度的 memory（`runs/mem_cross_f1_residual_hires_clusters`）。Shortlist 用 1,024（512 只保留 71.1% 的解，1,024 為 86.6%，2,048 為 99.0%；每個 task 的候選池約 3,800）。整條流程是 20 個 slurm job。

| Cross fold 1 dev | AUCCESS@100 | S@1 | S@2 | S@5 | S@10 |
|---|---|---|---|---|---|
| Baseline world model | 0.454 | 27% | 34% | 42% | 47% |
| Memory，不給 feedback | 0.363 | 25% | 28% | 34% | 37% |
| Memory，加 feedback | 0.394 | 25% | 33% | 37% | 42% |

| 比較 | ΔAUCCESS | z | 更早 / 更晚 / 相同 |
|---|---|---|---|
| Feedback：加 vs 不加（同一模型） | +0.031 | +2.80 | 16 / 4 / 80 |
| Memory 不給 feedback vs baseline | −0.091 | −4.08 | 9 / 31 / 60 |
| Memory 加 feedback vs baseline | −0.060 | −3.01 | 14 / 25 / 61 |

各 template（baseline / 不給 / 加 feedback）：00000 0.60 / 0.53 / 0.56、00002 0.88 / 0.85 / 0.92、00007 0.00 / 0.00 / 0.00、00014 0.34 / 0.07 / 0.09。

- **Feedback 的效果在 cross 上第一次達到顯著**（+0.031，z=2.80），更早解出的 task 是更晚的 4 倍。
- **但 memory 模型本身把 baseline 拖垮**，主要在 00014。**[推論]** memory 的修正裡有一部分是在訓練 templates 上學到的分布偏差：within 上 dev 與訓練 templates 相同所以有幫助，cross 上則有害。
- **修正：** `MemoryPredictor` 新增 `gate_empty`，沒有 context 的樣本完全跳過 memory 的修正，保證「不給 feedback」等於 baseline。有了 context 之後偏差仍可能出現，所以同時測兩組：(a) 已訓練模型只在推論時開 gate（`cx_clm_fb_gateinf`），(b) 訓練時就開 gate 重新訓練（`mem_cross_f1_residual_hires_clusters_gate` → `cx_clm_fb_gate`）。slurm job 33671 到 33673。

**Gate 的結果（2026-09-28）：沒有解決問題。**

| Cross fold 1 | AUCCESS | vs baseline | z | S@1 | S@2 | 00014 |
|---|---|---|---|---|---|---|
| Baseline | 0.454 | — | — | 27% | 34% | 0.34 |
| Memory 加 feedback，無 gate | 0.394 | −0.060 | −3.01 | 25% | 33% | 0.09 |
| 推論時開 gate | 0.407 | −0.047 | −2.78 | 27% | 37% | 0.13 |
| 訓練時開 gate | 0.407 | −0.048 | −2.84 | 27% | 36% | 0.14 |

- Gate 只讓第一次嘗試回到 baseline（S@1 27%），S@2 也較好；有了 context 之後整體仍然變差，幾乎全部來自 00014。00002 上 memory 加 feedback 則贏過 baseline（0.90 到 0.92 vs 0.88）。
- **[推論]** 問題不是「沒有 context 時的偏差」，而是 memory 在有 context 時學到的修正方式，在部分沒看過的 templates 上不適用。
- 下一步（slurm job 33728 到 33738）：cross 上的 adaptive 對照組（test-time gradient、initial scene 固定與 online），以及把 memory 輸出縮放為 0.5 和 0.25 的診斷（用 cross dev，只當診斷）。

**Cross 上的 adaptive 對照組與縮放診斷（2026-09-28，slurm job 33728 到 33738，shortlist 1,024）。**

| Cross fold 1 dev | 固定排序 | 加 feedback | Adapt 效果 | z | 更早 / 更晚 |
|---|---|---|---|---|---|
| World model，test-time gradient（20 步） | 0.454 | 0.508 | +0.053 | +2.70 | 20 / 15 |
| World model，memory（訓練時開 gate） | 0.454 | 0.407 | −0.048 | −2.84 | 10 / 22 |
| Initial scene，DQN 規模，DQN online | 0.393 | 0.446 | +0.053 | +2.44 | 26 / 11 |
| Initial scene，同樣資料，DQN online | 0.374 | 0.420 | +0.047 | +2.26 | 26 / 17 |

Memory 輸出縮放（`--mem-scale`，只在推論時，cross dev 上的診斷）：1.0 → 0.407（−0.048）、0.5 → 0.429（−0.025，z=−2.52）、0.25 → 0.450（−0.004，z=−0.65）。

各 template（baseline / test-time gradient / memory gate / 縮放 0.25 / initial scene 同資料 online / DQN 規模固定）：00000 0.60 / 0.66 / 0.58 / 0.60 / 0.36 / 0.55、00002 0.88 / 0.89 / 0.90 / 0.88 / 0.83 / 0.85、00007 0.00 / 0.04 / 0.00 / 0.00 / 0.31 / 0.12、00014 0.34 / 0.44 / 0.14 / 0.32 / 0.18 / 0.06。

World model 固定排序對 initial scene 固定排序：對同資料 +0.081（z=1.89）、對 DQN 規模 +0.061（z=1.80）。

- **Cross 上最好的是 test-time gradient**（0.508），00014 從 0.34 升到 0.44。
- **Memory 在任何強度下都沒有幫助**：修正越小越接近 baseline，但不會超過。問題不是修正得太猛，而是學到的修正方式無法轉移到新 templates。
- **Initial scene 模型在 cross 上受益於 online 微調**（+0.047 到 +0.053），與 within 上退步相反，但絕對分數仍低於 world model。
- 00007 上所有 world model 方法都接近 0，initial scene 同資料 online 則有 0.31。

### 1.13 目前的總結（2026-09-28）

**AUCCESS@100，每一格是該設定下的最佳版本：**

| 方法 | Within fold 0（400 個 task） | Cross fold 1（100 個 task） |
|---|---|---|
| 隨機排序 | 約 0.16 | 0.274 |
| Initial scene，DQN 規模，固定 | **0.649** | 0.393 |
| Initial scene，DQN 規模，DQN online | 0.608 | 0.446 |
| Initial scene，同樣資料，固定 | 0.560 | 0.374 |
| Initial scene，同樣資料，DQN online | 0.546 | 0.420 |
| World model，固定 | 0.479 | 0.454 |
| World model，test-time gradient | 0.512 | **0.508** |
| World model，memory 加 feedback | 0.536 | 0.407 |

**Adapt 的效果（同一模型，加 feedback 減固定）：**

| | Within（400） | Cross（100） |
|---|---|---|
| 學出來的 adaptation（memory） | +0.038（z=4.1） | −0.048（z=−2.8，相對 baseline） |
| Gradient 式 adaptation（test-time gradient） | +0.033（z=2.7） | +0.053（z=2.7） |
| Initial scene 的 online 微調 | −0.015 到 −0.041 | +0.047 到 +0.053 |

**結論 [推論]：**

1. 在看過的 templates 上，initial scene 模型靠記憶版面最強；world model 靠 adaptation 可以縮小差距，但追不上 DQN 規模。
2. 在沒看過的 templates 上，world model 本身就贏過 initial scene 模型，加上 test-time gradient 之後差距更大。
3. 學出來的 memory 在訓練分布內是最有效的 adaptation，但換到新 templates 就失效；gradient 式的 adaptation 不依賴學過的修正模式，所以能泛化。這正好回答 proposal 要比較的「學出來的 adaptation 對 AdaJEPA 式 gradient adaptation」。
4. 主要限制：cross 只有 fold 1、4 個 dev templates、一個 seed；within 的 memory 訓練資料與 dev 分布相同。

**建議的下一步（待使用者決定）：** (1) 在 fold 0、2 重複 cross（成本高，每個 fold 約一天）；(2) memory 加 test-time gradient 的混合版（便宜）；(3) 用更多 templates 訓練 memory，檢查泛化問題是否來自訓練 templates 太少；(4) recurrent memory 往後放。

---

## 2. Scorer 設計

World model 本身不會 rank。它只輸出 predicted latent，要另外有一個 scorer 把 latent sequence 變成分數。

### 2.1 主 scorer：錨定在 encoder 空間的 success head

- **輸入：** 整段 latent sequence，用 attention pooling 聚合，不只看最後一幀。PHYRE 的成功條件是持續接觸 3 秒，所以時間資訊是必要的。**[已驗證]** `simulator/task_utils.h:33` 定義 `kStepsForSolution = 3 * kFps`，也就是 180 個物理步。
- **訓練資料：** 用凍結的 V-JEPA encoder 對 **train templates 真實 rollout** 編碼出來的 latent。
- **Robustness：** 加 noise augmentation，或混入少量 base predictor 的 predicted latent，讓 head 對模糊的預測不要太敏感。
- **輔助 dense target：** 同時回歸「整段 rollout 中 green ball 跟 goal object 的最小距離」，simulator 可以直接給。AUCCESS 看排序，binary label 分不出「差一點」和「差很遠」。
- **訓練完就凍結，**所有 memory 變體共用同一個 head。

**為什麼 predictor 被 tune 之後 head 仍然適用。[推論]** Encoder 是凍結的。Predictor 不管怎麼 tune、加什麼 memory，訓練目標都是逼近 encoder 對真實未來的 latent。Head 學的是「在 encoder 空間裡成功的未來長什麼樣」，這跟 predictor 無關。Predictor 越準，輸入就越接近 head 訓練時的分布。Memory 改的是「預測哪一個未來」，不改「成功長什麼樣」。

**不要只用 predicted latent 訓練 head。** 那樣 head 會學到某一個 predictor 特有的誤差模式，換成別的 memory 變體就可能失準。

### 2.2 替代 scorer：state probe 加規則

- 用 train templates 訓練一個 probe，從 latent 讀出 green ball 和 goal object 的位置。
- 再用 PHYRE 的成功條件直接判斷，也就是有沒有持續接觸，或算出 dense 距離分數。
- **優點：** 完全不碰 test data。規則只看預測出來的狀態，所以沒有 shortcut。位置回歸也可能比「成功長相」更容易跨場景 generalize。**[推論]**
- **缺點：** probe 本身有誤差，接觸判定對小誤差很敏感。

### 2.3 不建議：goal latent distance

V-JEPA 2-AC 用 predicted latent 跟 goal image latent 的距離打分。PHYRE 沒有 goal image，成功的最終狀態也不只一種。從已知解法造 goal 會 leak GT。

---

### 2.4 Encoder：比較與 fine-tune

#### 2.4.1 原則：encoder 在 world model 訓練前就凍結

§2 的 scorer 設計、§3 的歸因方法，都依賴 encoder 空間固定不動。所以 encoder 不能跟 predictor 或 memory 一起訓練：

- **Scorer 會失效。** Success head 錨定在 encoder 空間，encoder 一動，head 就過期。
- **Collapse 風險。** Encoder 跟 predictor 同時更新時，encoder 可以把表示變得很好預測。Prediction error 會變低，但資訊也跟著丟了。
- **歸因會亂。** Memory 變體之間的差異，會分不清是 predictor 還是 encoder 造成的。

順序固定為：（選擇性）fine-tune encoder → 凍結 → 訓練 scorer 和 predictor → 訓練 memory。

#### 2.4.2 候選 encoder

| 候選 | 做法 | 備註 |
|---|---|---|
| A | V-JEPA 2，凍結 | Masked prediction 直接訓練 patch tokens |
| B | [LeVJEPA](https://levjepa.github.io/)-VideoMix-Large，凍結 | 見下方說明 |
| C | LeVJEPA-L，在 PHYRE rollout 上繼續 pretrain | Repo 支援 `resume.ckpt_path` |
| D | 小的 LeVJEPA encoder，在 PHYRE 上從頭訓練 | README 說 ViT-Tiny 在 16GB 單卡 12 小時就有明顯效果 |

LeVJEPA 的已知事實（2026-09-25 查 project page、GitHub README、HuggingFace model card）：

- 單一 encoder，用 crop 之間的 invariance loss 加 SIGReg 訓練。沒有 predictor、沒有 EMA target，也**沒有 action conditioning**。它只能當 encoder，不能當 world model。
- 釋出的 ViT-L/16 有 303M 參數，輸入 16 幀、224 px，輸出 CLS 加 16×14×14 個 patch tokens。訓練時取樣大約 7.5 fps，16 幀涵蓋約 2 秒。
- 跨幀是 block-causal attention：每一幀的表示只依賴過去的幀，適合 autoregressive world model 跟逐幀 encode 真實 rollout。
- 程式碼是 MIT，權重是 CC BY-NC 4.0。
- 論文說在 5.6–20.8 倍更少的 pretraining compute 下追平或超過 V-JEPA 2。**[未查證]** 這是論文自述，我們沒有重現。

選 encoder 對 world model 這邊沒有損失：V-JEPA 2-AC 的 action-conditioned predictor 是在 DROID 機械手臂資料上訓練的，不能直接用在 PHYRE。不管選哪個 encoder，predictor 都要自己訓練。

**B 的風險。[推論]**

- **位置資訊。** 訓練目標要求不同 crop 的表示一致，SIGReg 只加在 CLS 上。PHYRE 需要精確的位置跟接觸，crop invariance 可能讓表示不在乎絕對位置。README 說 patch tokens 會自然出現語意結構，但沒有證據說它保留精確座標。
- **小物體。** 224 px、patch 16 的網格是 14×14，每個 patch 大約對應 PHYRE 256 px 畫面上的 18 px。細的物體可能被吃掉。V-JEPA 2 也有類似問題。
- **很新。** Repo 在 2026-08-28 建立，被驗證的程度比不上 V-JEPA 2。

#### 2.4.3 比較方法

四個候選都凍結，在 train templates 真實 rollout 的 latent 上訓練，在 within 與 cross 的 val 上比較：

1. **位置 probe 誤差：** 從 latent 讀出 green ball 與 goal object 的位置。
2. **Success head 準確率：** 同 §2.1 的設定。

這一步不需要 predictor，就是 §3 sanity check 的延伸。先跑 A 和 B。其中一個及格，就不用 fine-tune。都不及格才開 C 或 D。

**及格線（2026-09-25 提出，要在看到結果前定案）。** 原則是用現成的東西當參照，不憑空訂數字。

| 項目 | 參照組 | 及格條件（cross-template val） |
|---|---|---|
| Success readout | 同樣架構的 classifier，輸入換成 simulator 給的物體座標軌跡。這代表可讀資訊的上限 | AUC ≥ 0.95，而且跟參照組差距 ≤ 0.03 |
| 位置 probe | 「直接用初始位置」baseline，也就是假設物體不動 | Green ball 位置誤差中位數 ≤ 9 px（半個 patch），而且不超過 baseline 中位數的一半 |

- **為什麼 success 的門檻訂這麼高：** 輸入是真實 rollout，影片裡直接看得到 green ball 有沒有碰到 goal。Encoder 如果連這個都讀不出來，代表它丟掉了任務需要的資訊，後面的 predictor 不可能補回來。
- **為什麼位置門檻是 9 px：** 一個 patch 大約 18 px（§1.2.1）。誤差超過半個 patch，就很難判斷接觸。
- **為什麼要比初始位置 baseline：** 很多物體整段 rollout 幾乎不動。只看絕對誤差，一個只會輸出初始位置的 probe 也可能看起來很準。
- **判定：** 兩項都在 cross val 及格才算過，within 一起報。A、B 都過，選 success AUC 較高的。都不過才做 C 或 D。
- 位置誤差用 PHYRE 256 px 畫面上的 pixel 計算，所有幀一起算。
- 「不超過 baseline 中位數的一半」是 2026-09-25 在跑第三步之前補上的量化定義，原本只寫「明顯低於」。

**2026-09-25 修訂（在跑第三步之前定案）。** §1.4 的結果顯示，學習式 success head 就算輸入 simulator 座標，在 cross dev 上也只有 0.75 到 0.95，所以上面 success readout 那一列的絕對門檻不再適用。主線 scorer 改成 probe 加規則，第三步的判準也跟著改：

| 項目 | 定義 |
|---|---|
| 主要判準：端到端 AUC | 真實 latent → 物體位置 probe → 規則（線段距離）→ dev AUC |
| 參照組 | 同一個規則直接用 simulator 座標，時間解析度相同（32 個時間點，每 4 幀一個） |
| 及格條件 | Fold 0 cross dev 上，端到端 AUC ≥ 參照組 − 0.03。Within 一起報 |
| 規則參數 | 硬門檻（最長連續接觸）和軟分數（Σ sigmoid((τ − d)/σ)）兩類，參數在 train 子集切出的 20% 校準集上選，校準集不參與 probe 訓練。參照組用同樣的方式選 |
| 診斷 | Green ball 位置誤差中位數，以及「直接用初始位置」baseline。原本的 9 px 門檻保留作參考，不再決定及格 |
| 比較 | Encoder A、B 各自 × 14×14 完整解析度 vs 池化 7×7 |

Probe 以物體為 query：用物體在初始場景的特徵產生 query，對該時間點的空間 tokens 做 cross-attention，輸出 (x, y) 和長條的角度。訓練目標是所有會動的物體（紅、綠、藍、灰）；靜態物體（黑、紫）直接用初始場景。A 和 B 都及格時，選 cross 端到端 AUC 較高的。

**第三步結果（2026-09-25，fold 0）。[已驗證]** `pipeline/encode_latents.py`（凍結、重疊視窗、32 個時間點）加 `pipeline/probe_rule.py`。Probe 在 400 task 子集的 80% 上訓練（約 1,280 條 rollout），規則參數在其餘 20% 的校準集上選，dev 各 400 條。

| Encoder | 解析度 | Cross 端到端 AUC | Within 端到端 AUC | Cross green 誤差中位數 / p90（px） | Within green 誤差中位數 / p90（px） |
|---|---|---|---|---|---|
| 參照：規則用 simulator 座標 | — | 0.972 | 0.978 | — | — |
| B LeVJEPA | 14×14 | 0.885 | 0.967 | 6.5 / 14.3 | 3.5 / 8.2 |
| B LeVJEPA | 7×7 | 0.861 | 0.955 | 7.8 / 19.6 | 4.7 / 13.4 |
| A V-JEPA 2 | 16×16 | 0.829 | 0.977 | 4.8 / 12.7 | 2.8 / 6.8 |
| A V-JEPA 2 | 7×7 | 0.772 | 0.949 | 11.8 / 74.2 | 5.4 / 14.1 |

「直接用初始位置」的 baseline 中位數：cross 131 px、within 121 px。

- **依事先定好的判準，A 和 B 在 cross 上都不及格。** 門檻是 0.972 − 0.03 = 0.942。最好的是 LeVJEPA 14×14 的 0.885。Within 上四組都及格。
- **位置誤差中位數小，端到端 AUC 卻明顯下降。** Cross 上中位數只有 5 到 7 px，但規則的接觸門檻是 2 到 4 px，p90 已經到 13 到 14 px。規則對接觸附近的誤差非常敏感。
- **V-JEPA 2 的位置中位數比較準，端到端卻比較差。**原因還沒查。
- **池化到 7×7 一致地變差。** LeVJEPA 掉 2 到 3 個點，V-JEPA 2 掉 3 到 6 個點，而且 V-JEPA 2 7×7 的 p90 跳到 74 px。
- **限制：** 只有 fold 0；probe 只用了約 1,280 條 rollout 訓練；只跑一個 seed。

依 §2.4.3 的規則，下一步是 C 或 D。這是新的一大段工作，要先跟使用者確認。

**第三步的兩個追加檢查（2026-09-25，fold 0，LeVJEPA 14×14 為主）。[已驗證]**

*檢查 1：誤差分析。* 用同一個 seed 重跑，AUC 完全重現。新增 task 內 AUC，只比同一個 task 裡的成功和失敗，這比較接近實際排序的情況：LeVJEPA cross 0.942（全域 0.885），V-JEPA 2 cross 0.838，參照組 cross 0.985。位置誤差跟接觸距離、速度、時間點的關係都不大。V-JEPA 2 在 cross 上把成功打成低分的 rollout 是 LeVJEPA 的兩倍多（23 vs 10 條）。

*檢查 2：每個 task 取 8 個成功、8 個失敗，用最遠點取樣讓 action 分散。* 訓練資料約為原本的 4 倍，manifest 是 `*_f0_train_sub400_div16`，latent 放在 `/mnt/meg/jjenchen/phyre-unc/latents/`。

| LeVJEPA 14×14 | 原本 4 條 / task | 多樣 16 條 / task | 參照組 |
|---|---|---|---|
| Cross green 誤差中位數 / p90 | 6.5 / 14.3 px | 2.8 / 6.1 px | — |
| Cross 端到端 AUC（全域 / task 內） | 0.885 / 0.942 | 0.913 / 0.917 | 0.972 / 0.985 |
| Within green 誤差中位數 / p90 | 3.5 / 8.2 px | 1.4 / 3.3 px | — |
| Within 端到端 AUC（全域 / task 內） | 0.967 / 0.980 | 0.978 / 0.990 | 0.979 / 0.995 |

多樣資料之後，within 已經追上參照組；cross 仍未達 0.942 的門檻。Task 內 AUC 每個 task 只有 4 條 rollout，雜訊很大，一條 rollout 就會改變 2 組配對。

*Cross 差距的兩個來源：*

1. **Template 00000 的藍色 goal 球讀不準。** 它是會動的，要靠 probe 讀。誤差中位數 11 到 15 px，多樣資料幫助有限；green ball 同時只有 2.6 px。Cross 的 16 個 train templates 裡只有 3 個有會動的藍色 goal（00006 罐、00022 和 00023 球），probe 很少看到這類物體。成功被打成低分的 12 條 rollout 全部來自 00000。
2. **Template 00011 是規則參數太緊。** 在真實座標上，規則算出成功 rollout 的最小距離是 2.6 px（幾何近似的偏差），probe 讀出 3 到 7 px，但校準集選出的門檻只有 2 px。在 dev 上改成軟分數 τ=8、σ=4，00011 從 0.760 升到 0.996，整體 cross 從 0.913 升到 0.936，參照組不受影響（0.973）。**這是用 dev 選參數的診斷，不能當成績。**

*含意。[推論]* 兩個來源都不是 encoder 精度的問題，所以目前沒有證據需要訓練 encoder（C 或 D）。比較直接的修正是：（a）校準集改成依 template 切分，讓選出的規則參數反映跨 template 的誤差；（b）增加 probe 訓練資料中會動的 goal 物體，例如程序化隨機場景，或把 dev templates 併進 probe 的訓練。**Fold 0 的 cross dev 已經被用來做診斷，之後的判斷要在 fold 1、2 上驗證。**

Script：`pipeline/analyze_errors.py`、`pipeline/goal_errors.py`、`pipeline/rule_sensitivity.py`、`pipeline/render_fn.py`。結果：`results/check1_*.json`、`results/check2_*.json`、`results/check2_fn_examples.png`。

**定案與乾淨驗證（2026-09-25）。** 使用者決定：encoder 定為 LeVJEPA、14×14、重疊視窗、不訓練；scorer 定為 probe 加規則，probe 用多樣資料（每個 task 8 成功 8 失敗）訓練；規則參數改用依 template 的 4 組交叉驗證校準（16 個 train templates 分 4 組輪流：12 個訓練 probe、4 個產生「沒看過」的預測；有會動藍色 goal 的 templates 分散到不同組），選完參數後最終 probe 用全部 16 個 templates 訓練。

| Cross dev | 依 task 校準 | 依 template 校準 | 參照組 | 及格線 | 說明 |
|---|---|---|---|---|---|
| Fold 0 | 0.913 | 0.949 | 0.976 | 0.946 | 已被診斷用過，只當參考 |
| **Fold 1** | — | **0.991** | 0.998 | 0.968 | **乾淨驗證，通過** |

- Fold 1 的 task 內 AUC 0.993（參照組 1.000），green 位置誤差中位數 1.95 px。規則校準選出軟分數 τ=4、σ=2。
- **各 template（端到端 / 參照組）：** Fold 0 為 00000 0.985 / 0.994、00008 0.938 / 0.938、00011 0.881 / 1.000、00018 1.000 / 1.000。Fold 1 為 00000 0.949 / 0.994，00002、00007、00014 都是 1.000 / 1.000。
- **剩下的弱點是 template 00000，**它的 goal 是會動的藍球，兩個 fold 都是最低的。
- Fold 2 沒有跑。結果檔 `results/tcv_f0_cross.json`、`results/tcv_f1_cross.json`；probe 權重 `results/probes/`。

#### 2.4.4 如果要 fine-tune

1. **只用 train templates 的 rollout。** Encoder 看過 test templates 的影片，cross-template 的結果就不乾淨。這是最容易不小心違反的一條。
2. **用自監督目標，不用 success label。** 用 success 做 supervised fine-tune 會把決策資訊燒進 encoder，「feedback 讓預測變好」的故事就講不清楚了。用 simulator 位置當輔助監督可以考慮，但它改變研究前提，不要一開始就用。
3. **先查位置資訊是被什麼丟掉的。[推論]** 如果 B 的位置 probe 很差，原因可能是 crop invariance 這個目標本身。那用同樣方法繼續訓練的 C 大概也救不回來，要改訓練設定，例如拿掉 local crops，或者改選 A。
4. **成本。** LeVJEPA 訓練只有 encoder 前向，而且會丟掉 95% 的 tokens，比 fine-tune V-JEPA 2 便宜很多。V-JEPA 2 還要跑 EMA target 和 predictor。C 和 D 在 lab 的 A6000 或 H100 上應該都可行。

---

## 3. 分辨 head 沒 generalize 還是 predictor 沒效

Head 只能用 train templates 訓練，val 拿來調參。在 test 上**評估**時可以換掉輸入 latent，這不算 leakage，只要不拿 test 結果回頭改模型。

| 餵給 head 的 latent | 量到的東西 |
|---|---|
| Encoder 對真實 test rollout 的 latent | 只有 head 的 generalization，等於假設 predictor 完美 |
| 該變體的 predicted latent | Head 加 predictor 的整體表現 |
| （不經過 head）latent prediction error | Predictor 本身，最乾淨的指標 |

判讀方式：

- **真實 latent 上就很差：** head 沒 generalize。整個 pipeline 的上限被 head 卡住，predictor 再好也沒用。
- **真實 latent 上好，predicted 上差：** 是 predictor 的問題。
- **兩邊都好：** pipeline 正常，可以開始比 memory 變體。
- **穩定性檢查：** 如果 head 在真實 latent 跟 predicted latent 上的差距，會隨 prediction error 下降而縮小，就代表 head 是穩的。

Within-template 和 cross-template 各做一次。**[推論]** 我預期 cross-template 最容易出問題的是 head。

**第一個 sanity check：** 在訓練任何 memory 之前，先確認 head 在 cross-template val、真實 latent 輸入時的表現。這一步不需要 predictor。如果這一步就不行，先換 probe 加規則，或加 augmentation。不然後面所有 memory 實驗都會被 head 汙染。

這一節對應 PROPOSAL_REFRAMED.md §7.2 的 representation/readout 跟 world-model gap 診斷。

---

## 4. Oracle scorer：用 test rollout 訓練的 head

研究問題是 predictor 能不能 adapt，scorer 只是量測工具。用 test templates 的真實 rollout 訓練一個 oracle head，可以拿掉 head generalization 這個 confound。

**可以做，但它是 diagnostic，不是 baseline 方法：**

- 表格上明確標成 **oracle scorer (test-trained)**，跟 train-only head 並列。
- 絕對 AUCCESS 不能當成方法的成績，也不能跟其他 PHYRE 論文比。
- 所有 memory 變體共用同一個 oracle head。所以變體之間的**相對**比較仍然公平。
- 最有說服力的結果：train-only head、oracle head、probe 加規則三種 scorer 下，memory 變體的排名一致。

### 4.1 Shortcut 風險

Predicted latent sequence 的第一幀已經包含場景跟紅球的位置。在 test 場景上訓練的 head 可能直接記住「這個場景、球放這裡會成功」，完全不讀後面的動態。這樣 memory 改了 predicted latent 的後段，分數也不會變，**adaptation 就量不到了**。

這種 shortcut 是洩題造成的副作用，不代表 rollout 沒意義。部署到沒看過的場景時，不會有這張答案表。

檢查與防範：

1. **Initial-frame ablation：** 只給 oracle head 第一幀的 latent。如果它就能準確判斷成功，代表在走 shortcut。
2. **限制輸入：** 只讓 head 看後段幀，例如後半段或最後 K 幀，這些一定得靠 predictor 滾出來。
3. **擾動測試：** 把 predicted latent 換成 shuffle 過或加大 noise 的版本。Oracle head 的表現應該明顯下降，沒掉就代表它沒在讀動態。

---

## 5. Baseline：initial-frame classifier（no-rollout）

如果 **train-only** head 在沒看過的場景上，光看第一幀就能判斷成功，這就是真的發現：不需要模擬動態，看場景加動作就夠了。PHYRE 原本的 DQN baseline 就是這種設計，直接吃初始畫面加 action，輸出成功機率。

所以要加這個 baseline，才能證明 rollout 有價值：

| 條件 | Scorer 的輸入 | 訓練資料 |
|---|---|---|
| Initial-frame classifier | 只有第一幀，也就是場景加紅球 | Train templates |
| Initial-frame classifier + feedback | 第一幀加過去 attempts 的資訊 | Train templates |
| 本方法 | Predictor 滾出來的整段 latent，predictor 讀 memory | Train templates |

- 本方法要贏過 initial-frame classifier，才能說 rollout 有意義。**[推論]** 我預期差距主要會出現在 cross-template。
- 比較「+ feedback」版本跟本方法的 adapt 幅度。如果 predictor 加 memory 的進步比較大，就能說 adaptation 發生在動態預測這一層，不只是分類器記住「剛剛那裡失敗」。

這對應 PROPOSAL_REFRAMED.md §7.2 表格中的 "No-rollout feedback scorer"。

---

## 6. 條件總表（2026-09-26 改寫）

Scorer 已定為 probe 加規則（§2.4.3），baseline predictor 已定為 base 大小加 52,349 條（§1.7.6）。舊版表中的學習式 success head、oracle head 項目已不適用。

| 類別 | 條件 | 回答的問題 |
|---|---|---|
| 核心 | Baseline predictor，不給 feedback | Adaptation 的起點 |
| 核心 | Memory 版本（context memory），給 feedback | Feedback 有沒有讓未試過的 action 預測得更準（RQ1–RQ3 的主體） |
| RQ1 內容 | Feedback 只給 action／只給成功或失敗／完整 latent／latent 加 residual | 哪一種資訊有用 |
| RQ2 轉移 | Context 與 query 是近距離（同群）vs 遠距離（不同群） | 哪些 attempt 的資訊能轉移 |
| RQ3 累積 | 1 次 vs 多次 attempt；只給最近一次 vs 全部 | 資訊能不能累積 |
| 對照 | 會 adapt 的 initial scene 模型（DQN online 式微調） | 公平的 adaptive baseline |
| 對照 | AdaJEPA 式 test-time gradient | 參數更新式 adaptation |
| 擴展 | Recurrent memory | 固定容量能保留多少 |
| 泛化 | 以上核心條件在 cross-template 上 | 沒看過的 templates |

**指標（§1.7.6）：** 同一個 base 有無 feedback 的 latent 誤差差值、會動物體附近的 latent 誤差、probe 讀出的位置誤差、AUCCESS 與 Success@k。

---

## 7. 執行順序（2026-09-26 依優先順序改寫）

舊版的第 1 到 4 步（pool、encoder 與 scorer、initial scene baseline、base predictor）都已完成。

**P0：現在開始，可以平行進行**

1. **第一層評估工具與 episode 取樣。** 在 within fold 0 dev 上固定 support–query episode：每個 task 固定 context attempts 與 query actions，query 同時包含成功與失敗。實作四個指標，先量 baseline（不給 feedback）的數字。這是所有 adaptation 實驗的前置條件。
2. **生成鄰近群集資料並 encode。** 依 §1.7.6 的提案，CPU 生成約半小時、GPU encode 約 3 小時。耗時長，所以要最早啟動。
3. **Cross-template 第一段。** 在 fold 1 cross 上訓練 initial scene 模型，並算真實軌跡的上限，約 1 小時、只用 1 張 GPU。它決定第 10 項值不值得做。

**P1：研究的核心**

4. **Memory 第一版。** 在 baseline predictor 上加 context memory：把 context attempts 的 latent 序列（和 residual）編碼成 memory tokens，predictor 用 cross-attention 讀取。用現有 s16r16 資料組 episode 訓練（每個 task 32 條，主要是遠距離配對）。在第一層上跟不給 feedback 比較。
5. **RQ1 內容消融。** 只給 action、只給成功或失敗、完整 latent、latent 加 residual。第 4 項有訊號之後就很便宜。

**P2：延伸研究問題與公平對照**

6. **RQ2：近距離與遠距離轉移。** 用第 2 項的鄰近群集資料。
7. **RQ3：累積。** 1 次 vs 多次 attempt，只給最近一次 vs 全部。
8. **Adaptive 對照組。** DQN online 式的 initial scene 模型微調，以及 AdaJEPA 式 test-time gradient。

**P3：決策效用與泛化**

9. **第二層 closed-loop 加上 adaptation。** 共用 shortlist 512、memory 最多更新 10 次（§1.3.3）。只在第一層有正面結果後才做。
10. **Cross-template 第二段。** 條件已滿足（§1.9：initial scene 模型在 cross 上只領先隨機 0.09），提升優先順序：在 fold 1 cross 資料上訓練 predictor，之後再加 memory。
11. **Recurrent memory 變體。**

**P4：收尾**

12. 多個 seed、fold 1 和 2、test set 的最終數字。
13. 選做：state-space world model 參照組（§1.7 review 提到的「為什麼不直接預測座標」）、訓練資料成功比例的調整、投影片更新（split 改成 16:4:5）、清理 `/mnt/meg` 上約 409 GB 的中間檔（2026-09-26 量測）。

**不做或暫緩：** encoder fine-tune、V-JEPA 2、7×7 池化、diffusion predictor、繼續搜尋 predictor 的訓練設定（§1.7.6 已定案）。

---

## 8. 結果判讀

| 觀察 | 結論 |
|---|---|
| Train-only head 在真實 latent 上就很差 | Head 卡住上限，先修 scorer，memory 結果暫不可信 |
| Initial-frame classifier 跟本方法差不多 | Rollout 沒帶來額外價值，至少在這個 split 上 |
| Oracle head 只看第一幀就很準 | Oracle 在走 shortcut，改用後段輸入或 probe + 規則 |
| 三種 scorer 下 memory 變體排名一致 | Predictor adaptation 的結論不依賴 scorer |
| Prediction error 下降但 ranking 沒變 | 更準的 latent 不一定改善決策，分開報 prediction 跟 decision utility |

---

## 9. 待確認

- **[已驗證]** PHYRE 成功條件是持續接觸 3 秒（`simulator/task_utils.h:33`）。
- **Frame sampling：** 已定案，見 §0 與 §1.2。
- **[已驗證] 成功的 rollout 會提早結束。** 達成成功條件就停止模擬（`simulator/task_utils.cpp:70-73` 的 `break`），失敗的則跑滿 1000 步。所以**序列長度本身就洩漏了成功與否**。用最後一幀補齊擋不住這個洩漏，因為畫面會在動作中途凍結。修正是所有 rollout 都跑滿完整模擬，見 §1.2.4。
- **[未查證]** Girdhar et al., "Forward Prediction for Physical Reasoning" 是否用「proposer top-K 加 forward model 重排」。引用前要查。
- **[未查證]** DQN baseline 在 within / cross-template 的 AUCCESS 數字。
- **Pool：** 先用 5K（§0）。Classifier top-K 的 coverage 和 recall 還沒量。
- **成本：** 見 §1.3。Predictor 做出來之後要實測 throughput，更新粗估。
- **Split：** 已定案 16:4:5 標準 fold（§0）。跑 3 個 fold（§0）。投影片 slide 11 要改。
