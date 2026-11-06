> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Query-token memory：執行前提案審查

## 官方小規模驗證：全部完成（2026-10-12）

本節取代下方尚未完成時的進度描述。Official cross fold1 與 within fold0 各固定100個場景；同一 split 所有方法使用相同的1,024個候選 actions、實際模擬成功標籤與 scorer，最多嘗試20次。主要指標 S@9 是前九次嘗試內成功的場景比例，並非 AUCCESS。Q0 與兩種 memory 分支各3個 seeds；原 base 與 AdaJEPA-style 各一個固定模型／執行。

| 方法 | Cross S@9 | Cross S@20 | Within S@9 | Within S@20 |
|---|---:|---:|---:|---:|
| 原 base | 12.0% | 16.0% | 49.0% | 64.0% |
| 固定 Q0（三 seeds 平均） | 17.7% | 24.0% | 49.7% | 62.3% |
| 真實 residual memory（三 seeds 平均） | 17.0% | 24.7% | 47.3% | 62.3% |
| 零 residual memory（三 seeds 平均） | 17.0% | 25.0% | 48.3% | 64.0% |
| 原 base＋AdaJEPA-style test-time update | 18.0% | 25.0% | 47.0% | 61.0% |

固定 Q0 指完整第一階段訓練後的 predictor＋initial tokens，評估時不更新。真實 residual 分支在失敗 action 後，以觀察减去「當下 tokens 的預測」更新 tokens、重新排序；零 residual 分支的 updater 獨立訓練，residual 歸零但仍有 action、失敗歷史與更新次數。AdaJEPA-style 沿用原 base，在每次失敗後直接更新最後 Transformer block 與 final LayerNorm；並非本 memory 的 updater，也不是原論文所有設定的完整重現。Cross 校準選 lr0.001／1step，within 選 lr0.001／3steps，皆 replay最近5個失敗 trials。

已直接核對兩個原 base checkpoint 的 `args`：cross／within 均為 `aux_probe=""`、`aux_lambda=0.0`、`motion_beta=0.0`。原 base 的訓練 objective 僅為未來 frames 1–31 的 latent L1，沒有啟用幾何監督；訓練程式雖有選用的 auxiliary probe loss，這兩個模型沒有使用。Q0 與 updater 的新訓練均使用 latent L1 + 0.1 geometry loss（經凍結probe讀取動態物體的位置、bar角度後對照真實幾何）。所以「不加tokens、相同額外訓練量與幾何loss的base」是拆解Q0增益所缺的對照；本輪尚未跑此對照。

主要結論：

1. **Cross 相對原 base 的改善，已在 Q0 出現。** S@9 從12%到17.7%；加入真實 residual 後平均17%，未再改善。第一階段同時包含prefix tokens、額外完整訓練與geometry supervision，這個對照不能拆出哪個因素造成Q0增益。
2. **Memory 更新的額外效益仍未建立。** Cross 真實 residual 與零 residual 平均S@9相同；within 真實 residual 比Q0低2.3pp、比零 residual低1pp。S@20也未顯示真實residual穩定超過零residual。
3. **AdaJEPA-style 在這個cross panel有觀察到改善，within沒有。** 相對各自原base，cross S@9 +6pp、S@20 +9pp；within S@9 −2pp、S@20 −3pp。Cross 的18%與memory平均17%僅差1pp，不能據此認定方法普遍優於memory。
4. **這輪不支持將先前開發場景的memory增益推廣到所有場景。** 但成功率本身也不能判定是資料量、loss、scorer或token updater哪一項造成失敗。

每seed S@9如下，可看出cross的更新效果不一致：

| Split／方法 | Seed0 | Seed1 | Seed2 |
|---|---:|---:|---:|
| Cross Q0 | 18% | 21% | 14% |
| Cross 真實 residual | 18% | 16% | 17% |
| Cross 零 residual | 18% | 18% | 15% |
| Within Q0 | 53% | 46% | 50% |
| Within 真實 residual | 53% | 44% | 45% |
| Within 零 residual | 54% | 45% | 46% |

真實residual相對Q0的S@9改變：cross 0／−5／+3pp；within 0／−2／−5pp。Cross同題比較的新增成功／失去成功為4/4、3/8、4/1；within為5/5、1/3、2/7。它確實會改善某些題，也會破壞其他题，而非單純沒有影響。

兩份最終彙總皆 `complete=true`、`raw_audit_passed=true`；TTA的原base分數與memory評估基線逐元素一致。已下載並核對SHA256，另從1,800個memory條件／場景與200個TTA場景的first-success紀錄獨立重算S@1/3/5/9/20，全部一致。三seeds共用100題，不能當作300個獨立場景。

- Memory：`results/small_official_20261010/SUMMARY_FIXED.json`，SHA256 `7885f50cc68a506dbb9ef33cbf04a954d564eeec43291b589674958a45de2601`。
- AdaJEPA-style：`results/adajepa_phyre_20261010/SUMMARY.json`，SHA256 `f6c099c340298edfc0be9e5ac862563f968c694414955d9f66881145f2571835`。

依使用者安排，正式評估完成後才開始training／inference加速；效能測試不改寫上述模型與結果。


## 最新進度：AdaJEPA-style within 完成（2026-10-11）

AdaJEPA-style within 的正式 100 題已完成。重新核對全部 raw scores、逐輪選擇、失敗 replay、reset／凍結參數旗標，並確認 candidates、labels、原 base scores 與 memory 評估逐元素一致。紀錄：`results/adajepa_phyre_20261010/status_20261011/TTA_WITHIN_VERIFIED.json`。

| 方法 | S@9 | S@20 |
|---|---:|---:|
| 原 base | 49% | 64% |
| 固定 Q0（三 seeds 平均） | 49.7% | 62.3% |
| 真實 residual memory（三 seeds 平均） | 47.3% | 62.3% |
| 零 residual memory（三 seeds 平均） | 48.3% | 64% |
| AdaJEPA-style base＋test-time update | 47% | 61% |

S@9 是前九次嘗試內成功的場景比例。這個 within panel 上，兩種更新方式都尚未顯示超越原 base 的平均成功率；不能僅由這張表判定失敗機制。

Cross seed0、seed1 全部完成；seed1 的 Q0／真實 residual／零 residual，S@9 分別 21／16／18%，S@20 分別 28／26／29%。Seed2 Q0 已完成（S@9 14%、S@20 19%），真實 residual 25/100，零 residual 待跑。Cross AdaJEPA-style 尚待 memory 完成後啟動；整輪尚未結束。以下為先前檢查時的紀錄。

## 官方小規模驗證：within 完成，cross 部分完成（2026-10-11）

Within 九個訓練（3 Q0、3 真實 residual、3 零 residual）及九組 100 題評估全部完成。Q0 每 seed 完成 52,344 次更新；兩個 updater 分支每 seed 完成 10,000 scheduled slots，實際 optimizer steps 分別為 4,107／4,297／4,576。Cross 已完成 seed0 的三條件與 seed1 的 Q0。

已重新從 **1,300 個完整場景紀錄**核對 shortlist IDs、實際 labels、穩定排序、每輪排除已試 action 後的 argmax、成功停止、第一次分數等於自身 Q0、原 base 跨 seeds 分數一致，以及所有成功次數與彙總。全部通過。程式與結果：`results/adajepa_phyre_20261010/status_20261011/{audit.py,RESULTS.json}`。

官方 within fold0，相同 100 題的 S@9：

| 方法 | Seed0 | Seed1 | Seed2 | 平均 |
|---|---:|---:|---:|---:|
| 原 base（相同模型） | 49% | 49% | 49% | 49.0% |
| 固定 Q0 | 53% | 46% | 50% | 49.7% |
| 真實 residual | 53% | 44% | 45% | 47.3% |
| 零 residual | 54% | 45% | 46% | 48.3% |

**這輪 within 沒有觀察到真實 residual 的 S@9 改善。** 相對自己的 Q0，三 seeds 為 0／−2／−5pp，平均 −2.3pp；相對零 residual，每個 seed 都低 1pp。相對原 base 平均低 1.7pp。S@20 原 base 64%、Q0 62.3%、真實 residual 62.3%、零 residual 64%；不能說較長預算已救回真實 residual 的優勢。

同題成對比較更能看清 tradeoff：S@9 真實 residual 相對 Q0，seed0 多解 5 題但失去 5 題，seed1 多解 1 題失去 3 題，seed2 多解 2 題失去 7 題。它並非完全不改變決策；改善部分題目，同時傷害其他題目。這些結果還不能單獨判定是訓練資料、loss、scorer 或更新機制的問題。

官方 cross fold1 目前完整 seed0：S@9 原 base **12%**、Q0 **18%**、真實 residual **18%**、零 residual **18%**；S@20 分別 16／25／27／25%。目前 seed0 的主要 S@9 增益來自第一階段，尚未見額外 memory 增益。Seed1 Q0 的 S@9 為 21%；其他分支與 seed2 未完整，不能列三 seeds 平均。

Cross seed1 在 `00008:197`、action3393 的第一幀數值檢查停住：max0.0078125、mean1.3775e-6，超過原 mean1e-6 guard。初始 raster／features 已確認逐元素相同；核對已通過：重新編碼的 initial 與 shortlist 完全相同、完整觀察 latent 與原編碼程式完全相同、改動 future frames 後第一幀 encoder hidden 完全相同；統一 PCA projection 的 tensor shape 後，第一幀 latent 也完全相同。這筆稀疏差異可由 projection shape 的數值捨入重現（99th percentile 為 0，0.87% entries 不同）。只為這筆已驗證觀察寫入原始完整編碼 cache，保留所有 target 值與全域 guard。紀錄在 `status_20261011/OBSERVATION_VERIFIED.json`。Cross 已以 job **34783** 恢復續跑，並修復其後 summary／AdaJEPA 的排程依賴。尚未影響已完成的 within 結果。

AdaJEPA-style within 已完成 25 題校準，選定 lr0.001、每輪3steps、recent5，正式100題評估正在跑。校準分數不是正式比較結果。Cross AdaJEPA 等待 cross memory 續跑完成；尚不能判斷它是否優於 memory。


## 新增 base＋AdaJEPA-style test-time update（2026-10-10，已提交）

使用者要求新增參數更新 baseline。本輪沿用原 base predictor，不增加 memory tokens，也不先做新一輪離線訓練；每次選到失敗 action 後，用它的觀察更新 predictor，再重新 ranking。相同場景的權重會持續累積；換場景才恢復原 base 並清空 replay。

這是 **AdaJEPA-style predictor-only 的 PHYRE 移植**。參考[官方 pinned 實作](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/planning/adajepa.py)，更新最後一個 Transformer block 與最後的 LayerNorm，固定 encoder、PCA、action embedding、其他 predictor layers、輸出 linear 與 scorer。官方預設另外更新 encoder；本輪不包含該分支，也不是原 MPC benchmark 的完整重現。

Online loss 使用原 base 預訓練的 latent L1（時間 1–31）；各已觀察 attempt 等權平均。這裡保留 AdaJEPA「使用原模型的 prediction objective」的原則；官方模型實作使用 MSE，我們的 base 使用 L1。Replay 為最近五次已執行失敗 attempts；每個 attempt 都從自己的初始場景／placement action 預測，不把 reset 前後串成連續影片。Adam 在每次 adaptation call 重新建立，權重在同場景內延續，與 pinned 官方 trainer 的行為一致。

先在既有 dev 資料校準六組：learning rate `3e-5 / 3e-4 / 1e-3` × 每輪 `1 / 3` gradient steps。Cross 固定 20 題（四個 dev templates 各五題、每題 1,024 actions），within 固定 25 題（每 template 一題、每題既有 512 actions）。兩邊分別依 S@9 選設定，同分依 S@5、S@1，再依較少 steps／較小 learning rate；之後固定設定評估目前的官方 100 題／split、共用 1,024 候選池與實際模擬 labels。主要指標 S@9，次要 S@1/3/5/20，另記 update、ranking、觀察處理耗時與峰值顯存。

TTA 使用同一個原 base，replay 沒有隨機抽樣，因此每個 split 跑一個 deterministic baseline，不把三次相同執行假稱為三個訓練 seeds。Memory 仍保留三 seeds。這是同 interaction budget 的整體方法比較，各方法會自行選到不同 actions；不是固定 evidence 或相同離線訓練量的純機制對照。

目錄 `results/adajepa_phyre_20261010`；`PLAN.json` 固定設定與 source／runtime hashes。GPU smoke 34742 通過：連續三次觀察、每次三步梯度更新，參數確實改變、凍結部分不變、reset 精確還原；另通過真實 1,024 候選池的三 trial closed-loop audit。這些不是效果結果。正式 TTA cross／within jobs **34754／34755** 已排隊，會分別接在 memory cross／within seed0 完成後使用空出的 GPU；summary **34756**。全流程最多四張 GPU。

### 共用評估修正

先前 smoke 34731 在真實 trial 的第一幀一致性檢查停住，尚未開始正式訓練／評估。確認初始 raster 與完整模擬第一幀逐 pixel 相同；差異來自 encoder 的 batch shape：同一畫面以 batch32 編碼，相對完整觀察第一幀最大差 `0.100586`、平均 `0.002302`，batch1 則最大 `0.000977`、平均 `1.86e-7`。這與先前歷史快取的稀疏微小差異是不同問題；前置檢查只驗證 single-image 路徑，沒有覆蓋實際 batch32。

修正固定初始 encoder 每次輸入一張畫面；predictor 候選評分仍 batch32。GPU check **34741** 通過原編碼重現與實際失敗 case 的 32-image caller 檢查。所有候選池必須重新編碼、重新 base ranking、重新選 shortlist／模擬 labels。新資料放 `assets_fixed`，正式輸出放 `evaluation_fixed`，舊 code／資料／smoke 均保留；舊未開始的 formal jobs 已取消並替換。原先已完成的歷史 prefix experiment 不受此次新資料路徑問題影響。

修正版四個 shortlist jobs **34743–34746** 執行中，接 labels **34747**、eval smoke **34748**、memory cross **34749**／within 三 seeds **34750–34752**、memory summary **34753**。新的權威排程為 `results/adajepa_phyre_20261010/jobs.json`。


## 小規模官方 split 驗證（2026-10-10，已提交）

這輪要回答：換成官方 split 的固定場景後，現在的 prefix-token 方法是否仍能比固定 Q0 更有效地使用嘗試次數，以及改善是否來自 residual 本身。

| 設定 | Cross | Within |
|---|---|---|
| Fold | 1 | **0，使用者已確認** |
| 評估場景 | 官方 test 固定 100 題，5 templates 各 20 題 | 官方 test 固定 100 題，25 templates 各 4 題 |
| 模型 | 重用先前完成的 3 seeds | 使用既有 within0 base，重新訓練 3 seeds |
| 固定 scorer | 既有 predicted_mix_s0 | 既有 within0 original probe |

同一設定中的四個比較條件為原 base、固定 Q0、真實 residual updater、獨立訓練的零 residual updater。零 residual 仍有已嘗試 action、失敗回饋與更新次數。主要指標是 S@9，另列 S@1/3/5/20，以及每個 seed 的成對差異；不把三 seeds 的相同題目當成 300 個獨立場景。Scorer 在各設定內固定，cross／within 使用各自的 head，因此不能把兩者差值單純歸因於 split 難度。

Within 第一階段使用原 base 的完整 52,349 條訓練軌跡、8 epochs、batch 8；每 epoch 捨去末尾不足一個 batch 的 5 條，共 **52,344 次更新、418,752 次 trajectory exposure**。訓練 pretrained predictor 與 active Q0，loss 為 latent L1 + 0.1 geometry。第二階段凍結 predictor／Q0，只訓練 updater；每個分支 20,000 episodes／10,000 個兩場景 slots。實際 optimizer steps 取決於是否有失敗 support，完成後會另列。

Within 第二階段保留 1,200 個訓練場景（每 template 48），從既有 cluster 與 s16r16 資料組成候選／query。Support 29,723 條，其中成功 13,026；query 30,079 條，其中成功 13,166。角色依 action group 分開，不使用 Near 門檻、不預先只留下 failures。Support 由當下 predictor／tokens＋scorer 選擇，成功即停止；只有實際選到的失敗 action 可更新 tokens。Residual 每次都重新計算為「觀察 latent − 當下預測 latent」。

每題從官方 cache 前 5,000 個 action 中取有效候選，再用該設定的原 base＋固定 scorer 選共同的 1,024 條 shortlist；選池只讀初始畫面與 action，不讀成功 label。所有 shortlist action 的結果以 simulator 重算。評估時只為已選中的失敗 trial 編碼完整觀察；可跨模型重用的是觀察，不是 residual。

執行目錄：`results/small_official_20261010`。200 題初始資料準備與完整訓練／評估依賴圖已提交，最多四張 GPU。編碼檢查、CPU 架構／因果 replay、三個分支的短程 GPU 訓練，以及連續八次更新的完整梯度 replay 均通過。四張 GPU 已開始建立候選 shortlist；之後自動接續 outcome 重算、線上評估檢查與正式訓練／評估。正式訓練尚未開始。新編碼與原程式完全一致；歷史 within 快取存在平均約 1.18e-7 的稀疏數值差異，已保留核對紀錄。歷史實驗與 checkpoint 均保留。


## 完整 memory 結果：全部完成（2026-10-09）

**九次正式訓練、十八組評估與最後彙整全部完成，目前無執行中或排隊作業。** 原排程 summary 稽核通過；另用獨立 NumPy 程式重新核對全部 1,800 個 static／dynamic 場景紀錄，包括候選 ID／label、穩定排序、排除已試 action 後的 argmax、成功停止、每個 updater 第一動分數與自身 Q0 完全一致，以及所有首次成功次數與彙總數字。全部通過。以下結果取代前面的中途進度。

每個 split 為相同 100 個場景、相同 1,024 候選池與 scorer，以下是三 seeds 平均的 **S@9（九次嘗試內成功率）**：

| 方法 | Cross | Within |
|---|---:|---:|
| 原 base | 47.0% | 54.0% |
| 第一階段，固定 Q0、不更新 | 48.0% | 52.0% |
| 第二階段，真實 residual 更新 | **50.7%** | **57.7%** |
| 第二階段，零 residual 更新 | 49.3% | 55.3% |

真實 residual 相對自己第一階段 Q0，cross **+2.7pp**、within **+5.7pp**；相對原 base，兩邊都是 **+3.7pp**。Within 每個 seed 都高於 Q0 與原 base；cross 相對 Q0 為兩個改善、一個持平，相對原 base 仍有一個 seed 較低。

| Split／seed | Q0 | 真實 residual | 零 residual | 真實−Q0 | 真實−零 residual |
|---|---:|---:|---:|---:|---:|
| Cross 0 | 50% | 53% | 50% | +3pp | +3pp |
| Cross 1 | 46% | 46% | 48% | 0pp | −2pp |
| Cross 2 | 48% | 53% | 50% | +5pp | +3pp |
| Within 0 | 49% | 56% | 51% | +7pp | +5pp |
| Within 1 | 54% | 60% | 58% | +6pp | +2pp |
| Within 2 | 53% | 57% | 57% | +4pp | 0pp |

**目前可以說這個修正後版本在主要 S@9 指標出現 memory 更新收益，within 較一致；不能把全部收益歸因 residual 內容。** 零 residual 也優於固定 Q0：它仍保留已嘗試的 action 與更新歷史，並且只有觀察失敗才更新。真實 residual 相對零 residual 的平均增益僅 cross +1.3pp、within +2.3pp；cross 並非所有 seed 都贏，within 有一個 seed 持平。

次要的 **S@20** 也要一起保留，避免只挑有利指標：

| 方法 | Cross S@20 | Within S@20 |
|---|---:|---:|
| 原 base | 51.0% | 67.0% |
| 固定 Q0 | 55.0% | 68.0% |
| 真實 residual | 56.0% | 73.0% |
| 零 residual | **56.3%** | **73.3%** |

到二十次嘗試時，真實 residual 兩邊平均都未超過零 residual。S@20 超出訓練最多八次更新的長度，屬預先登記的壓力測試；仍不能省略此結果。這些都是既有開發面板、三個 seeds 的描述性比較，尚不是獨立測試集或統計上可靠的普遍泛化結論。新架構與 scorer-policy 訓練同時改變，也不能由舊／新版本差異單獨歸因哪一項修正。

訓練量核對：各 Q0 seed 完整 52,272 updates；各 updater 分支完整 20,000 episodes／10,000 排程步。因第一動成功而沒有 updater 梯度，實際 optimizer 次數依 seed 為 4,749／4,758／4,734，兩分支一致；不存在未完成訓練冒稱完整結果的情況。

證據：[完整 summary](../results/prefix_staged_20261008/SUMMARY.json)、[獨立 1,800 場景 raw audit 與逐 seed 配對](../results/prefix_staged_20261008/final_analysis/AUDIT.json)、[核對程式](../results/prefix_staged_20261008/final_analysis/audit.py)。Summary 和 audit 已下載，與 shared server SHA 一致；逐題成功次數另在本機重新加總核對。


## 原 base 與第一階段模型的直接比較（2026-10-09）

**第一階段沒有全面勝過原 base。** 前九次嘗試的成功率 cross 略升、within 略降；放寬到二十次則平均改善。這次兩邊都不使用觀察後的 memory 更新：比較原 pretrained base 與訓練完成的 predictor＋固定 initial prefix tokens（Q0）。Cross／within 各同一批 100 場景、每場景同一組 1,024 actions、相同 frozen `predicted_mix_s0` scorer。三 seeds 的原 base candidate IDs／labels／scores 逐元素一致；全部 600 份原始 score arrays 重新排序，first-success 與已保存結果一致。

| Split／指標 | 原 base | 第一階段三 seeds 平均 | 差異（百分點） |
|---|---:|---:|---:|
| Cross S@1 | 26.0% | 26.0% | 0.0 |
| Cross S@9 | 47.0% | 48.0% | +1.0 |
| Cross S@20 | 51.0% | 55.0% | +4.0 |
| Within S@1 | 23.0% | 22.3% | −0.7 |
| Within S@9 | 54.0% | 52.0% | −2.0 |
| Within S@20 | 67.0% | 68.0% | +1.0 |

S@9 的三個 Q0 seeds：cross **50／46／48**，within **49／54／53**。Cross S@20 為 56／55／54，三 seeds 都高於原 base 51；但不能從三 seeds 和既有少數模板開發面板宣稱廣泛泛化或顯著性。Within S@9 沒有 seed 高於原 base：兩個較低、一個持平。

逐題配對也显示有得有失。Cross S@9 各 seed 相對 base 的「新解出／原本解出但失去」場景數為 10／7、5／6、7／6；within 為 5／10、9／9、6／7。因此小的淨差異不代表兩模型都解出同一批題。

平均 scene ranking AUC：cross 0.72476→0.73190（84 個同時有成功及失敗候選的場景），within 0.84778→0.86443（99 個）。Within 三 seeds AUC 都提高，但 S@9 沒有提高；這说明整體成功／失敗配對排序改善，可以同時伴隨最前幾次嘗試表現不佳。AUC 評估全部候選配對，S@9 只取決於第一個成功 action 是否進入前九名，兩者不是同一目標；此現象沒有單獨識別失敗機制。

另使用**同一份 400 條 historical base-dev trajectories**，重新執行原 base 與三個 final Q0 checkpoints 的 GPU 前向測量（job 34722，已完成）。這份資料與上面的每場景 1,024 排序候選池不同，不能把誤差結果直接當成 within 全候選的誤差。

| 固定 400 條 dev 的預測誤差（越低越好） | 原 base | Q0 三 seeds 平均 | 相對變化 |
|---|---:|---:|---:|
| Latent L1，排除 t0 | 0.119949 | 0.120932 | +0.82% |
| 原 frozen probe 的 geometry loss | 1.021492 | 0.917750 | -10.16% |

Geometry loss 使用訓練時同一原 probe、每條軌跡固定八個時間點、dynamic objects 的位置及適用時的角度項；它是加權且正規化的 loss，不是像素 MAE。三個 seeds 的 latent L1 都略高於原 base，geometry loss 都較低。新版 dev_initial L1 約 0.1355 是「原權重加上隨機 prefix」後的預測，不是原 base 的 0.11995；不得用 0.1355 作為 base baseline 宣稱新版勝出。

這是固定相同 scorer 下兩套模型的實際表現比較。新版同時加入 prefix、額外完整訓練，以及 geometry loss，不能把差異單獨歸因 prefix tokens；同一固定 probe/scorer 也不能排除對不同預測分布的適配差異。之後判斷 updater 的增益，應和同 seed 的 Q0 比；判斷整套方法是否優於既有系統，再和原 base 比。

證據：[ranking 與逐題配對](../results/prefix_staged_20261008/stage1_comparison/RANKING.json)、[matched dev metrics](../results/prefix_staged_20261008/stage1_comparison/DEV_METRICS.json)、[比較摘要](../results/prefix_staged_20261008/stage1_comparison/COMPARISON.json)。程式與輸出 SHA 已核對，不改動既有訓練／評估版本；其餘 updater 評估繼續原排程。


## 最新進度（2026-10-09，08:01 UTC）

**九次正式訓練全部完成；十八組評估完成十四組，剩四組正常執行。** 三個 Q0 seeds 均完成完整 52,272 次 optimizer 更新。六個 updater 分支均完成各 20,000 episodes／10,000 排程步，實際 optimizer 更新數依 seed 為 4,749／4,758／4,734，真實與零 residual 分支相同。各 seed 有 14,482／14,460／14,519 個第一動成功的 episodes，依登記不產生 updater 梯度，沒有額外重抽失敗場景。所有 checkpoint hash、凍結參數、父 checkpoint 與配對初始化均通過核對。

Q0 的 cross S@9 為 50／46／48，平均 **48.0%**，原 base 為 47%；within 為 49／54／53，平均 **52.0%**，原 base 為 54%。這是第一階段模型結果，不能當作 memory 更新效果。

目前所有三條分支都已完成的成對結果如下，每個 cell 都是同一 split 的完整 100 場景 S@9；尚未完整的 seeds 不混入平均：

| 已完成的成對評估 | 固定 Q0 | 真實 residual | 零 residual |
|---|---:|---:|---:|
| Cross seed 0 | 50% | 53% | 50% |
| Cross seed 1 | 46% | 46% | 48% |
| Within seed 1 | 54% | 60% | 58% |

目前有改善也有持平／輸給零 residual 的情況，尚不能宣稱穩定收益。剩餘進度：cross residual_s2: 18/100; within residual_s0: 33/100; within residual_s2: 24/100; within zero_s0: 26/100。Summary job 34721 會在全部完成後自動核對原始分數並彙整。這次另逐題重算十四組已完成評估的首次成功統計，均與 complete counts 一致；全套最終 raw-score 稽核仍待 summary 完成。[狀態與核對紀錄](../results/prefix_staged_20261008/LATEST_STATUS.json)。


## 已確認並提交：修正架構的兩階段重訓（2026-10-08）

**啟動核實：端到端 GPU smoke 34693 已全部通過，正式第一階段三個 seeds（34694／34703／34712）已在三張 GPU 執行。** 後續六次 updater 訓練與十八組評估會依相依排程自動接續。

驗證涵蓋 40 次 Q0 optimizer 更新、兩個 updater 分支各 32 個排程步，以及 1,024 候選的閉環與原始分數稽核。每個 updater smoke 有 64 個 episodes，其中 18 個有失敗 support、46 個零 support，實際各做 16 次 optimizer 更新；這是小樣本工程驗證，不是正式有效訓練量估计。額外的八次連續更新＋八個 query 壓力測試，顯存峰值 **17.59 GiB**，完整梯度與逐次 residual 重播均通過。原 base 在同一測試場景的 1,024 個分數與上一版逐元素相同。所有 smoke 權重不沿用到正式訓練。證據：[SMOKE](../results/prefix_staged_20261008/SMOKE.json)、[reference replay](../results/prefix_staged_20261008/REFERENCE_REPLAY_AUDIT.json)。

使用者已確認優先執行兩階段版本，範圍選擇已解決。整套三 seeds、九次正式訓練、十八組 cross／within 評估及 summary 都已提交；由新的端到端 GPU smoke 成功後自動啟動。舊 ABCDE 完整評估及共用 prefix smoke 已完成。

第一階段從原 pretrained predictor 加上新的 32 個 initial prefix tokens 開始，兩者一起訓練。使用原 base 的 52,276 條 trajectories／1,600 場景，batch 8、8 epochs，共 **52,272 次 optimizer 更新、418,176 次 trajectory presentations／seed**。每個 epoch 丟棄最後不足一批的四筆，抽樣順序與上一版完整預算兩階段實驗一致。Loss 是 latent L1＋0.1 geometry，geometry 由原 frozen probe 提供；這和原 base 的純 L1 objective 不同，並且是加在既有 pretraining 之上的額外完整預算。

第二階段固定 predictor 和 initial tokens，每個 seed 分成真實 residual／零 residual 兩個 fresh updater 分支。每個分支跑 20,000 個 scene episodes，兩個 scene 一組，共 10,000 個排程訓練步。每個 episode 隨機選最多 1／2／4／8 次更新，由當下 predictor＋tokens＋固定 adapted scorer `predicted_mix_s0` 逐步選 trial；候選包含成功與失敗，成功即停止，不重複 action。Residual 為選中 action 的觀察 latent 減當下預測，梯度重播時逐次重算，不快取 residual，也不對離散 argmax 求梯度。

每個 episode 以最後已觀察失敗形成的 tokens 預測最多八個與候選分組隔離的 query，使用 L1＋0.1 geometry loss。兩個 scene 各佔一半權重，分開 backward 以降低顯存。**第一動就成功的 episode 保留、不繼續找失敗**：記錄 Q0 loss，但因 predictor／Q0 已固定，沒有 updater 梯度。若一組兩個 scene 都沒有失敗 support，optimizer 不更新，learning-rate schedule 仍按排程步前進。因此實際 optimizer 次數可能低於 10,000，必須和 episode 數、零 support 比例一起報告。兩分支使用相同 scene／query／長度上限抽樣及初始 updater，實際選到的 actions 會因策略不同而分歧。

評估固定最終 checkpoint，不依 cross 成績選模型。比較原 base、各 seed 的固定 Q0、真實 residual 更新、零 residual 更新；cross／within 各全部 100 個場景，主要 S@9，次要 S@1/3/5/20。Updater 第一動分數必須與自己的 Q0 完全一致，原始分數會核對 argmax、去重、success-stop 及成功次數。S@20 是超過最多八次訓練更新的壓力測試。

限制：訓練使用既有 support-half 的 29,868 個候選／1,200 場景，其中 12,952 個成功；沒有 Near 門檻、沒有失敗預篩，但每場景池仍小於部署 1,024 個候選。沿用的 cross／within 都是 development panels；within 是同模板新生成場景。零 residual 對照也會改變後續選 action 分布，因此差異不是純粹固定 supports 下的 residual 資訊效果。

作業：端到端 smoke **34693**；第一階段 **34694／34703／34712**；真實 residual **34695／34704／34713**；零 residual **34696／34705／34714**；十八組評估接續對應 checkpoint；summary **34721**。初始三張 GPU，後續依 oprime 的八張 GPU 容量排程，不占用別人的作業。CPU 架構／資料／抽樣與 raw audit 測試已通過。新 GPU smoke 已驗證三個階段的實際 optimizer、連續八次 residual 更新的完整梯度與最大顯存，以及真實 1,024 候選閉環。

登記：[manifest](../results/prefix_staged_20261008/manifest.json)、[DATA_AUDIT](../results/prefix_staged_20261008/DATA_AUDIT.json)、[jobs](../results/prefix_staged_20261008/jobs.json)、[訓練與評估實作](../results/prefix_staged_20261008/code/prefix_staged.py)。正式權重／resume 保留在 shared server run 目錄；以下較早段落的「範圍未定、正式未提交」已被本節取代。


## 最新進度與完整 ABCDE 結果（2026-10-08）

**舊完整評估全部完成，新架構 GPU smoke 通過；新版正式重訓尚未提交。**
舊 18 組 memory 評估及 12 組無 memory 評估已完成，summary 的 complete／raw_audit_passed 均為 true。本次另從 summary 內 30 組、每組 100 題的首次成功次數，重新計算 S@1/3/5/9/20，全部一致；本次本機核對是 summary 內逐題資料，原始 trace 稽核由 server 的 summarizer 執行。

以下每個 split 都是全部 100 題，三 seeds 平均，S@9 表示九次嘗試內成功率，並非 AUCCESS。

| 方法 | Cross S@9 | Within S@9 |
|---|---:|---:|
| 原 base | 47.0% | 54.0% |
| A：凍結 pretrained predictor＋memory | 46.0% | 56.7% |
| B：pretrained predictor＋memory 一起訓練 | 44.7% | 62.7% |
| C：random predictor＋memory 一起訓練 | 47.7% | 60.3% |
| D：pretrained predictor，無 memory | 47.3% | 60.7% |
| E：random predictor，無 memory | 45.7% | 57.3% |

Within 的 B 相對原 base 增加 8.7 個百分點，但對應的無 memory 續訓 D 也增加 6.7 個百分點，因此不能把全部提升歸因 memory。B−D 為 within +2.0pp、cross −2.7pp；C−E 為 within +3.0pp、cross +2.0pp。這些是三 seeds 的描述性差異，尚不能宣稱穩定或顯著收益。Cross 的 seed 波動仍大，例如 B 為 49／38／47，C 為 38／53／52。

全部仍是舊 reader＋隨機失敗 support 版本，每組 10K updates；C/E 並未使用完整 base 訓練預算，不能當作 scratch 已收斂比較。兩個面板都已用於開發，不是新的獨立測試集。證據：[完整 summary](../results/query_token_dynamic_full_eval_20261007/SUMMARY.json)。

新版 job 34690 已完成：真實訓練資料上三次 updater optimizer steps，逐次 current-residual 重播在 GPU 完全一致；凍結 Q0 的預測保持一致、scorer 選 trial、成功停止均通過。這是工程正確性驗證，不是效果或正式訓練結果。小規模測試耗時 7.46 秒，GPU 峰值 9.05 GiB，不能據此推算完整訓練時間。[SMOKE.json](../results/prefix_policy_20261008/SMOKE.json)。

最新 scheduler 沒有此帳號的排隊或執行中作業。正式訓練矩陣仍待釐清 ABCDE 或完整預算 Q0→updater 兩階段；保留使用者先前要求的完整 initial-token 訓練量。以下準備與提交段落保留歷史經過，當前狀態以本節為準。


## 修正後版本的準備與接續執行（2026-10-08）

使用者要求舊完整評估跑完後，用修正架構重跑。共用實作已完成並在實際 server Python 環境通過 CPU 工程測試；目前只提交 **GPU smoke 34690，依賴舊完整評估 summary 34672 成功**。正式新訓練尚未提交，正在釐清重跑 ABCDE 五組三 seeds，或完整預算 Q0→updater 的兩階段三 seeds。兩者的 freeze 與訓練預算不同，不能默認縮短之前要求的 Q0 訓練量。

實際結構：原 base 是分軸的 spatial／temporal self-attention transformer。新實作把 32 個 memory prefix slots（width 384）串接到每個時間位置的 spatial tokens 前；原有 spatial attention 同時處理 prefix 與 scene，temporal attention 同時處理兩者各自的時間軸。所有原有 QKV／MLP 權重可沿用，沒有額外 reader。這不是把所有 time×space 改成單一全域 attention。Prefix 的臨時 activations 在一次 action prediction 後丟棄，不能把預測不同候選造成的變化寫回持久 memory；只有實際觀察到的 residual 能更新 episode tokens。

Q0 從第一個 action 就參與 self-attention。這種 prefix 即使隨機初始化也會改變 attention normalization，因此**不保證初始預測等於原 base**；不能沿用舊 zero-output adapter 的 identity 宣稱。Initial tokens 需要獨立、足量的初始預測訓練，具體預算隨最終實驗矩陣固定。

Scorer-policy 共用流程已實作：每一步對所有未試候選用當下 Qt 重新打分，穩定 argmax 後才取得那個 action 的 label／trajectory；成功停止，失敗才計算 residual 並更新。選 action 不對 argmax 求梯度；訓練重播選出的前綴時，權重保持相同，逐次重算當下 residual 並保留完整 BPTT，不能快取 residual。Gradient checkpointing 已實作並驗證與非 checkpoint 梯度一致。

候選資料核對：沿用訓練場景原 support-half 的全部候選，包含成功與失敗，沒有 Near 篩選。1,200 場景共有 29,868 個候選，其中 12,952 成功；1,189 場景至少有一個成功候選，所有場景均有失敗候選。29,349 個 query actions 與候選依 action group 隔離。此訓練候選池仍小於部署的 1,024 pool，必須明確登記，不稱完全相同 deploy distribution。第一個 action 成功的 episode 不能偷偷略過成功繼續找失敗；如何監督零次更新與如何計算正式訓練預算，需在 launch 前寫清楚。

CPU 測試通過：prefix 真正進入 shared self-attention、Q0 第一動作有效、候選選擇不接收 future／labels、success-stop／去重、動態 residual 狀態精確重播、固定 predictor 對 tokens 的梯度、gradient checkpoint 等價（最大梯度差 0），batch／serial 打分差 1.11e−16。GPU smoke 接續舊評估後驗證實際資料和三次 updater 訓練更新；不作效果證據，其權重不會沿用到正式訓練。

實作與登記：[prefix_policy_predictor.py](../pipeline/prefix_policy_predictor.py)、[CPU_AUDIT.json](../results/prefix_policy_20261008/CPU_AUDIT.json)、[PREPARATION.json](../results/prefix_policy_20261008/PREPARATION.json)、[jobs.json](../results/prefix_policy_20261008/jobs.json)。

## ABCDE 全 100 題閉環評估擴充（2026-10-07，已提交）

使用者要求把 ABCDE 的小面板評估補完整。這輪保留既有 15 個 final checkpoints，沿用相同候選池、主要 scorer `predicted_mix_s0`、選 action／success-stop／去重／token reset 規則；cross 與 within 各全部 100 題，三個 seeds。主要終點 S@9，次要 S@1/3/5/20；S@20 超出八次更新的訓練長度，不稱官方 AUCCESS。

這是對舊 cross-attention reader＋隨機 support 訓練模型擴充評估，沒有重新訓練，也沒有冒稱已實作 prefix＋training-time scorer 方法。兩個評估集都是既有 development panels，within 為同模板生成場景。固定-query 評估之前已涵蓋全部 100 題；本輪補的是部署主策略 full recurrence 的完整閉環。其他 ablations 保留原 20 題範圍。

- D/E 沒有 memory，部署排序不會更新：直接從已保存的全 100 題分數精確推導首次成功排名。12 個 model/seed/split、共 1,200 題已完成；其中原 20 題共 240 條閉環的每一步分數與排序均完全重現。
- A/B/C 三 seeds、兩 splits：共 18 個 GPU 作業。每組重用舊的 20 題原始 full-recurrence 紀錄，共 360 條；每組補 80 題，共 1,440 條新閉環，最多 20 次嘗試。既有紀錄以唯讀用途的 symlink 引用，不覆寫舊結果。
- 新 wrapper 只擴充場景覆蓋，數值實作直接載入舊版 frozen code；15 個 checkpoints hash、候選來源及 scorer 一致性已核對。
- GPU smoke 先對 A-cross、B-within、C-cross 各重跑舊案例的前三次嘗試，逐元素核對 score／token state／selected actions。通過後自動啟動正式 array，最多五 GPU，最後自動彙整並重新核對所有原始閉環。

作業：smoke **34670** → full evaluation **34671（0–17，最多五個並行）** → summary **34672**。GPU smoke 三個案例全部通過逐元素重播，正式 array 的 0–4 已在五張 GPU 執行，其餘依並行上限排隊；完成狀態以新 run 的 jobs／SMOKE／SUMMARY 為準。新結果不可先填成舊 20 題的數字。

登記與實作：[manifest.json](../results/query_token_dynamic_full_eval_20261007/manifest.json)、[evaluate_full.py](../results/query_token_dynamic_full_eval_20261007/evaluate_full.py)、[CPU_AUDIT.json](../results/query_token_dynamic_full_eval_20261007/CPU_AUDIT.json)、[jobs.json](../results/query_token_dynamic_full_eval_20261007/jobs.json)。

## 使用者澄清：已跑版本與原意的差異（2026-10-07）

使用者明確指出期望：predictor 使用 self-attention transformer，memory tokens 作為 prefix context；training 時也要依當下 predictor／tokens 的 scorer 排名選擇要嘗試的 support actions。後續設計以此為準。

**目前已跑版本不符合這兩點。** 程式 QueryTokenPredictor.forward 在原 predictor block 後額外做 cross-attention，prediction features 作 Q、memory 作 K/V；沒有把 memory 與 prediction tokens 串成共同 self-attention 序列。Updater 用 residual 更新 memory 的 cross-attention 是另一個模組，不應混為同一設計。訓練 episode 隨機抽失敗 support，沒有透過 scorer 選取；scorer 驅動選 action 只在部署閉環及離線 reference-policy 診斷出現。提案把 policy-driven training 延後到下一階段，是助理的實作／規劃選擇，不能視為原方案已獲驗證。

目前所有數據的有效範圍應明確標為「cross-attention reader＋隨機失敗 support 訓練」。已觀察到的錯誤修正屬於這個變體；**不能據此否定使用者的 prefix self-attention＋training-time scorer 選 action 方案**。本次只澄清規格並核對舊作業，尚未實作／提交上述正確版本。

ABCDE 舊版已全數完成：5 組 × 3 seeds 的訓練、cross／within 評估及 summary audit 均通過。三 seeds 平均閉環 S@9 如下，每個 split 僅預選 **20 題**（固定-query 評估另外涵蓋 100 題），不能直接和後來兩階段版本的全 100 題 S@9 比較。

| 組別 | 舊版設定 | Cross S@9 | Within S@9 |
|---|---|---:|---:|
| 原 base | 固定參考模型 | 60.0% | 60.0% |
| A | 凍結 pretrained predictor，訓練 memory 模組 | 56.7% | 60.0% |
| B | Pretrained predictor 與 memory 一起訓練 | 50.0% | 68.3% |
| C | Random predictor 與 memory 一起訓練 | 51.7% | 68.3% |
| D | Pretrained predictor，無 memory | 53.3% | 56.7% |
| E | Random predictor，無 memory | 45.0% | 60.0% |

證據：[cross summary](../results/query_token_dynamic_20261005/SUMMARY.json)、[within summary](../results/query_token_dynamic_20261005/WITHIN_SUMMARY.json)，各 15 模型 complete／audit_passed=true。訓練均為額外 10,000 updates；C/E 沒有 pretrained 權重，也沒有完整 base-sized 訓練預算，不能作已收斂 scratch 能力比較。A/B/C 同樣使用 cross-attention reader 與隨機失敗 support，尚未測過澄清後的 prefix＋policy-driven training。

**原 base 兩邊都是 60% 的獨立核對：** 重算原始候選分數後，確實是兩個不同 20 題子集各成功 12 題。Cross 與 within 的場景及模板完全不重疊；選題由預先固定的 task hash 決定，不依成功率挑題。對全部 15 模型、每 split 20 題的 300 組 baseline indices／labels／scores，與後來全 100 題評估保存的原 base 分數逐元素比對完全一致，summary 也一致，未發現複製 baseline 或混用 split。

| 同一原 predictor＋同一固定 adapted scorer | Cross S@9 | Within S@9 |
|---|---:|---:|
| ABCDE 使用的 20 題子集 | 12/20 = 60% | 12/20 = 60% |
| 其餘 80 題 | 35/80 = 43.75% | 42/80 = 52.5% |
| 全部 100 題 | 47/100 = 47% | 54/100 = 54% |

舊子集的 S@20 也不同：cross 12/20、within 14/20。Cross 20 題按四模板各五題，成功數 5／5／0／2；within 按十模板各兩題。60% 相同是樣本結果巧合；但小子集的代表性確有局限，尤其 cross 子集的 base 比全體高 13pp。ABCDE 表只支持該小面板的比較，不是官方 benchmark 或全 100 題效果。證據：[BASELINE_PANEL_AUDIT.json](../results/query_token_dynamic_20261005/BASELINE_PANEL_AUDIT.json)、[核對程式](../results/query_token_dynamic_20261005/audit_baseline_panels.py)。

以下保留已執行變體的規劃及分析，不能當作澄清後方案的實作規格。

## 新實驗規劃：先訓練 Q0 predictor，再訓練 updater（2026-10-05）

**狀態（2026-10-07 15:11 UTC）：九次訓練、cross／within 全部三 seeds 的主要評估與次要診斷已完成；兩份 summary 的原始結果核對均通過。舊實驗的六項暫停已全部解除。** 主方案為先訓練 pretrained predictor＋Q0，再固定 predictor、從頭訓練 updater；不沿用 v1 的 memory 權重。

### 完成結果：九次嘗試內的成功率

每個 split 都計入全部 100 個場景，使用相同候選池與 scorer。表中為三個 seeds 的平均；原 base 是同一個固定模型。這是 S@9，不是 AUCCESS。

| 方法 | Cross | Within |
|---|---:|---:|
| 原 base | 47.0% | 54.0% |
| 第一階段模型，固定 initial tokens Q0 | 46.0% | 55.0% |
| 第二階段，真實 residual 更新 tokens | 45.0% | 58.0% |
| 第二階段，residual 永遠置零的對照 | 46.0% | 55.7% |

Within 的 Q0 三 seeds 分別解出 57／53／55 題，residual 為 60／58／56 題，zero 為 54／56／57 題。真實 residual 相對自己的 Q0 都有改善（+3／+5／+1 題），但相對 zero 並非每個 seed 都贏（+6／+2／−1 題）。Cross 的 Q0 為 42／49／47 題，residual 為 41／47／47 題，zero 為 43／48／47 題，仍無更新收益。

目前可以說：**這個兩階段方法在既有 within 面板出現收益，尚未轉成 cross 收益。** 不能因此斷言 residual 普遍有效，或直接確定 cross 失敗機制。兩個面板都已在開發中使用；within 是同模板新生成場景，非官方 fold。三 seeds 與有限模板不足以證明廣泛泛化。

完成證據：[cross summary](../results/query_token_staged_20261005/SUMMARY.json)、[within summary](../results/query_token_staged_20261005/WITHIN_SUMMARY.json)，兩者 complete／raw_audit_passed 均為 true；本次亦逐題重算全部成功數。各分支第一次嘗試與自身 Q0 的成功結果相同。先前 cross 原始分數獨立核對見 [CROSS_PRIMARY_AUDIT.json](../results/query_token_staged_20261005/CROSS_PRIMARY_AUDIT.json)。


使用者指定的方向是：先訓練「pretrained predictor＋initial query tokens」的無經驗預測，再加入 support/query 訓練 updater。這裡將「只訓練 updater」具體化為凍結 predictor、Q0、predictor 讀取 tokens 的 adapters，僅訓練 residual/action encoder 與狀態更新網路；不是偷偷連 predictor 一起 fine-tune。

### Cross 機制追查：真實 latent 與 support–query 修正方向（2026-10-07）

**本輪最強的新證據是：在 updater 淨掉題的 00014 模板，失敗 support 與成功 query 需要相反方向的幾何修正，但更新幾乎總把成功 query 推向錯誤方向。** 這把先前籠統的「泛化不好」縮小為一個可測量的 support–query 適用關係問題。仍未證明是 loss、support 抽樣或架構造成這種行為。

本輪只做 CPU 分析，未新增訓練、probe adaptation 或 GPU 推論。既有 `probe_adaptation_cross_20261001` 已保存相同 cross pack 的真實 latent→probe 輸出，因此直接復用。核對 pack 路徑／meta SHA、原始與 adapted probe SHA、全部 100 場景的候選 indices／labels／真實 geometry、原 base 的 distance／score 完全相等；另對三 seeds × 100 場景 × 兩 probes 的 Q0 和更新後分數重算，600 組全部通過。程式與證據：[investigate.py](../results/query_token_staged_20261005/mechanism_followup/investigate.py)、[RESULTS.json](../results/query_token_staged_20261005/mechanism_followup/RESULTS.json)、[controls.py](../results/query_token_staged_20261005/mechanism_followup/controls.py)、[CONTROLS.json](../results/query_token_staged_20261005/mechanism_followup/CONTROLS.json)。

**1. Probe 能從真實 latent 讀出 cross 成功所需的資訊。**

在相同 100 場景／1,024 候選上，保留同一計分公式：

| 輸入／讀取方式 | S@9 |
|---|---:|
| 真實幾何→score | 83/100 |
| 真實 latent→原始 probe→score | 80/100 |
| 真實 latent→目前 adapted probe→score | 80/100 |
| 第一階段 Q0 的預測 latent→目前 probe | 平均 46/100 |
| 真實 residual 閉環更新 | 平均 45/100 |

前三列使用未來真值，只是離線診斷，不能部署；固定排名與閉環也不能作逐誤差項相加。00007 的 14 個有解場景、00014 的 20 個有解場景，兩種 probe 在真實 latent 上全部第一名成功。這排除了「probe 普遍不會解讀這些 cross 模板」作為主要說法。缺口集中在預測 latent 與 probe 的介面：仍不能區分預測本身缺少必要資訊，和 probe 無法讀取預測 latent 的分布；不能據此說 probe 完全無責任。

**2. 00014 的更新確實把成功 query 朝錯誤方向推。**

以下固定原 base 選的一條失敗 support，對相同 query 比較。00014 有 24 個非空 support 場景，其中 19 個 query 有成功候選；三 seeds 共 57 次有解比較。這不是 57 個獨立場景。

- 57/57 次：失敗 support 的真實距離比 Q0 預測更遠，但成功 query 的真實距離比 Q0 預測更近（各自在該場景／seed 內平均時間與對應 actions）。
- 其中 56/57 次：updater 卻把成功 query 的平均距離也增大。
- 成功 query 的平均必要距離修正為 −15.56 px，實際更新為 +3.10 px。
- 以現有 scorer 的 soft 接觸權重重視相關時間、沒有新增 Near 門檻時，必要修正 −16.92 px，實際更新 +3.88 px，方向仍相反。
- 成功 query 的表面距離 MAE 在 56/57 次變差（各 seed：18/19、19/19、19/19）。
- 完整距離軌跡的「實際修正」與「通往真實幾何的必要修正」平均 cosine：失敗 query +0.286，成功 query −0.384；成功 query 在 55/57 次為負。改用真實 latent 經 probe 的幾何作參考，成功 query 仍為 −0.372。

因此，這個現象不是只看一個平均 L1 推論出來的，也不只是原始幾何和 probe 目標不一致。**觀察到的幾何結果符合「把失敗 support 的修正過度套用到本來會成功的 query」**。但並未直接讀出模型內部因果規則，不能宣稱已證明 updater 在做均勻廣播或忽略 action。

**3. 排名變差主要伴隨成功分數被壓低，zero-residual 對照較輕。**

為避免每次比較到不同 action，本輪先用 Q0 固定挑出最高分成功 action 與最高分失敗 action，再比較同一對 action 更新前後的分數差。00014 的 57 對中，37 對 margin 變差，其中 33 對成功 action 分數下降；13 對失敗 action 分數上升，兩類可能重疊。平均成功 score 改變 −1.610、失敗 score −0.266，margin 改變 −1.344（原 log-score 單位）。

相同固定 support/query 的成功幾何 MAE：Q0 16.809 px，train/eval zero-residual updater 17.227 px，真實 residual updater 19.854 px。AUC：Q0 0.88149、zero 0.88127、real 0.87510。真實 residual 分支的問題較明顯，但兩分支是分別訓練，不能把差異解讀成單一模型只移除一個輸入的純因果效果。

**4. 不能把下一步簡化成「多放成功資料」。**

重新依原 dedup 索引核對訓練 metadata：29,349 個 updater query 中已有 12,702 個成功 actions（43.28%）；按實際的均勻 scene／query 抽樣，其預期成功權重約 42.69%。這是資料池與抽樣期望，沒有冒稱實際訓練每一步的抽樣比例。因此成功樣本整體稀缺並非已證實主因。更需要關心的是：訓練組合是否要求模型學會「這條失敗 support 的修正，對哪些 query 適用、對哪些 query 應相反或不適用」，以及 loss 是否懲罰破壞成功所需接觸幾何。

**5. 目前可定位的兩種問題。**

00007 的 Q0 幾何／排名本來已嚴重錯誤，updater 沒有有效修復；真實 latent 經 probe 卻可正確排序所有 14 個有解場景。00014 則提供更清楚的錯誤更新證據：模型修正失敗 action 的同時，破壞成功 action。不能把後者視為整個 cross 所有失敗的唯一原因。

本輪的離線 oracle 替換也顯示：在 00014 的 57 次固定 query 比較中，只用真值或真實 latent 解碼結果修復成功候選，或只修復失敗候選，都可由更新後 22/57 提升到 57/57。這是使用未來 labels 的非物理解碼干預，只能說相對分數兩側都有可修正的誤差，**不能拿來當方法效果，也不能據此唯一歸因到成功或失敗一側**。

後續最有辨別力的實驗應鎖定相同 predictor，分開改一個因素：先比較不同失敗 support 對同一成功／高分失敗 query 對的影響；再比較現有 loss 與直接保護相對成功分數／接觸幾何的 loss。需沿用相同訓練資料與 seeds、保留 Q0 和 train/eval zero-residual 對照，不能同時改 support、loss 與 predictor。若要主張 predicted-latent 的 probe 失配，另需在 train/calibration 上做對這個新 Q0／updater 的 probe 適配，不能拿真實 latent 的好結果直接否定它。以上後續訓練尚未提交。

### Cross 為何失敗：候選、起步與更新分開分析（2026-10-07）

**主要可觀察瓶頸：部分未見模板的成功候選被排得很後，updater 幾乎沒有救回；在 00014 還出現成功候選被更新壓後。** 這是失敗所在的定位，不等於已經證明某種資料量或 loss 設計是唯一根因。

原始分數、候選 labels、Q0 不變性、逐步選擇及停止條件的 100 場景 × 3 seeds CPU 核對全部通過。新增 [analyze_cross.py](../results/query_token_staged_20261005/analyze_cross.py) 與 [CROSS_FAILURE_ANALYSIS.json](../results/query_token_staged_20261005/CROSS_FAILURE_ANALYSIS.json)；幾何 oracle／配對首次更新分析為 [cross_diagnostics.py](../results/query_token_staged_20261005/cross_diagnostics.py)、[CROSS_MECHANISM_DIAGNOSTICS.json](../results/query_token_staged_20261005/CROSS_MECHANISM_DIAGNOSTICS.json)。未重新訓練、未新增 GPU 評估。

**一、絕對失敗率與 memory 的負收益是兩個問題。**

Residual 平均成功 45/100。三 seeds 合計 165 次失敗：112 次候選有解但 Q0 和更新後都未找到、48 次候選完全無解（16 場景 × 3 seeds）、5 次 Q0 能解但更新後掉題。分別佔失敗 67.9%、29.1%、3.0%。這些是相同場景的重複 seeds，不能當 300 個獨立場景。

候選有解但 Q0 未解的 114 次，updater 只救回 2 次（1.8%；within 是 15/132=11.4%）。兩次都是初始最佳成功排名第 10；原本排在第 20 名以後的 95 次，沒有一次救回。有解但仍失敗的案例，成功候選數中位數為 9，初始最佳成功排名中位數為 96（within 為 18 個、第 28 名）。因此 cross 修正需要跨越的排名差距較大。這是不同模板／難度面板的描述，不是相同難度下的因果比較。

**二、按模板看，00007 是既有的困難區，00014 是本輪 updater 淨退步來源。**

每模板 25 場景 × 3 seeds；表內為成功次數，分母 75：

| 模板 | 原 base | 固定 Q0 | Residual 更新 | Zero-residual |
|---|---:|---:|---:|---:|
| 00000 | 45 | 41 | 41 | 41 |
| 00002 | 66 | 70 | 70 | 70 |
| 00007 | 0 | 2 | 2 | 2 |
| 00014 | 30 | 25 | 22 | 25 |

00007 的 25 場景有 11 個候選無解；有解的 14 場景，成功候選數中位數只有 5。這個模板原 base 已失敗，Q0／updater 都未有效解決。00014 的三次 updater 掉題沒有任何救回抵銷，正好構成整體 real−Q0 的 −3/300（−1pp）；另外兩模板各一次救回抵銷一次掉題。

例如 seed 1 的 00014:026：Q0 最佳成功 action 排第 4。第一次嘗試失敗後，保持 Q0 的剩餘排名應是第 3，更新後變第 28，最後仍排在剩餘候選第 21，九次未解；zero-residual 分支第 6 次成功。這是具體修錯的案例。另一個掉題 00002:104 卻沒有單純的排名單調變差，不能用同一故事解釋全部五次。

**三、相同首次觀察已能看到問題，不需要假設多輪狀態先崩壞。**

對所有「Q0 第一次失敗且候選有成功 action」的 185 個 seed／scene 配對，Q0、real、zero 在第一步有完全相同分數並選同一 action。比較同一個首次 support 後、排除已試 action 的相同剩餘候選：AUC 為 Q0 0.68389、real 0.68034、zero 0.68232。Within 的對應值是 0.87202、0.87756、0.87230，方向不同。此平均未做難度匹配，也不把單一平均差當成普遍退化證明。

00014 的 51 個此類配對中，有 42 個的「最高分成功 action − 最高分未試失敗 action」分數差在更新後縮小。00007 的平均 AUC 0.44770→0.42337，而 00002 是 0.83208→0.84452。修正的適用性高度依賴模板；不能說所有 cross 場景都變差，也不能把現象全解釋成閉環 support 路徑逐漸偏離。

**四、平均 L1／幾何仍可能掩蓋成功候選的退步。**

固定原 base 的一條失敗 support、相同 queries，74 場景 × 3 seeds 的 Q0→real：L1 0.119118→0.119198（這輪 cross 整體沒有下降）；全部 query 表面距離 MAE 56.608→56.467 px；成功 query 32.266→33.808 px；AUC 在有定義的 58 場景 × 3 seeds 上 0.69478→0.68904。原始 probe 的 AUC 也下降 0.67695→0.67097，成功 query 幾何誤差也上升 30.688→32.267 px，故不足以歸因為 adapted probe 單獨失配。

尤其 00014：L1 0.121486→0.121400，整體幾何 36.287→35.567 px，但成功 actions 幾何 16.809→19.854 px（約 +18%）。00007 第一階段也有同類問題：原 base→Q0 的 L1 0.111128→0.110231、整體幾何 102.710→86.806 px，成功 actions 幾何卻從 27.749→44.834 px，AUC 0.54305→0.45846；updater 再降到 0.43332。此處平均預測改善確實沒有對準成功排序。

以上都是固定 support 的離線診斷，不能當作每一步閉環 latent 誤差。場景配對 bootstrap（保留三 seeds 群組、條件於既有模板與模型）對 cross AUC 差 −0.00574 的 95% 區間約 [−0.01302,+0.00092]；先前 S@9 差 −1pp 的區間也跨零。證據支持追查這些失敗模式，尚不足以宣稱整體 cross 更新必然有害。

**五、真實幾何 oracle 排除了什麼？**

用保存的真實軌跡表面距離，代入同一個固定計分公式，在相同 1,024 候選上重排：cross S@1=81/100、S@9=83/100，候選池最多可解 84/100；00007 有解的 14 題與 00014 有解的 20 題全部在第一名成功。Within 同樣檢查為 S@9=96/100、候選有解 99/100。

這只是一個使用未來真值的離線診斷，不能當可部署結果。但它顯示：若拿到正確幾何，現有幾何→score 公式足以解大部分有解場景。因此主要缺口位於 predictor／probe 產生的幾何及其更新，而非計分公式本身完全無法辨識成功。這個 oracle 同時繞過 predictor 和 probe，**不能再由它區分兩者各自責任，也不代表加某個 geometry loss 就必然達到 83%。**

目前最合理的工作假說：均勻 query 的平均 latent／幾何監督，以及隨機失敗 support，學出的修正對某些未見模板不能保留成功候選所需的接觸幾何。模板特定的 Q0 誤差是起點，updater 在 00007 無法補救，在 00014 又削弱成功候選。仍待分開驗證 objective、support 分布與 probe 解碼失配；現有資料不能確認「多收同分布資料」或「加長訓練」會解決。K=1/2/4/8 的共同診斷雖有八場景，64-query 子集中可定義 AUC 的僅一場景 × 3 seeds，不足以支持跨深度結論。

下一步應先做固定 Q0／support／候選的成功-vs-高分失敗幾何與 margin 檢查；若要分離 predictor 和 probe，使用真實 latent 經同一 probe，再與真實幾何 oracle 比較。不能直接從目前结果跳到重訓或任意 Near 門檻。

### Within 為何仍失敗：逐題歸因（2026-10-07）

**主要瓶頸是：成功 action 本來就排得不夠前，而 updater 的修正不足以把它提升到九次嘗試內。** Within 整體從固定 Q0 的 55% 提升到 58%，所以不能把所有未解題都稱為 memory 破壞。下面把「剩餘失敗」與「更新造成的退步」分開。

本次 CPU 分析重讀全部 100 場景 × 3 seeds 的原始候選分數與閉環紀錄，重新核對候選 ID／label、初始 Q0 分數、逐步 argmax、去重、success-stop 與重播旗標，全部通過。程式與逐題證據：[analyze_within.py](../results/query_token_staged_20261005/analyze_within.py)、[WITHIN_FAILURE_ANALYSIS.json](../results/query_token_staged_20261005/WITHIN_FAILURE_ANALYSIS.json)。以下 300 次是同一批 100 場景重複三 seeds，**不是 300 個獨立場景**。

**一、失敗主要發生在哪一層？**

| Residual updater 未解出的原因分類 | 三 seeds 合計 | 平均每 100 題 |
|---|---:|---:|
| 候選池有成功 action，但 Q0 與更新後都未在九次內找到 | 117 | 39 |
| Q0 本來能解，更新後反而失敗 | 6 | 2 |
| 1,024 個候選中完全沒有成功 action | 3 | 1 |
| 合計失敗 | 126 | 42 |

99/100 場景都有成功候選，因此「沒有抽到成功 action」不是此面板主要限制。直接由更新造成的掉題只佔失敗的 6/126（4.8%）；最大部分是已有的排序問題未被修正。Q0 原本未解、但候選有解的 132 次中，updater 救回 15 次（11.4%），同時掉了 6 次原本能解的案例，淨增加 9 次成功。

**二、修正通常只能救回本來已接近前列的候選。**

| Q0 最佳成功 action 的初始排名 | 評估次數 | 更新後九次內成功 |
|---|---:|---:|
| 1–9 | 165 | 159 |
| 10–20 | 47 | 14 |
| 21–50 | 55 | 1 |
| 51–1,024 | 30 | 0 |

仍失敗但候選有解的 123 次中，成功候選數中位數為 18；最佳成功 action 的初始排名中位數是 28，最後一次打分的全候選排名中位數是 27。比較最後同一組未試候選，70 次修正方向有改善、37 次變差、16 次不變；方向改善常不足以跨過嘗試預算。這是對已失敗案例的描述，不是獨立的因果實驗。

**三、平均 prediction 指標改善，不代表成功候選所需的誤差改善。**

固定原 base 選出的第一個失敗 support，使用相同 queries，77 個有 support 場景 × 3 seeds：

| 指標 | 固定 Q0 | 真實 residual 更新後 |
|---|---:|---:|
| Query latent L1 | 0.084939 | 0.084857 |
| 全 query 的目標表面距離 MAE（px） | 12.816 | 12.400 |
| 成功 query 的目標表面距離 MAE（px） | 5.842 | 6.068 |
| AUC（76 個有定義場景 × 3 seeds） | 0.86033 | 0.86458 |

L1 只降約 0.096%；整體距離誤差改善，但成功 actions 的距離誤差略差約 3.9%。原始 probe 也出現成功 query 距離誤差上升（5.354→5.586 px），所以不能把現象只歸咎於 adapted probe。這些是固定一條 support 的離線診斷，**不是閉環每一步的 latent 誤差紀錄**，不能直接宣稱已證明每個失敗的成因。

實作上的合理解釋是：目前 loss 為 latent L1 加 original-probe 幾何監督；幾何項平均動態物體／採樣時間的位置和角度，沒有直接要求成功 action 排在高分失敗 action 之前。Query 均勻抽樣、support 隨機抽失敗，也沒有刻意對準部署時最容易被高估的候選。因此模型可以改善平均軌跡，卻未充分修正排名前端。**這是由實作與數據共同支持的假說，尚未完成 loss 與 support 分布的因果拆分，也不能直接推論資料量不足。**

**四、第一階段也留下局部退步，不能全算 updater 的責任。**

模板 00019 的 10 場景 × 3 seeds：原 base 解出 21/30，固定 Q0 13/30，更新後 12/30。相對原 base 少掉的九次成功中，八次的淨差已在第一階段出現。例：seed 0 的 00019:034，最佳成功排名由原 base 第 5 降到 Q0 第 31，updater 沒救回。第一階段整體仍是 55% 對 base 54%，所以這是模板特定退步，不能說整體起步都變差。

確實也有錯誤更新，例如 seed 0 的 00020:193：Q0 最佳成功排名第 4，觀察第一個失敗並更新後，在相同未試候選中降到第 71（若保持 Q0 為第 3），最後九次未解。三 seeds 所有此類由能解變不能解的案例只有六次；它們值得追查，但不是剩餘失敗的大宗。

**現有資料不能支持的結論：** 不能斷言 updater 隨觀察次數增加必然崩壞。在可共同比較 K=1/2/4/8 的八場景中，AUC 子集只有五場景 × 3 seeds，平均約 0.8750／0.8742／0.8737／0.8756，未呈現單調退化；樣本很小。也不能說「多試幾次沒用」：固定 20 題壓力測試的 residual 三 seeds 合計成功從 S@9 的 36/60 增至 S@20 的 46/60，但這不能外推成全部 100 題的 S@20。

下一個最有辨別力的驗證，是在固定 predictor／候選／support 的条件下，直接量測成功 action 相對高分失敗 action 的 score margin 與接觸相關誤差，再分開對照 ranking-aware loss 和 policy-selected support。先驗證修正目標是否對準排名前端；不能僅由這批結果決定加資料或加訓練時間。本次未新增訓練或 GPU 評估。

### 問題與目前證據

第一階段回答：沒有任何觀察時，這個架構能否預測任意 action，並保有可用的 ranking？第二階段回答：在同一個起點與固定 predictor 下，利用觀察到的 residual 更新 tokens，是否真的提高選 action 的能力？

動機是 v1 B 的第一步成功 5/20、base 7/20；B 不更新時初始排序在 9 次可解 12/20，實際更新只解 11/20。00000:050 的 base 成功排名 3、B 初始排名 8，更新後 20 次未成功。這不是缺少 Q0 loss 的因果證明；新方案讓起步品質與更新品質可以分別測量。

### 第一階段：訓練沒有 support 的 Q0 predictor

輸入為 initial scene latent、任意 query action，以及全域可學習的 Q0；輸出該 action 的完整預測 latent。每個場景都從相同 Q0 開始，但 predictor 同時讀取場景與 action，所以不是所有 action 共用相同預測。

- 沿用 32×256 tokens 與現有 predictor cross-attention 結構。Q0 從第一步就啟用，沒有 support 時也不 bypass。
- 從原 pretrained backbone 開始；訓練 backbone、Q0，以及 predictor 讀取 tokens 的 qn/q/kv/o adapters。Encoder、original geometry probe 與主要 adapted scorer 固定。
- 新 adapters 的輸出投影為零，所以訓練開始時預測等於原 base。Updater 在此階段不參與 forward，也不進 optimizer。
- 不需要 support。依使用者要求匹配原 base 的完整資料與訓練量：使用原 `cross_f1_train_s16r16_pca128` pack 的 52,276 條 trajectories、1,600 場景／16 模板；不再只用 memory 的 1,200 場景／29,349 query 子集。成功與失敗 actions 都保留，不用 Near 門檻。第二階段仍限原 1,200 個 updater-train scenes，support/query 互斥。這兩個階段的資料角色必須分開登記。
- loss 沿用 latent L1 + 0.1×original-probe 幾何 loss。這輪不新增 ranking loss、probe adaptation 或 base 蒸餾，避免同時改太多因素。
- 三個 seeds，每 seed **8 epochs、effective batch 8 trajectories、52,272 optimizer updates、418,176 次 trajectory 呈現**。每個 epoch 對全部 52,276 條打亂、保留 6,534 個完整 batches、捨去尾端四條；匹配原始 base loop，不能只把 epoch 名稱或 episode 數設相同。
- 沿用原 optimizer 預算設定：AdamW peak LR 3e-4、weight decay 0.05、OneCycle pct_start 0.03、clip 1。Scheduler total_steps 必須改為 52,272。記錄逐 epoch loss／評估与最後權重；主要使用第八 epoch final checkpoint，不因 cross 結果挑權重。
- 顯存不足時用 microbatch／gradient accumulation 維持 effective batch 8 與相同 loss reduction／step 數；不能只增 batch 卻少跑 optimizer updates。Tokens 額外運算及幾何監督會影響時間，因此以資料呈現量及更新量匹配，不用相同 GPU 小時代替。
- 仍按目前規劃從已有 pretrained backbone 初始化，所以這是**額外一份原 base 大小的訓練預算**，不是包含原 pretraining 的總預算相同。Loss 仍保留已規劃的幾何項；原 base summary 的 aux_lambda=0，故只稱資料／訓練預算匹配，不稱完全重現原 base recipe 或純架構公平比較。
- 使用原 base 的既有 dev split 記錄第一階段曲線，保留原 train/dev 分界；不把 dev 當新 test、不依結果挑 cross checkpoint。原 memory-calibration 的 400 場景属于完整 base-training 範圍，因此第一階段不能再把其中 32 題稱為未見 calibration；它們只能是 stage-1 train-seen／stage-2 updater-held-out 診斷。第二階段仍使用該預先固定 32 場景、每場景最多 64 個 queries，比較完整三 seeds，分開報 L1、幾何與主 scorer AUC。完整八 epochs 仍不等同已證明收斂。

此 checkpoint 定義新的「固定 Q0 baseline」。必須同時比較它與原 base；若第一階段就退步，應先解釋這個差異，不能把之後的結果全歸咎 updater，也不能偷偷換一個在 cross 表現較好的 checkpoint。

### 第二階段：固定 predictor，只訓練 updater

對每個第一階段 seed 的相同 checkpoint，初始化 updater；在相同場景的互斥 support/query 上訓練：

1. 用當下 Qt、initial scene 與 support action 預測軌跡。
2. 觀察真實軌跡，計算 rt = true latent − 當下 Qt 的 prediction。
3. Qt+1 = updater(Qt, rt, action)。
4. 用同一個固定 predictor、更新後的 Qt+1 預測未見 query actions，計算 query loss。

Residual 使用當下 predictor／狀態，不用固定 pretrained reference residual。只快取真實 latent／geometry，不能快取會隨 tokens 改變的 residual。訓練保留 residual prediction→state→query 的完整梯度；predictor 的權重雖固定，仍需對 token 輸入求導，不能把整個 predictor forward 包進 no_grad。

**凍結清單：** backbone、Q0、predictor 的 token read adapters、encoder、全部 probes/scorer。

**可訓練清單：** residual_in、action_in、updater time/space embeddings、state/observation/FF norms、update attention 與 update FFN。Qt 是每個 episode 的動態狀態，不是測試中透過 optimizer 更新的參數。

Updater 初始設為 identity：attention output projection 與 update FFN 的最後投影（包括 bias）初始化為零。因此在 updater 訓練前，任意次觀察更新都保持 Qt=Q0、預測等於第一階段模型；需要驗證第一／第二個 optimizer step 的梯度能啟動各 updater 路徑。

主要 loss 是更新後的 latent L1 + 0.1×幾何 loss。**第二階段不把 L0 當正則項**：Q0、predictor 與 read adapters 已固定，L0 對 updater 梯度為零，加入它只會增加常數或改變整體縮放。L0 只用作不變性監控。

Support 首輪沿用現有隨機失敗 replay，requested K∈{1,2,4,8}、不足截短、均勻抽一個有效正 prefix 計算 query loss。三個 seeds 及所有對照匹配 scene/query/support/time draws。20,000 scene episodes、batch 2、10,000 updates，full BPTT；不要求十九條 support，不排除低 support 數的場景。這個抽樣偏重較短 prefix，記錄實際深度分布，不宣稱均等訓練各深度。

### 最小可歸因實驗矩陣

| 組別 | 來源與訓練 | 部署時行為 | 要回答的問題 |
|---|---|---|---|
| 原 base | 已有權重，不訓練 | 固定排名 | 原始能力參考 |
| Q0 baseline | 第一階段，三 seeds | 永遠使用 Q0 | 新預測架構的起步能力 |
| Residual updater | 各自繼承相同 Q0 baseline，第二階段三 seeds | 根據當下真實 residual 更新 Qt | 完整適應是否優於同一起點？ |
| Zero-residual updater | 相同 baseline、配對 updater 初始化／序列，第二階段三 seeds | 保留失敗 action／更新次數，residual 輸入永遠置零 | 真實 residual 是否提供額外資訊？ |

第一階段三次＋第二階段兩組各三次，合計 **9 次訓練作業**，不是九個完全獨立的從頭模型。每 seed 的兩個第二階段分支共享同一個第一階段 checkpoint、相同新 updater 初始化、資料抽樣與預算。

Zero-residual 對照在訓練和評估都置零，避免只在測試把輸入變成未見分布；仍知道哪個 action 失敗，故不是無資訊。Residual 對照移除了輸入資訊也改變部分參數的有效梯度／容量，需列限制，不能由單次差異作過強機制宣稱。不要使用打亂未來 query residual 等有洩漏風險的控制。

Updater 的凍結條件確保第二階段任何 checkpoint 的未觀察狀態預測與第一階段逐 action 一致。若 S@1 改變，先當作 implementation／讀取設定錯誤處理，不當作 updater 效果。

### 公平比較與評估

主要部署終點為 **S@9**，全部 100 個既有 cross 場景與全部 100 個 within 場景，各三 seeds；第一步成功、候選無成功 action、AUC 無定義的場景仍計入成功率分母。固定候選池、scorer、合法性規則、tie-breaking 與嘗試預算；遇成功停止、不重複 action，每個新場景重設 Q0。這些是已檢視的 development panels；within 是同模板新生成場景，非官方 fold。

依序回答三個比較：

1. Q0 baseline 對原 base：第一階段是否保有／改善預測與排名？
2. Residual updater 對自己的 Q0 baseline：固定相同 predictor 後，更新是否有收益？
3. Residual updater 對配對 Zero-residual updater：收益是否需要 residual，而非只靠 action／失敗順序？

Q0 baseline 的整段無更新軌跡可由第一步全候選分數推導，不需每一步 GPU 重算。各第二階段 checkpoint 的第一步分數必須與該 baseline 相同，這是必要工程 invariant。實際閉環的後續 action 路徑可以不同；那是政策效果，不能稱相同 support 的純資訊比較。

次要報告 S@1/3/5、逐題首次成功次數、逐模板效果、Q0／更新後的 L1／幾何／AUC／最佳成功排名，以及成功候選被推後的幅度。固定 base-policy failed prefixes K=1/2/4/8、同一 query 的診斷先限制在預先固定的 20 場景；跨 K 趨勢必須使用共同場景並列 coverage，避免不同難度群體造成假累積退化。

S@20 在固定 20 場景作超出八次更新訓練範圍的壓力測試，不作主要終點，不稱官方 AUCCESS。主要全 100 場景 S@9 優先於昂貴的多 head／多 variant 診斷。

三個 seeds 分別報告與給出配對差、平均／範圍。場景 bootstrap 區間條件於當前模板及模型；不把三 seeds×100 scenes 當 300 個獨立場景。少量模板不足以支持廣泛泛化主張。保留逐模板结果，避免 00007 這種集中退步被平均掩蓋。

方法及選模規則固定後，需要新的確認場景面板，預先核對 ID／scene-goal hash 無曝光、沿用合法性規則並在評估前凍結名單。只跑預先選定的方法與必要 controls、全部三 seeds。若無有效未見確認面板，結論限於 development data；不冒稱官方 test，也不在確認結果後再調參。

### Batching、資源與後續分流

第一階段沒有 support replay，但改為原 base 完整八 epochs／52,272 updates，不能沿用先前 10,000 updates pilot 的時間估計。原 base 記錄耗時 12,618 秒（約 3.5 小時，含評估）只是歷史參考，新 tokens／幾何 loss／硬體負載使速度可能不同。第二階段 predictor 固定但須保留對 tokens 的梯度，不能假定全程等同 no-grad inference。按相同形狀批次處理 scenes/support prefixes，真實資料可快取，residual 必須即時計算。訓練總 wall time 先用實際 smoke 測量各階段，再乘正式 updates；現有 v1 時間不能直接當新方案估計。

最多五張可分配 GPU 並行。每個 seed 的兩個第二階段作業以 aftercorr 依賴自己的第一階段 checkpoint；Q0 array 最多三張、兩個 updater arrays 各最多兩張，依賴關係確保合計最多五張。正式評估等所有訓練核對後進行：先六個 Q0 seed/split 評估，再十二個 updater seed/branch/split 評估，各階段最多五張；within/cross 均是獨立 array elements，不再以全部 cross 完成作 within 的共同前置條件。依使用者最新優先指示，舊 pending GPU 評估已 hold；舊 within 的 34414_6 已 requeuehold，確認完成的 111 個場景／條件結果可在重啟時跳過。Smoke 成功後 34414_7 也已 requeuehold，兩張 GPU 優先供新正式訓練。新兩個 split 完成並核對後自動 release；動作見新 run 的 PRIORITY_ACTIONS.json。

這輪不混入 joint fine-tuning 或 scratch。若固定 predictor 的 updater 有效，後續再加「從相同第一階段 checkpoint、fresh updater 開始共同訓練」的分支，並保留直接 L0 監督，測試放開權重是否增加收益。那時 L0 才有實際正則梯度，且第一步不再保證不變。

若固定 predictor 下仍無效，先分析固定 query 修正和閉環排名，再单獨比較 random support 與 policy support。Policy 候選池必須含成功與失敗，不能只在 failures 裡排名；資料不足先補收集，support/query 保持隔離。選 action 不看未來，遇成功停止；保存 action 序列重播時重算當下 residual。不能把固定舊 policy 的序列稱持續 on-policy。這個資料問題不與本輪架構更動同時改。

Scratch 保留作後續獨立問題，需要 random predictor 的第一階段有足夠且以 calibration 檢查的訓練預算，再進入相同 updater 實驗；不是沿用短預算失敗來否定架構。後續分支尚未提交，不包含在九次作業中。

### 提交前工程驗證

Q0 從第一步參與 forward；第一階段開始重現原 base；第二階段初始 updater 是 identity；stage-2 optimizer 僅包含凍結清單以外的指定 updater 參數；對 tokens 的梯度通過固定 predictor；第一步／Q0 分數在 stage 2 各 checkpoints 不變；residual 使用選 action 當下 Qt 的 prediction；支持與 query 隔離；batch／serial loss-gradient 等價；模型／state roundtrip、因果閉環重播、success-stop／去重、raw AUC 重算。Smoke 只檢查工程正確性，不以 cross 表現挑實驗。

---

## 以下為 v1 執行與更早提案的歷史紀錄

> 最新執行設定（2026-10-05）：使用者已明確指定 **residual = 真實 latent − 當下 predictor（包含目前 tokens）的預測**。正式實驗登記改為 [query_token_dynamic_20261005](../results/query_token_dynamic_20261005/manifest.json)，最新作業見 [STATE.md](../STATE.md)。下文是更早、使用固定 reference residual 的單組 prototype 審查，不能當作最新實作規格。

五組動態版本各 200 episodes 的 GPU smoke 已通過，包含配對初始化／抽樣、完整梯度、動態 residual、狀態與模型載入，以及閉環選擇／狀態／分數重播。正式十五次訓練已提交為 **34479（最多五張 GPU）**；cross 評估／彙整為 34480／34481，within 準備／評估／彙整為 34482／34483／34484。尚無正式訓練效果結論。

最新完成狀態（2026-10-07 更新）：十五次正式訓練、全部 cross／within 評估與兩份 summary audit 均已完成。數字見本文最上方的 ABCDE 結果表；下文僅保留方法歷史設定。

新五組為 A 凍結 pretrained＋tokens、B 可訓練 pretrained＋tokens、C 隨機 predictor＋tokens、D 可訓練 pretrained 無 memory、E 隨機 predictor 無 memory。各三 seeds，B/D、C/E 配對初始 predictor，五組配對場景/query/support 抽樣。Encoder 與 geometry probe 維持固定；pretrained comparison reference 只用來報 baseline，不生成動態 residual。

每組 20,000 個場景 episodes，每兩個場景一個 batch，共 10,000 optimizer updates；每場景最多八個 query。每條失敗序列均勻抽一個 prefix，算 L1＋0.1×幾何 loss。訓練每次重播都依當下權重及 tokens 重算 support prediction/residual，梯度穿過預測、residual 與所有狀態更新，不 detach。只重用既有真實 latent/geometry 資料，不重用固定 residual。部署時保存排名過程中已選 action 的預測，失敗後直接相減；獨立重播核對同一 batch32 預測、每次狀態與最後分數。

這是固定預算的隨機失敗序列 supervised training，尚未加入定期刷新政策軌跡的訓練。正式閉環評估會根據各模型當下排名選 action、每次失敗後更新。另報相同 reference-policy prefix 與相同 query 的診斷。主要閉環指標 S@9；S@20 包含較長更新的外推，不稱為標準 AUCCESS。最多五張 A6000 同時訓練；C/E 是否已收斂需看曲線，不能只憑二萬 episodes 判斷初始化優劣。

**訓練預算補充核對：** [原 base 與本輪的實際預算](../results/query_token_dynamic_20261005/TRAINING_BUDGET_COMPARISON.json) 顯示，原 cross base 用 52,276 條軌跡跑八個 epochs，合計 52,272 次 optimizer updates／418,176 次軌跡訓練呈現，記錄耗時 12,618 秒（約 3.5 小時，含驗證）。本輪每組只有 10,000 次 updates、約 159,660–159,687 次直接 query 監督，另有 memory support 的動態 residual 計算。五組匹配的是**額外訓練預算**，不是包含 pretraining 在內的總預算；B/D 已有原 base 的訓練，C/E 沒有。C/E 若輸給 B/D，不能因此判定 from-scratch 方法較差。這輪可比較短預算的學習行為，完整能力比較還需要較長且依 calibration 曲線檢查收斂的實驗。沒有因此更改目前已登記的作業。

動態 residual 使原本 `latest` 消融的含義改變：現在它重設 updater 的輸入到 Q0，但 residual 來自前一個當下 predictor，因此可能間接保留歷史；須稱為 **reset-update 對照**，不能宣稱完全只讀最新觀測。Zero/static 同樣只作輸入／狀態診斷，不作唯一機制證明。

先前 [query_token_ablation_20261005](../results/query_token_ablation_20261005/manifest.json) 的固定 reference 快取與五組各 200 episodes GPU smoke 已完成並通過，但未提交正式訓練；它們只保留作工程紀錄，不能當成這次動態 residual 方法的效果證據。

2026-10-05。使用者已確認「同一組 tokens 隨經驗更新，換場景從共同起點開始」的理解，要求先 review 再執行。本次檢查提案與已寫的 prototype，另做 CPU 結構／梯度核對。沒有提交 query-token GPU smoke、訓練或模型效果實驗；既有作業維持原狀。

**判斷：這個方向值得做 bounded pilot。資訊流合理，也更直接實作逐次累積經驗；但目前草稿不足以用單一成敗判斷 residual 或累積記憶是否有效。開跑前應補上更新長度一致的主要評估、資訊／累積性的對照，以及實際 GPU 狀態重播檢查。** 這份報告是 proposal/code review，不是新模型效果結果。

## 1. 這個方法實際會學什麼？

- 全域可學習的起點 `Q0`：32 個、各 256 維的 query tokens。
- 看到一次失敗後，以該 action 的 residual 作 cross-attention 的 keys/values，tokens 作 queries，得到新狀態 `Q(t+1)`。
- 候選 action 的 predictor 隱藏特徵再讀取 tokens，產生該 action 的預測 latent；固定 probe／scorer 產生分數，選尚未執行的最高分 action。
- 成功就停止；失敗且仍有嘗試預算，才觀察該 action 的真實軌跡、再更新。
- 換場景重設到 `Q0`。測試中改變的是 tokens 的數值狀態；cross-attention、predictor adapter 與 probe 的網路權重不作 gradient update。

目前實作的 residual 是 `真實 latent − 固定 base 的預測 latent`。它是對 base 誤差的一次觀測；不是「最新已修正 predictor 尚未修好的誤差」。這個定義有穩定的參考點，與已有 residual 資料語義一致，第一輪可保留。不要將它描述成每次更新後 residual 必定縮小，也不要將不同 action 的 residual 當作可直接相加的修正量。

候選 action 的初始畫面／action 仍進入 predictor，所以同一組 scene tokens 可以對不同候選產生不同修正。這不是把同一向量直接加到所有 action 的最終預測。

## 2. 提案合理的部分

**固定大小的狀態。** 每場景維持 32×256 個值，float32 為 32 KiB；不隨失敗次數增加而無限增加 predictor 的記憶輸入。一次新觀測仍需處理 residual，但不用每次重新編碼全部舊經驗。這是計算結構的優點，尚未量到端到端速度提升。

**訓練覆蓋了「更新」這個動作。** 草稿抽 K=1/2/4/8 條不同失敗，逐條更新，對每個 prefix 的相同 query 計算 L1＋0.1×幾何 loss，梯度穿過連續狀態更新。這比只訓練一次更新、卻希望部署能反覆使用，更符合方法本身。

**起點可控。** Predictor 的 tokens 輸出投影從零初始化；空觀測另有 bypass。初始完整預測等於 base，學到修正後也能保留「尚未有經驗就用 base」的政策。`Q0` 本身仍可收到後續更新損失的梯度，不是被永久凍結。

**資料邊界合理。** 所有 1,200 個合法訓練場景保留，沒有 Near 門檻。Support/query 是既有固定、不重疊的 action 群；query 的真實未來只算 loss。閉環選 action 的函式只接受初始 latent、action、初始幾何與當前 tokens，真實 future 在該 action 失敗後才進入 updater。

以上是程式與 CPU 檢查支持的結構判斷；不是 trained-model 能力證明。

## 3. 分開報告訓練長度內與更長更新；沒有十九條 support 的先決條件

**澄清：方法不需要預先備齊十九條 support，也不需要同時保留十九條。每次只讀新執行的一條 action 的 residual，舊經驗由固定大小 tokens 承接。** 二十次嘗試是 assistant 草稿的成本設定，不是使用者架構的必要條件；十九只是這個預算下可能的最長更新次數。

草稿訓練 K∈{1,2,4,8}，但閉環預算 20 次 action；連續失敗時，最後一次決策可使用 19 次更新後的 tokens。後半段屬未受訓練的長度外推。若這裡崩掉，不能直接歸因於 query-token 架構。

以下數字只描述預先切好的訓練 support 區，不是部署的 1,024 候選池，也不是部署時必須先收集的資料量。真實資料盤點：1,200 題中，1,079 題至少有八條失敗 support，只有 **226 題至少有十九條**；中位數十四條，最少兩條。不能為了補齊十九步而重複同一 action 假裝新經驗，也不應只保留那 226 題。

短序列訓練的 recurrent updater 可以繼續更新，沒有架構上的八次上限；較長序列是否有效需要測量，不必先要求每題具备同樣長的訓練歷史。

**建議修改評估登記（評估分層，不是開跑的資料門檻）：** 保留全部場景和「不足 K 就截短」的訓練規則；主要閉環結果限於前八次更新之內，例如 Success@9（最多八次更新後選第九個 action）。同一條二十次嘗試軌跡仍可報 Success@20，但它是長序列壓力測試，分開報第 1–8／9–19 次更新的表現。這個長度界線是訓練覆蓋範圍，不是新的物理 Near 門檻。

若未來目標是 100 次嘗試持續更新，需要另處理更長訓練序列或獨立的長度外推驗證；不能從二十步直接宣稱成立。

證據：[episode 與 loss](../pipeline/query_token_experiment.py)；[資料／CPU 核對](../results/query_token_proposal_review_20261005/STRUCTURE_AUDIT.json)。

## 4. 必須區分：用了 residual，還是只是多了一組可訓練參數？

目前 `update_state` 包含 action embedding、線性層偏置與 FFN，即使 residual 全零，tokens 也可以改變。即使不作更新，直接讓 predictor 讀取 learned Q0 與已訓練 adapter，也可能改變預測。空 memory 等於 base 的檢查，不能排除模型在第一次失敗後只啟用一個通用修正器。

這不是已找到 bug：action 與「已經失敗」本身也可能有資訊。問題是**tokens 有變、ranking 有變，不能單獨證明 residual 有用或歷史被有效累積。**

至少加入下列對照，先在固定真實 support 序列、固定 query 上比較：

| 條件 | 實作 | 問題 |
|---|---|---|
| Base | 無 memory 修正 | 最基本的收益參考 |
| 正常累積 | `Q(t+1)=Update(Qt,rt,at)` | 完整方法 |
| 只用最新經驗 | 每次 `Update(Q0,rt,at)` | 累積舊經驗是否比最新一條更有用？ |
| 零 residual | 同 action／更新次數，但 residual 置零 | 分數改變是否主要可由 action、更新步數與偏置產生？ |
| 啟用固定 Q0 | 第一次失敗後讀取 Q0，但不更新 | 是否僅靠額外通用 adapter 就有同樣效果？ |

這些是測試時輸入／狀態消融，可能有分布偏移。若某消融大幅變差，不能單獨當作機制的唯一因果證明；若將來要強聲稱「residual 是關鍵」，再補相同預算的訓練消融。第一輪不需要因此立刻新增大量訓練。

固定序列診斷後，再於相同場景／候選／嘗試預算比較 base、正常累積、只用最新經驗的真實閉環；後兩者可能走不同 action 路徑，這是政策效果，不再是相同 query 的純資訊對照。

## 5. 訓練與部署的 support 分布差異仍在

目前訓練逐條均勻抽「已知失敗」的 support，選取不依目前模型 ranking；部署卻執行當前模型最高分的 action。草稿的闭環程式確實會每次重排，但訓練 sampler 沒有模擬這個選取機制。

這不是資訊洩漏，也不使 supervised pretraining 無效；但不能說「訓練已經學習在自己選出的失敗之後恢復」。之前自然首次失敗與隨機失敗不同的問題，換成 query tokens 並不自動消失。

第一輪可保留便宜的隨機失敗訓練，明確定位為預訓練，並保留固定序列與真實閉環兩種評估。若只在真實閉環失敗，優先追查 policy 造成的 support/query 偏移，再考慮從 training scenes 收集 base／当前政策的真實前綴做後續資料實驗；不直接判定 tokens 方法無效。任何政策序列都須遇到成功就停止，不能跳過成功、繼續挑後面的失敗來冒充實際軌跡。

## 6. Loss 可以保留，但要清楚它實際加權了什麼

保留已有正面受控證據的 L1＋幾何監督是合理的。幾何 loss 仍是物件位置／方向，不是成功排序本身；本輪仍需要實際看 ranking 與閉環解題，不能只用 L1 或位置誤差選結論。

另外，均勻抽 K=1/2/4/8、再將該 episode 的 prefix losses 平均，**不是讓各更新深度有相同訓練權重**。依本輪各場景可用 support 數重算，第一步約占 47.04%、第二步 22.04%、第八步 2.81%。平均每 optimizer step 約有 3.692 個受監督 prefix。

這種偏重早期更新有理由，因為首次修正已是舊方法的弱點；不用因為不均勻就立即改權重。但必須按 prefix 報指標，不能只看整體平均，就宣稱後續累積學得一樣好。與 K1 舊版本同為 20,000 steps，也不代表相同的 forward／query-target 監督量或 GPU 預算。

## 7. 舊版本能當參考，不能把差異全歸因於 tokens

相較正在跑的 fresh memory，新版本同時改了：

- 舊模式看真實 latent＋residual，新 updater 只看 residual＋action／時空編碼。
- 每次編碼經驗改成固定容量的遞迴狀態。
- K1 訓練改成多個 prefix 的訓練，監督量與計算成本也改變。

第一輪應回答「這套完整方法是否帶來有用的閉環適應」。若之後要聲稱 query tokens 比原 memory 架構更好，需要匹配輸入與序列訓練的架構對照，不能直接拿兩個不同設定的均值當成因果證明。

初始 32 tokens 是有上限的工程選擇，目前沒有容量充分性的證據。先保留單一容量；若要追容量問題，再用驗證資料做小型預先設定對照，避免同時掃架構、loss、資料和 readout。

## 8. 程式檢查結果與尚缺項目

已做 CPU 結構核對：

- 三次更新的逐 prefix backward，與把 losses 相加後一次 backward，所有 trainable parameters 的最大梯度差為 **6.94e-18**（tiny model、float64）。目前的 recurrence 沒有因逐次反傳而意外斷梯度。
- 初始 tokens 能取得梯度；更新不就地覆寫起點，重設後只用最新 residual 與保留歷史的狀態不同。
- 非零 decoder weights 下，empty bypass 仍等於 base。
- 用合成的非零 decoder 檢查，確實存在「residual 為零仍更新」「啟用固定 Q0 就改變預測」的路徑。這是結構可能性，不是訓練後模型會忽略 residual 的證據。

尚未驗證：實際 GPU/bf16 K=1/2/4/8 的梯度、峰值記憶體、完整 probe readout，以及 closed-loop 的真實更新重播。現有 `summarize_closed` 能重播保存分數所選的 actions、success-stop、去重、預算和 area，但只檢查 tokens 有變，**沒有獨立重算 residual → tokens → score 的鏈**。

開跑前需要一個小型真實 GPU smoke：至少兩次失敗更新，確認同一 residual 序列的逐步／整段處理一致、儲存載入 tokens 後的分數一致、無未執行 action 的未來資訊流入、跨場景 reset、每次失敗後重新排名、網路權重不變。不要只要求 smoke 能執行完成。

## 建議採用的第一輪

保留 32×256 tokens、固定 base residual、從零初始化、全部 1,200 題、L1＋幾何監督與逐 prefix 訓練。**先修主要評估長度、補固定序列的資訊／累積性對照，完成 GPU 状態重播，再啟動。** 不需要先增加任意 Near 門檻、新 ranking loss 或額外複雜 gate。

閉環 20 題／split 是可控成本的探索面板，每題就占 5 個百分點，三 seeds 也不是三套獨立場景；不得用它當成穩健的泛化定論。先報逐場景、每次更新和 seed 的結果，再決定擴大。

程式與證據：[tokens 模組](../pipeline/query_token_predictor.py)、[訓練／閉環草稿](../pipeline/query_token_experiment.py)、[CPU 核對程式](../results/query_token_proposal_review_20261005/check_structure.py)、[核對結果](../results/query_token_proposal_review_20261005/STRUCTURE_AUDIT.json)。本次未修改 prototype 或訓練 manifest，以上修訂尚未實作／排程。


## 2026-10-05：已完成 A／B seed 0 的失敗歸因補查

這是保存分數的 CPU 重算，不是新訓練或新推論。可重現程式與資料：`results/query_token_dynamic_20261005/analyze_interim_failure.py`、`INTERIM_FAILURE_ANALYSIS.json`。固定面板與診斷面板 AUC 均由原始 labels/scores 重算核對；閉環原 base 的首次成功排名也與 trace 相符。

B 在沒有任何觀察、memory 尚未啟用時，已使用訓練後 backbone：20 題第一步成功為 5 題，原 base 是 7 題。依照 B 的初始分數一直嘗試、不啟用 memory 的反事實排序，9 次與 20 次內都能解 12 題；實際 full memory 都是 11 題。這個比較保持 B backbone 固定，但它是關閉 memory 的部署消融，不能代替從頭無 memory 訓練的 D 組。

唯一跨越這兩個成功預算門檻的損失是 `00000:050`：原 base 的首次成功排名 3；B 初始無 memory 排名 8；B full memory 20 次未成功。A 凍結 base 則在第 15 次成功，其第一次失敗更新後，未嘗試候選中的最佳成功排名從不更新時應為 2 掉到 32。這直接定位出此案例的更新有害，但不能推廣為所有場景的根因。

| 模板 | 共同 AUC 場景 | B 減 base 的平均 AUC |
|---|---:|---:|
| 00000 | 20 | +0.028088 |
| 00002 | 5 | +0.049831 |
| 00007 | 14 | -0.219812 |
| 00014 | 19 | +0.087095 |

B 的總平均退步集中在 00007；A／B 各有 33 個場景提高、25 個降低。模型／模板固定的場景 bootstrap 95% 區間：A 平均差 +0.008807，區間 [-0.018384, 0.037893]；B -0.010546，區間 [-0.055557, 0.032474]。這些區間不能代表跨 seed 或新模板的變異。

訓練程式確認：query loss 永遠在至少一次 support 更新後計算，沒有直接監督空 memory 的 query prediction；support residual 的預測仍有間接梯度。B 可能出現 backbone 與 memory 共同適應後，單獨 backbone 的部署第一步品質下降，但尚未藉由訓練消融證明。訓練 support 是隨機失敗 replay，部署則用模型高分失敗；loss 是 latent L1 加 original probe 幾何誤差，不是 ranking／閉環成功率。這些是已知設定差異，其因果貢獻仍待檢驗。

新 B 的 cross latent L1 由 0.117286 增至 0.119842；本輪並非「L1 改善卻 ranking 退步」的例子。Original head AUC 改善而主要 adapted head 退步，說明效果依賴 readout，但尚不能把根因定為 probe distribution shift。

既有 fixed-prefix full／zero／static 診斷在 K=1/2/4/8 各只有 8/7/5/3 個 AUC 可定義場景，跨 K 群體不同；static 保留啟用中的初始 tokens，也不是 backbone-only。這些消融僅作線索，勿據此斷言隨更新次數必然惡化。D 組 cross 與其餘 seeds 尚待評估；此次未修改模型、排程或正式實驗設定。


## Staged run 的執行證據

新 run：`results/query_token_staged_20261005/`。訓練／模型／評估實作 `pipeline/query_token_staged.py`，結構測試 `pipeline/test_query_token_staged.py`；正式入口使用 run/code 的 frozen copies。CPU 測試核對 active Q0、identity updater、凍結 reader 後對 tokens 的梯度、batch／serial 的預測與梯度、optimizer step 後 Q0 不變、zero-residual 輸入不變性、checkpoint roundtrip 與完整 epoch 預算；最大預測差 5.55e-16、梯度差 6.07e-18。資料核對與 manifest 保存於 DATA_AUDIT.json；正式權重不沿用 smoke。

GPU smoke 包含三個階段各 100 updates、真實 1,024 候選的 raw metric 核對，以及有失敗更新的九次閉環 causal replay。Gate 僅在 smoke 成功後提交完整九次訓練；九次訓練完成另核對 frozen weights、配對初始化／draws、parent checkpoint hash，再執行 cross／within。精確即時作業 ID 與狀態以 run/jobs.json 和 STATE.md 為準。


Staged run 啟動更新：GPU smoke 已完整通過；三段各 100 updates 的訓練迴圈時間為 Q0 40.65 秒、residual 69.66 秒、zero 43.30 秒，峰值顯存 7.29／14.33／9.16 GiB。這些只作成本估計與工程證據，不是模型效果或正式訓練完成的證據。正式訓練不載入 smoke 權重。正式作業：Q0 34540（3 seeds）、residual 34541（3）、zero 34542（3）、training audit 34543、Q0 eval 34544（6 seed/split）、updater eval 34545（12）、cross/within summaries 34546/34547、完成後釋放舊評估 34548。


## 2026-10-06：完整訓練完成、初步 calibration 與評估修復

九個正式模型與 TRAINING_AUDIT 均完成。Q0 三 seeds 各 52,272 updates，兩種 updater 各三 seeds／10,000 updates；本次獨立重新核對九個 checkpoint SHA256，並從 raw scores 重算全部六個 updater 的 final calibration AUC。

| 三 seeds 平均、相同 32 場景 | AUC |
|---|---:|
| 各自的固定 Q0 baseline | 0.813637 |
| 真實 residual updater | 0.818091 |
| Zero-residual updater | 0.812641 |

每個 seed 的真實 residual 均略高於自己的 Q0 和 zero 對照。此面板只有一條固定失敗 support，32 個場景是 updater-held-out，但已用於第一階段 predictor 訓練，不能視為全模型未見資料、cross 結果或閉環成功率。真實 residual 減 Q0 的平均差 +0.004454，對當前模型／模板條件化的場景 bootstrap 95% 區間 [-0.000185, 0.008837]；真實 residual 減 zero 為 +0.005449，區間 [0.000988, 0.009707]。區間不包含模型重訓或新模板的不確定性，且是已檢視的 development 面板。

第一階段 historical base dev 400 條軌跡的 latent L1：初始原 base 0.119949；final seeds 0/1/2 為 0.122556／0.120937／0.120963，平均 0.121485。沒有因完整八 epochs 就獲得 latent L1 改善；新 loss 另有幾何項，不能據此直接推斷 ranking，也不能說訓練不足已被證實排除。

正式評估第一次輸出前，`static_case` 未建立目錄就呼叫 `save_npz`，六個 Q0 評估因此失敗。這是 assistant 的輸出路徑實作疏漏，原 smoke 沒有直接覆蓋新的 static-only 儲存路徑。維護中的 source 已補 mkdir，新增測試驗證第一次存檔與重啟不重算。為保留已登記 code/checkpoint provenance，正式補跑以 `recovery_20261006/prepare_output_dirs.py` 預建所需目錄，繼續使用原 frozen 數值程式與權重；沒有重訓或改任何模型輸出。

新 Q0 評估 job 34625（六個 seed/split tasks），updater job 34545 的依賴已從失敗的 34544 改為 34625；檢查 scheduler 已由 DependencyNeverSatisfied 轉為正常 Dependency。補跑目前等待 GPU，尚不能聲稱已完成評估驗證。後續 summaries 34546/34547 與釋放舊評估 34548 保留。

證據：`results/query_token_staged_20261005/INTERIM_TRAINING_CALIBRATION_AUDIT.json`、`TRAINING_AUDIT.json`、`recovery_20261006/RECEIPT.json`。


## 2026-10-07 03:05 UTC：cross 主要閉環結果完整

修復後全部六組 Q0 評估完成；六個 updater 的 cross 主要閉環也全部完成。每組使用相同 100 個場景／候選、最多九次嘗試，第一步成功與候選無解的場景均保留在分母。

| 方法 | Seed 0 | Seed 1 | Seed 2 | 平均 S@9 |
|---|---:|---:|---:|---:|
| 原 base（同一模型） | 47/100 | 47/100 | 47/100 | 47% |
| 固定 Q0 | 42/100 | 49/100 | 47/100 | 46% |
| 真實 residual updater | 41/100 | 47/100 | 47/100 | 45% |
| Zero-residual updater | 43/100 | 48/100 | 47/100 | 46% |

獨立由 saved scores 重算 Q0／原 base 的首次成功排名，以及所有六個 updater 的逐步 argmax、去重、成功標籤與 success-stop。每個 updater 第一輪全候選 scores 均與同 seed 的 Q0 baseline 完全相同；所有 trace 的狀態／分數重播檢查通過。第一步成功數 Q0／兩 updater 各 seed 為 19、21、27，原 base 為 26；因此第二階段已確實保住第一步預測。

真實 residual 相對 Q0 的 S@9 淨變化各為 -1、-2、0 題；相對 zero 為 -2、-1、0 題。平均均 -1 percentage point，模型／模板固定、以場景為重抽樣單位且保留三 seeds 群聚的 95% 區間為 [-2.667, +0.667] percentage points。這表示此面板尚未顯示 residual 更新有用，也不能將微小負差宣稱普遍或顯著有害。先前 calibration AUC 的小幅改善尚未轉化為 cross 閉環收益。

原始核對：`results/query_token_staged_20261005/CROSS_PRIMARY_AUDIT.json`。這是完整 cross 主要終點的中期報告；全部次要診斷／within 尚未結束，總 SUMMARY 未產生。Within 的三個 Q0 已完成；updater 四組正在執行、兩組排隊，不以部分場景推估完整 within 成功率。
