> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Multi-turn 加速補驗與計算時間評估（2026-10-14）

本輪只做工程驗證與計時，沒有啟動正式的多 fold 方法比較。使用 RTX A6000 48GB，最多同時使用 6 張 GPU。原始記錄在 [multiturn_validation_20261014](../results/multiturn_validation_20261014/)，完整實驗設計見 [multi-turn plan](MULTITURN_UPDATER_ADAJEPA_PLAN_20261014.md)。

## 補了哪些檢查

起點改為這次選定的 Cross fold 1 Q0 λ=0、Within fold 0 Q0 λ=0.03。這裡的 λ 是 initial predictor 的幾何 loss 權重；updater 的 query loss 仍為 latent L1 + 0.03 × geometry。

- 新增每次 token 更新後都計算 query loss 的訓練路徑，整條更新歷史保留梯度。以 activation checkpointing 重算中間值，沒有截斷 BPTT，也沒有重用過期 residual。
- Cross／Within、real／zero 四個分支，各比較兩條獨立的 reference／fast 訓練歷史：12 個自然 slot，再加一個壓力 slot。每個 slot 有兩個 episodes；壓力 episodes 確實各更新 19 次。逐步比較 action、support、loss、gradient hash、更新後權重；檢查凍結 predictor／probe 不變。
- Real updater、zero updater、Q0＋AdaJEPA-style 在 32 候選池比較完整 20 次嘗試，檢查每輪 scores、action、tokens／適配權重相同。
- 在真正的 1,024 候選池跑滿 20 次嘗試，在第 1、2、20 次排名重新跑 reference 比對全部分數；每次更新都讀取實際選中 action 的觀測。
- Q0／base 的 AdaJEPA-style 補跑搜尋範圍最重的組合：lr=1e-3、每輪 3 steps、連續 19 次更新，核對 reference／fast 與場景重設。另將新 bare-base adapter 與原移植版比較 1／3 steps、最近 5 筆 replay 的 loss 與權重。

長序列壓力測試刻意選沒有成功 action 的池，確保能走完 19 次更新。這些結果回答實作是否一致、需要多少記憶體與時間，**不回答新 updater 是否提升成功率**。驗證使用既有受控 CUDA 設定，不能推廣成預設 CUDA 下所有長程訓練都 bit-exact。

第一次整合測試發現，從 updater 訓練切換到 TTA 時，凍結參數仍留下上一階段的 `.grad`。新 adapter 在凍結前先清空 gradients，之後重跑；原失敗檔案保留。這是新驗證流程的階段切換問題，沒有改寫先前正式實驗結果。

## 怎麼量時間

Initial predictor：每個 protocol 的 base／Q0 各跑 256 次真實 optimizer steps，前 16 次當暖機；另外量 pack 載入、400 筆 dev 診斷、checkpoint 寫入。

Updater：每個分支量 64 個自然 slot，前 4 個當暖機。計時包含資料讀取、自己選 support、重新計算當下 residual、各更新狀態的 query loss、完整 backward、optimizer。19-update 壓力測試另外列出，不混入自然平均。

Inference：reference／fast 交錯執行六次完整排名，排除各自第一次暖機。20-trial 計時另外扣除特別插入的 reference score audit；仍保留實際 observation 取得、更新和少量診斷成本。觀測是否命中 cache 逐筆記錄，不能把 cache 命中帶來的縮時當作模型加速。

每 10 秒記錄一次實體 GPU UUID 與程序。檢查有無別人使用同張 GPU；這不代表整台主機的 CPU、檔案系統與儲存裝置完全獨占。

## 實測與驗證結果

**全部通過；`SUMMARY.json` 的 `audit_complete=true`。** 四組訓練比較合計 52 對 slots／104 對 episodes；長序列 action、loss、gradient 與更新後權重相同。六個完整候選池 episodes 共跑 120 trials，18 次完整 score audit 全部相同。Q0 三個更新分支的初始 scores 也相同。AdaJEPA 的 base／Q0、19 次更新、3 steps／最大 lr 與重設檢查均通過。

541 筆 GPU telemetry 都未發現同張 GPU 上的其他計算程序。原先失敗／取消的版本保留，摘要採用修正後完成的結果。

| 協定／分支 | 自然訓練平均秒／slot | 10,000 slots 投影 | 無更新 episodes | 兩個 episodes 都更新 19 次時，秒／slot |
|---|---:|---:|---:|---:|
| Cross real | 2.61 | 7.26 小時 | 86/120 | 70.7 |
| Cross zero | 2.29 | 6.36 小時 | 86/120 | 50.4 |
| Within real | 2.46 | 6.84 小時 | 93/120 | 85.8 |
| Within zero | 2.12 | 5.89 小時 | 93/120 | 64.0 |

自然計時每列只有 60 個暖機後 slots。Cross 有 31/60 slots 完全沒有 optimizer 更新，Within 是 37/60；所以這裡的 10,000 slots **不是 10,000 次有梯度的 optimizer updates**。自然序列平均只更新 0.43–0.52 次／episode。訓練過程若改變這個分布，時間投影必須跟著改。自然訓練記錄到的 GPU 峰值約 real 20.1 GiB、zero 5.3 GiB；19-update 壓力序列亦在 48GB 卡上完成，但未單獨存該段峰值。

Initial predictor 暖機後平均約 0.279–0.316 秒／optimizer step；加上實測的載入、dev 診斷和 checkpoint 寫入，完整 8 epochs 投影約 **4.10–4.63 小時／模型**。這只量了各 256 steps，不能排除完整資料第一輪讀取或長時間 I/O 帶來的額外延遲。

| 完整 1,024 候選池 | Cross 20 trials | Within 20 trials |
|---|---:|---:|
| Real updater | 301 秒 | 298 秒 |
| Zero updater | 260 秒 | 262 秒 |
| Q0 AdaJEPA-style，1 step | 293 秒 | 283 秒 |

上述已扣除額外 reference audit。Real 的 19 次 observation 存取只有 1／3 次 cache hit，zero 則有 17／16 次；**不能由這張表認為 zero 方法的計算本身快了約 40 秒**。主要差異包含觀測 cache。

一般數值設定下，完整初始排名約從 10.6 秒降至 10.2 秒，省 2.4–3.3%。在受控 CUDA 長序列中，以更新後兩個 audit 時點比較同一模型狀態，排名省 11.8–14.8%；這只有各兩個配對時點，且 reference 總是在 fast 之後執行，不當作通用加速倍數。後者的 fast 排名平均約 12.8–13.1 秒／輪。兩種數值模式的絕對秒數不可直接混合比較。

Memory 更新本身約 **0.004 秒**；AdaJEPA 最近 5 筆 replay 的更新約 **0.054 秒／1 step、0.160 秒／3 steps**。因此把 updater 再加快，對整題速度幫助有限；重新預測全部候選才是主成本。

## 為什麼前一次估計不準

上一輪 24 次 matched base／Q0 訓練合計 **129.64 GPU-hours**；base 單次 3.19–9.60 小時，Q0 單次 3.52–8.62 小時。假設永遠滿載，單訓練除以 4 張 GPU 就是 32.41 小時，還沒有算 evaluation、資料準備與排隊。

先前用短程、順暢時段的速度推整輪，忽略了長時間的進度間隔與當時另有共用 GPU 的驗證工作。現在的記錄可以確認有很大的慢速區間，不能僅由進度間隔就判定全部來自 GPU 共用、I/O 或 scheduler 的哪一項。

這一輪還多了兩個不能沿用舊估計的因素：updater 改為各更新狀態都有 query loss、最長 19 次更新；evaluation 每次更新都要重排 1,024 個候選。舊的單次靜態排名耗時，不能拿來當完整 multi-turn episode 耗時。

## 新 folds 的資料成本

四個新 folds 的既有薄資料每個只有約 6,400 條 trajectories，不能當作完整的約 52,000 條 initial training 資料。但也不必把每個 fold 的全部 trajectories 重新 encode。

本次逐 task／action 盤點完整收集配方，重疊場景的既有 manifests 一致，並用現行 generator 抽樣重現；缺少的場景則只產生 action IDs，沒有啟動正式 rollout 收集。四個新 folds 合併、跨 initial／cluster 配方去重後，需要 154,051 個不同 task-action trajectories：134,757 個已在現有 packs，另外 1,412 個已有 standalone latent，剩 **17,882 個需要 encoding**。

另外抽查 32 個共享 trajectory：Cross／Within packs 的 latent 完全相同，存在的 standalone 檔也相同；額外 cache 檔案全部檢查 shape／dtype，PCA hash 與既有註冊一致。這是盤點與抽查，不是全部 latent 的重新計算驗證。正式準備時仍須建立各 fold 的明確索引、來源 hashes 與缺檔檢查。

可以重用的是固定 encoder／PCA 下的觀測表示；每個 fold 的 predictor、probe／scorer 訓練範圍仍要依該 fold 建立。若另行變更 encoder 或 PCA，這個重用估計就不適用。

新資料處理還包括 CPU simulation、資料整理及新的候選池。候選池不能直接沿用歷史 base shortlist：主實驗要用不依模型分數的固定有效候選，各 baseline 都重跑。

## 怎麼估算整輪

新增 8 次 initial training、12 次 updater training；主評估為 36 組 × 100 題。另有 600 個 AdaJEPA 設定搜尋 episodes、1,200 個線上歸因 episodes。共同 support 的離線診斷尚未固定 query 數，另外估。

靜態方法每題只需排名一次，共 1,200 次排名。四個動態方法共 2,400 題，若全部跑滿 20 次，共需 48,000 次排名，也就是約 4,915 萬次 candidate predictions。每次排名是 32 個 B32 batches。這是本輪主要的 inference 計算量。

時間投影使用「動態方法平均嘗試 5／10／20 次」三種情境；不是預測成功率或統計信賴區間。新的候選池尚未正式評估，不能把舊 shortlist 的成功率直接代入。

Updater 的投影只在自然訓練的成功／失敗分布相近時成立。模型訓練後的 policy 若變差、平均失敗次數變多，實際更新深度與訓練時間都會增加；不能只拿初始階段的平均值作完工承諾。正式訓練應在 1,000 slots 後，按實際更新深度、非零梯度 slots 與最近計時重新估剩餘時間。

以下使用平均 10–20 次嘗試，並沿用目前自然訓練深度的投影。不同 folds 的模型／scorer 仍可能改變耗時；bare-base TTA 的完整池成本以 Q0 TTA 作代理，沒有宣稱已量完每個方法每個 fold。

| 工作 | 數量／條件 | 投影 GPU-hours |
|---|---|---:|
| 新 initial predictors | 8 次完整 8 epochs | 34.5 |
| Real／zero updater | 12 次，各 10,000 slots；早成功比例維持目前程度 | 79.1 |
| 缺少的訓練 latent＋新主候選池 encoding | 17,882 條 trajectory＋600×1,024 個初始輸入 | 18.1 |
| 主評估 | 36 組，動態方法平均 10–20 trials | 102.3–202.6 |
| AdaJEPA 設定搜尋 | 600 episodes | 24.9–50.2 |
| 兩個線上歸因分支 | 1,200 episodes | 29.0–53.8 |
| **上述小計** | **尚不含新 heads、離線共同 query 診斷、排隊** | **287.9–438.2** |

資料 encoding 的投影採 30 個暖機後樣本，整條 trajectory 平均 1.36 秒（樣本 p90 2.47 秒），單個初始輸入平均 0.0665 秒。需要新增的 trajectory latent 約 26.7 GiB。CPU 同場景 simulation＋存檔平均約 0.72 秒；換場景第一次另有初始化成本。主候選池的 CPU render／labels、manifest 建立、大量 pack 複製與驗證尚未量整批完成時間；不要把這些默認為零。現有 index 可以重用，但新的 fold-specific packs／索引尚未建立。

Probe／scorer 新 fold 配方尚待寫入執行 manifest。歷史幾何 probe 配方耗時約 Within 12 分、Cross 62 分／次，資料量分別為 6,382／26,042 trajectories；這只是舊配方參考，不能冒充這次新 fold 的完整實測，也未含新的 predicted-latent scorer cache 製作。

**排程建議：只算新增 20 次模型訓練，目前投影約 113.5 GPU-hours；6 卡的理想值是 18.9 小時，實務先留 24–36 小時，前提是資料與 heads 已備妥。**

**若連主評估、TTA 搜尋與兩個線上歸因分支一起做，6 卡的理想滿載值約 48–73 小時，實務先預留 4–6 天；若只有 4 卡，先預留 5–8 天。** 這是上述條件成立時、考慮執行相依與一般餘裕的排程預算，不是量出的完工時間或保證上限。未量測的資料整理／new-head 工作需要另外補時，scheduler 排隊另計；離線共同 query 診斷必須先固定規模才可併入。若 updater 每題平均更新深度顯著上升，這個預算也必須上修。

GPU-hours 是分配一張 GPU 執行這個流程的累積時間，含流程中 CPU／I/O 等待；不是純 GPU kernel busy time。除以 GPU 數只是全程滿載、沒有相依等待的理想值。新 fold 的資料、head、initial predictor、updater 有先後依賴，因此實際 wall time 必定還受排程影響。

## 重現與原始證據

- [原始驗證與成本摘要](../results/multiturn_validation_20261014/SUMMARY.json)：由 `pipeline/summarize_multiturn_validation.py` 重新核對 raw histories 後產生。
- [舊訓練實際耗時](../results/multiturn_validation_20261014/HISTORICAL_TIMING.json)。
- [資料重用盤點](../results/multiturn_validation_20261014/data_reuse.json)、[latent 抽查](../results/multiturn_validation_20261014/reuse_latent_check.json)。
- 新 runtime：`pipeline/multiturn_runtime.py`；主要驗證：`pipeline/validate_multiturn_runtime.py`、`pipeline/validate_multiturn_tta_depth.py`。
- 各 job 使用的 frozen code、失敗與重跑輸出都保留在同一結果目錄；validation smoke weights 不可當作正式 trained updaters。
