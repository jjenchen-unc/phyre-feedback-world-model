> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Support 配方的 within 新場景評估

使用者已授權把 within 加入目前實驗。本輪沿用 Uniform、Near、Mixed 各三個 seeds 的九個已完成 checkpoint，不重訓模型。目的是檢查：同一套 support 訓練配方，在看過的模板上是否能改善新的場景，並與原 cross 面板的效果對照。

## 場景來源與曝光限制

原有十個訓練模板各有 100 個已編譯場景，均曾出現在既有資料中，沒有可直接使用且完全未曝光的 cache 場景。因此從原始、未修改的模板產生器中，取尚未被編譯或記錄的參數組合。

使用模板 00003、00005、00006、00011、00016、00017、00018、00019、00020、00021，各十題，共 100 題。它們是本輪續訓實際使用的十個模板。先套用各模板原有的 required/excluded solvability flags，再以 seed 20261004 固定抽樣，不依九個模型的效果、候選池是否有解或 support 分類挑場景。

多數模板要求 `BALL:GOOD_STABLE`；00020 的原始設定沒有可解性要求，本輪也不額外替它加上。部分模板原先還做全體場景 diversity 子集選擇，本輪沒有重新執行該步，而是在剩餘合格場景中均勻抽樣。

每個新 task ID 必須不在全部 8,337 個已編譯任務，以及既有 manifests 的 2,500 個已記錄任務中；另外對場景完整幾何、目標物件 ID 與關係計算 hash，排除換 ID 的重複場景。也確認產生器重建既有編譯任務時，場景／目標 hash 完全一致。未改動官方 compiled tasks、cache 或模板程式，新任務以獨立 binary 交給 simulator。

因此這是**相同模板的新生成場景評估**，不是官方既定 within fold 的 benchmark。和 cross 的模板組合、場景挑選方式仍不同；within 有效而 cross 無效可提供泛化線索，但不能唯一歸因為跨模板轉移。

## 相同的模型與 action 程序

九個 memory 直接引用 `support_utility_20261004/models/` 最後一步的 checkpoint；逐一核對 2,000 steps、上游 manifest 與權重 hash。Base、原 probe、三個 adapted probes 與 PCA 都固定，不在新 within 上調整。

每個新場景使用原 action cache 的前 5,000 個 action 向量，以相同 action mapper／occlusion 規則保留合法 action。此時只產生初始影像與幾何，不模擬成功標籤，也不以結果篩候選。

為保持歷史 cross shortlist 的生成程序，先使用原 base＋原 probe 的 vectorized legacy-tanh 分數穩定排序，取前 1,024 個。這個 legacy 程序只用於候選池生成；正式排名仍使用與目前 cross 相同的 adapted probe 和穩定 float64 log scorer。若不足 1,024 合法候選，資料作業會停止，不能偷偷補挑容易場景。

對這 1,024 個 action 全部實際模擬取得成功標籤，記錄 1,017 physics steps、每八步一幀，共 128 幀。沿用 pinned LeVJEPA、overlap 編碼、固定 PCA128 與 32 個時間點。保存整條真實 latent，以保持成功／失敗 query 的 L1 評估與 cross 一致。

初始影像與 recorded rollout 第一幀須一致。工程測試也檢查 encoder 的時間因果性、query future 不進入 probe 輸入、空 memory 等於 base，以及 base cache 的精確重現。

## 評估內容

主要 head 固定 adapted probe seed 0，其餘兩個 adapted heads 與原 head 都保留。每個模型都比較同 head 的 base，使用全部相同候選和相同真實結果。

| 面板 | Support 與 query |
|---|---|
| 自然第一次失敗 | Base＋主要 head 的第一個 action 若失敗，提供該次經驗；其餘 1,023 個 action 作 query |
| 精選 support | 候選按原 20/40/40 action 群切分；固定 query，分別提供一條 N、H、F |

N/H/F 的定義、top20 高分門檻、穩定同分排序、donor hash 選取與原 cross pilot 相同。精選面板只在三類都存在的共同場景比較，實際 coverage 要等新 rollout 和 base cache 完成才知道；不能預先宣稱全部 100 題都可比較。

自然面板的第一步已成功、沒有成功 query，以及 AUC 無定義的題目都保留流程統計。主要分析是有自然失敗且 AUC 可定義的場景平均差，分母另報。主要對照為 Near−Uniform、Mixed−Uniform、Mixed−Near，也報 memory−base。

保存逐 query 分數、幾何、latent L1/L2、成功／失敗分數變化、排名與正確配對被改錯的比例。結果按模板、seed 分列；場景 bootstrap 區間條件於這十個模板與已訓練模型，不把 action、head 或同場景 donor 當成獨立場景。沒有在此輪加入 K4 或 100 次閉環決策，不能將離線 ranking 改善當成閉環成功率改善。

## 作業鏈與驗證

Run：`results/support_within_20261004/`；大檔位於 `/mnt/meg/jjenchen/phyre-unc/support_within_20261004/`。所有程式凍結在 run/code，保留 SHA256SUMS。場景 selection SHA 為 `879f6bec3143f5102f8a5582508450f3f381eb03bebba87422c5fe41f4134a27`。

| 作業 | Job |
|---|---:|
| 選新場景與曝光核對 | 34405 |
| 兩場景各八 action 的端到端 smoke | 34406 |
| 全部新場景的合法候選初始影像 | 34407 |
| Base shortlist 預篩 | 34408，兩個 shards |
| 獨立曝光與 smoke 模擬核對 | 34417 |
| 全部 shortlist rollout | 34409，等待預篩與獨立核對成功 |
| Latent 編碼 | 34410，兩個 shards |
| 建立 pack | 34411 |
| 同一組四個 heads 的 base cache | 34412，兩個 shards |
| 凍結正式評估 manifest | 34413 |
| 九個既有 checkpoint 的 within 評估 | 34414，最多兩個 GPU 同時執行 |
| 原始陣列核對與完整彙整 | 34415 |

Smoke 已通過：實際模擬、編碼、pack、base cache、memory 評估及 16 個輸出陣列重算皆成功，兩個測試場景都有自然第一次失敗。獨立核對另確認 100 個場景無記錄曝光，16 個 action 的初始影像與模擬標籤一致。Smoke 沒有同時具備 N/H/F 的場景，因此尚未直接覆蓋精選 support 路徑。正式候選渲染已完成；後續依賴作業的即時狀態以 jobs.json、STATE.md 與 Slurm 為準。

全量資料須重新收集，不能把原 cross cache 直接當成 within。目前模型評估已有部分完成，見下方中期結果。兩份結果分開保存。


## 中期結果：2026-10-05 UTC

舊 support-sampling 模型已有四個 checkpoint 完成 within 評估：Uniform、Near、Mixed 的 seed 0，以及 Uniform seed 1。另兩個 seed 1 正在執行，三個 seed 2 排隊，完整九模型彙整尚未完成。這不是新的 dynamic query-token 實驗。

獨立從原始 truth 與預測分數重算主要 head 的自然面板 AUC，並核對 query 一致、support/query 不重疊與 checkpoint provenance。100 個場景中，23 個第一步成功而没有 support；77 個有失敗 support，其中 76 個剩餘 query 同時有成功與失敗，可以計算 AUC。下表是這相同 76 個場景的逐場景 AUC 平均。

| 模型 | 平均 AUC | 相對 base |
|---|---:|---:|
| Base | 0.842029 | — |
| Uniform seed 0 | 0.862352 | +0.020323 |
| Near seed 0 | 0.862316 | +0.020286 |
| Mixed seed 0 | 0.862915 | +0.020885 |
| Uniform seed 1 | 0.863064 | +0.021035 |

這支持目前已完成模型在此 within 面板的平均離線 ranking 有改善；三配方 seed 0 差異很小，不能據此認定 Near 或 Mixed 更好。尚未做完所有 seeds，也沒有此輪閉環成功率結果。模型沿用舊 R0 續訓、Near 條件篩選的 171 個訓練場景，因此結果只適用於這批舊模型，不能當成新 query-token 或全 1,200 場景重訓的結果。此面板為同模板的新生成場景，非官方 within fold；和 cross 的差異不能唯一歸因於模板泛化。

原始分數獨立核對：`results/support_within_20261004/INTERIM_NATURAL_AUDIT.json`。
