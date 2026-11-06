> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Memory 失敗原因：獨立檢視

2026-10-04。使用者要求由獨立 agent 檢視。本報告直接檢查凍結程式、實際訓練抽樣紀錄、manifest、逐 query 結果及既有歸因實驗；沒有重新訓練、變更作業或修改實驗程式。Cross 新一輪尚有一個模型未完成，以下新效果只使用完整配對的 seeds 0、1。Within 尚無效果結果。

**目前最有證據的問題是：訓練要求 memory 降低平均預測誤差，但這不等於學會保護成功 action 的排序。資料與部署經驗的差異也確實存在；然而還不能把它、資料量不足或架構問題，指定成唯一根因。現有數字有足夠工程核對，較大的缺口是「尚未把幾個學習原因分開」，不是已發現整批結果算錯。**

| 使用者的問題 | 本次判斷 |
|---|---|
| 實驗不嚴謹嗎？ | 最新 matched pilot 的內部比較相當受控；但只允許對這個配方、初始化和開發面板下結論。缺少同條件的起點 R0、訓練集行為評估及資料規模對照，不能唯一歸因。 |
| 資料不夠多嗎？ | 經驗種類少、模板不均衡是事實；「只要再加資料就會好」沒有證據。這輪成功 query 並不稀少。 |
| 設定有問題嗎？ | 有已驗證的目標與決策落差，也有已量到的訓練／部署 support 差異。N 的定義沒有直接測到使用者想要的 support–query 適用性。 |

## 先把不同的「輸」分開

新 support pilot 是第一次失敗後的固定 query 排序；這不是閉環 AUCCESS。主要固定 adapted head 下，58 個可算 AUC 的場景、兩個 seeds 平均：

| 模型 | Ranking AUC |
|---|---:|
| Base | 0.6772 |
| Uniform | 0.6300 |
| Near | 0.6239 |
| Mixed | 0.6295 |

同 query 換成精選 N/H/F support 的面板，37 個可算 AUC 場景中，base 為 0.6729；九種「訓練配方 × support 類別」平均落在 0.6217–0.6272，N 並未明顯救回效果。每場景每類只有一條固定 donor，不能據此說所有 near-success 經驗都沒有資訊。

但前一輪 geometry memory 的閉環 AUCCESS 是 0.4622，高於 base 0.4541；Success@100 則是 63.8%，低於 68.0%。它能讓部分題更快成功，也會丟掉一些 base 能解的題。因此「memory 全面無效」不符合現有結果。

新舊 AUC 也不能直接比較：先前 geometry 表格的 0.6767 使用不同 K1 context、57 題分母及多 head 平均，新 pilot 的約 0.63 使用新自然 context、58 題和主要 head。訓練集合也不同。這些不是同一條受控 learning curve。

證據：[新 cross 配對結果與逐模板分母](../results/support_utility_20261004/INTERIM_CROSS_SEEDS01_DETAIL.json)、[既有 probe／geometry 對照](PROBE_GEOMETRY_RESULTS_20261001.md)、[閉環損失歸因](MEMORY_CLOSED_LOOP_LOSS_ANALYSIS_20261002.md)。新結果主要 head 的 2,292 份 AUC 陣列已用成功／失敗逐對比較重算；本次沒有獨立重算全部 latent 誤差陣列。

## 哪些原因有較強證據？

**1. 平均 latent 誤差不是成功排序的可靠替代目標。這有受控介入證據。**

過去相同資料、抽樣、預算與 R0 起點，只增加幾何 loss，適配 heads 平均 AUC 從 0.6413 升到 0.6767，latent L1 卻從 0.114315 變成較差的 0.114966。它證明「訓練在意什麼」會改變排序，而且更低 L1 不是必要改善方向；尚未證明某個新 loss 已解決全部問題。

目前程式仍將未來 31 幀、所有 latent 元素平均，另加 0.1 倍平均物件位置／bar 方向誤差。沒有直接要求成功者排在失敗者前、保護接觸關鍵時段，或在非空 support 不適用時保持 base。模型可以透過現有 attention／null token 少用記憶；問題是這個行為沒有被單獨驗證，而不是架構必然不能忽略記憶。

程式：[support_utility.py 的 train](../results/support_utility_20261004/code/support_utility.py)（324–330 行）、[geometry_loss](../results/support_utility_20261004/code/memory_geometry_followup.py)（43–63 行）。

**2. Near 訓練與實際第一個失敗的分布差異很大，但不是完整解釋。**

本次直接從保存的真實軌跡距離重算全部 74 個自然失敗 support：**70 個最小距離 ≥20 px、2 個 N、2 個中間距離**。Near 訓練卻每一步只看 N；部署遇到的幾乎全是另一類經驗。這是已量到的差異，不再只是猜測。這裡的「遠」沒有額外宣稱它符合 support 分區的 H 百分位規則。

可是精選面板刻意提供 N，三組仍輸 base。因此只能說自然 support 差異是合理的影響來源，不能說「改成 N 部署就能解決」。

**3. 使用者的適用性假設，這輪只測到其中一小部分。**

N 的意思是「這條失敗軌跡有某一個取樣時點離目標很近」。它不是「action 參數接近某個成功 action」，更不是「這條經驗的 residual 能修正目前這個 query」。一條 support 對 query A 有幫助，對 query B 可能有害；給 support 一個整體 N/H/F 標籤沒有描述這個關係。

實際模型已看見 query 初始 latent 和 action，並逐層讀取記憶；support 同時提供真實 latent 和相對 base 的 residual。不能把失敗歸因成「沒有 query conditioning」或「直接把 residual 複製到全部 action」。缺少的是適用性被學會的行為證據。

程式：[MemoryPredictor](../results/support_utility_20261004/code/common.py)（context_residual、encode_memory、forward）；[分類与抽樣](../results/support_utility_20261004/code/support_utility.py)（classify、episode）。

**4. Readout 仍是交互因素，不能宣稱已排除。**

本輪 Uniform 對 base 的 AUC 差，在原 probe 約 -0.0024，在三個 adapted probes 分別約 -0.0472、-0.0476、-0.0449。不是只壞在單一 adapted seed，但不同 head 下差距很不同。原 probe 的絕對 AUC 也較低，不能用較小差距說它比較好。

訓練幾何梯度來自原 probe，部署主要用 adapted probe；後者適配的是 base／原 R0，未適配新模型。此前適配已改善 base 和 R0，卻沒有消除 memory 差距。現有證據支持「讀出有影響」，不支持「所有問題只是 head 沒適配」。

## 資料究竟少在哪裡？

本次從九組完整 draws.json 重算，而非只看可用資料清單：

- 相同 seed 的三組場景／query／時間抽樣 hashes 完全相同；全部九組皆 2,000 steps，base／probe 不更新。
- 實際成功 query 比例為 **41.79%–42.51%**；每 2,000 步只有 **11–17 步** 沒有成功 query。「訓練幾乎只看到失敗 query」不符合這輪資料。
- 共同训练只剩原 1,200 場景中的 **171 題（14.25%）**。模板 00021 有 55 題，00020 只有 2 題。場景均勻抽樣仍會讓前者獲得較多訓練。
- 每場景可用 N 的中位數只有 **2 條**，至少四分之一場景只有 1 條。Near 每次訓練實際只看 **442–453** 條不同 support；Uniform 為 **1,407–1,431** 條。抽樣配方也一起改變了經驗多樣性，尚未分離內容與重複率。
- 最終 171 題的 **525 條 N 中，0 條初始就已接近**。先前完整盤點的 428/2,758 初始接近比例，不應拿來解釋這輪已筛選子集的失敗。

這支持優先懷疑資料的覆蓋、多樣性與 support–query 關係；不能直接證明總量不足。尚無固定資料配方的規模曲線，也沒有「训练場景已能改善排序，但新場景失敗」的同條件證據；目前無法區分學不會、過擬合或轉移失敗。

逐項資料、來源 SHA 和可重現程式：[DATA_FINDINGS.json](../results/independent_memory_review_20261004/DATA_FINDINGS.json)、[audit_existing_data.py](../results/independent_memory_review_20261004/audit_existing_data.py)。距離取自已保存 simulator 幾何，本次未重跑 simulator；訓練 N 的初始距離沿用已有獨立 DATA_AUDIT 核對的 inventory。

## 嚴謹性：已做到與尚未做到

本次讀取的凍結 code 七個檔案在共享伺服器全部通過 SHA256SUMS。程式有相同 query／head 比較、support/query 隔離、空 memory 等於 base、base 完整分數重現、query 未來不進 readout，以及權重／來源 hash 核對；既有 [DATA_AUDIT](../results/support_utility_20261004/DATA_AUDIT.json) 另外重算分類、分區與自然第一個失敗。**未發現足以推翻最新主要比較的實作錯誤；這不是對全部 pipeline 無缺陷的保證。**

主要推論缺口：

1. **沒有同一新面板的未續訓 R0。** 三組都從已會造成傷害的 R0 出發。Near 輸 Uniform 可歸因於本輪配方介入；三組輸 base 不能單獨證明是這 2,000 步新學壞，可能繼承了起點問題。
2. Cross 只有反覆使用的四個模板；三個 seeds 共用 base/R0。新的 selected 面板 39 題，更完全缺少 00014。不能把場景／seed 數當作獨立新機制數量。
3. 自然 AUC 排除第一次成功和剩餘候選無正負類可比較的場景。這個條件指標有意義，但不能換名為全體成功率。
4. Uniform 的自然 AUC 傷害很集中：00007 約 **0.5431→0.3689**；00000 約 **0.6075→0.6094**。均值負向不是所有模板都負向，也不能事後刪除 00007。
5. 2,000 步與一個 loss 權重沒有证明收斂。記錄的是抽到的訓練批次 loss，不能當固定 validation learning curve。多跑同一設定也無法自動回答根因。

Within 新面板可補同模板新場景的證據，但使用新生成場景及不同模板組合；within 好、cross 差只會加強轉移問題的線索，不能單独把全部差值歸因於模板外推。

## 最少還需要哪些能區分原因的檢查？

按資訊價值排序，以下是建議，**本次沒有啟動**：

| 優先序 | 最小檢查 | 可以區分什麼／判讀條件 |
|---|---|---|
| 1 | 將未續訓 R0 放進這輪相同自然與 selected 面板；保留全部新模型。另在固定訓練場景評估 matched support/query，清楚標成已見資料。 | 若新模型≈R0 且兩者都差，主要是繼承問題；若新模型退步，才追本輪更新。若訓練場景也無益，先查目標／優化；若訓練有益、新 within 無益，才有過擬合證據。 |
| 2 | 在預先固定的小型 query 面板，每類提供多條 donor；同時保留自然 support、empty、不同場景／同模板 donor。記錄每個 support 對不同 query 的收益與傷害。 | 若同一 N 對不同 query 正負皆有，且 support–query 關係比 N/H/F 更能預測收益，優先做適用性資料／目標。單一 donor 或 attention 權重不能回答。 |
| 3 | 在固定配方下增加不同場景／donor數，保持訓練預算一致；另用一個固定資料延長訓練控制。先鎖定 heldout評估與實用門檻。 | 新資料有益而額外重播無益，才支持覆蓋／多樣性不足；延長訓練就改善則支持優化預算不足。兩者都不改善仍不等於所有資料無用。 |
| 4 | 只有在上述控制後，做單一目標介入，或為新 memory 做 matched head 適配；base接受同預算適配。 | 若相同 latent 更換公平適配 head 後恢復，支持 readout 原因；若改目標才恢復排序，支持 supervision 原因。不要同時改資料、架構、head、loss。 |

第一次修正把可解 action 推出預算的現象已定位，並不代表造成那次修正的學習原因也定位了。現階段最合理的研究判斷是：**先補起點／訓練集行為與 support–query 適用性控制，再決定是否增加資料；沒有足夠證據直接重做架構，或只把訓練量加大。**
