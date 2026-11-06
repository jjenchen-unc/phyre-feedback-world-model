> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Memory 失敗歸因：完整實驗設計

2026-10-01 最新：probe 適配與幾何 loss 對照已完成且核對通過，見
[結果](PROBE_GEOMETRY_RESULTS_20261001.md)。Compact task-centric latent 接續
門檻通過、已授權，但尚未實作／提交。當前沒有執行或排隊的工作。

日期：2026-09-30。狀態：設計完成；階段 0/A 已完成且核對通過，見 [結果](ATTRIBUTION_STAGE_A_RESULTS.md) 與 [執行設定](ATTRIBUTION_STAGE_A_RUN.md)。2026-10-01 使用者另授權階段 F 的 probe 適配 pilot，已提交，見 [設定](PROBE_ADAPTATION_RUN.md)。其後使用者授權接續 loss／representation 測試；[幾何 loss 對照](MEMORY_GEOMETRY_FOLLOWUP_RUN.md) 已排入 afterok 工作鏈（34265–34268），等待 probe 彙整完成。Task-centric latent 尚未實作或提交，將依事前記錄的結果條件接續。

這份設計要回答：**memory 為什麼沒有改善 cross 解題？「預測更好」在什麼條件下能改善排序？**

我們不把一次實驗當成找出唯一根因。每個對照只歸因到實際改動的環節；若兩個因素互相影響，就報告交互作用，不把結果硬拆成各自占多少百分比。

## 1. 已知結果，以及這次要補的缺口

目前已確認：

- 空 memory 輸出等於 base，已排除新模型的空 context 偏移。
- 固定 query 診斷中，平均 latent L1 降低，但成功 query 的誤差可能增加。
- 穩定 scorer 可恢復小型驗證集的部分排序；在 100 題 cross 閉環，卻沒有改變已解題集合。
- 最新 cross traces 中，四個微調模型在 57 個可比較場景裡，有 31–35 個在第一次失敗更新後，就降低成功候選的排名。

尚未分清楚的是：

| 問題 | 要隔離的因素 |
|---|---|
| 平均 prediction loss 是否衡量錯了重點？ | 全體 latent 誤差、成功候選誤差、接觸幾何誤差的差別。 |
| Probe 是否讀錯模型產生的 latent？ | 固定 latent，改變讀出方式；不能只看真實 latent 上的表現。 |
| Scorer 是否不符合真正成功條件？ | 真實幾何經 scorer，是否仍把成功排錯；包含時間解析度問題。 |
| Support 是否被錯用到 query？ | 同一 query、相同 K，改變完整的 support 經驗。 |
| 多輪選擇是否放大錯誤？ | 固定共同歷史的預測效果，與各模型自行產生歷史的效果。 |
| 問題來自資料配方或訓練目標？ | 固定 query 抽樣、K、初始化、訓練預算，分別改 support 來源和 loss。 |

## 2. 全部實驗共同遵守的比較規則

### 2.1 同一個比較單位

固定 query 的一個比較單位為：

`場景 × 初始 action pool × query IDs × support 歷史 × checkpoint`

除了指定的介入外，其他項目相同。每次都記錄實際 K，不能只寫 requested K。先在場景內平均，再平均場景；同一場景的不同 prefix、support donor、成功／失敗配對不當成獨立場景。

成功 query 的真實軌跡、結果只可作監督目標或離線評估。若某個診斷使用 query 真實未來來選 support 或修正預測，必須標為 **oracle 診斷**，不能拿來宣稱可部署的改善。

### 2.2 三層資料分開

1. **D0：既有開發資料。** 現有 cross 100 題與先前 memory 驗證集，繼續用來除錯和分析。它們已影響研究方向，不再宣稱是乾淨的確認結果。
2. **D1：新抽的場景。** 在既定模板中，用事先固定的 task-ID 抽樣規則選未參與本系列分析的場景，每模板先規劃 25 題。先查曝光紀錄，再鎖定 IDs。這測的是新場景，不是新模板。
3. **D2：跨模板確認。** 預先指定至少 3 個切分，先檢查其模板是否重疊。每個切分使用只由該切分訓練模板訓練／校準的 base、probe、memory。若需要重訓 base／probe，列成獨立成本；不能拿已看過驗證模板的 base 來冒充整套系統的 cross。

已有 checkpoint 可以先做 D0 的機制分析；不足以支持未見模板的一般結論。D2 若沒有足夠互不重疊的模板，明確限制結論，而不是靠增加同模板場景或 action 數充當泛化樣本。

每個新模板先抽 25 題是起始採樣規格，不是已達統計檢定力的保證。依 D0／訓練資料估計場景與模板變異，先做精度／檢定力模擬，再鎖定 D1、D2 樣本數；不看 D2 treatment 結果後才決定要不要加樣本。

### 2.3 固定 query 面板與部署面板分開

- **機制面板：** 收集政策、support 集合先凍結，再把所有待比較條件會使用的同場景 support 聯集排除，得到共同 query 集。所有模型用完全相同的 query IDs。Query 不依 treatment 的好壞或真實成功率補樣。
- **閉環面板：** 使用完整候選池，政策只能讀取已執行 action 的結果；成功就停，最多 100 次嘗試、10 次 memory 更新、保留 4 次失敗。它才回答實際的解題收益。

如果為省成本再抽 query，使用 label-blind 的固定 hash 抽樣，所有條件共享。主要分析使用完整共同 query；抽樣版只能作先導檢查。全失敗／全成功的 query 集保留，AUC 記為未定義；不得以 0.5 代替，也不得因為不好算就換場景。

### 2.4 鎖定數值路徑

正式新比較統一使用 float64／log-space 的穩定接觸分數，保留精度直到排序。Legacy scorer 只在數值消融保留。固定 encoder、probe、幾何實作、batch、初始 latent 路徑與同分處理。

原 shortlist 與用穩定 scorer 重新產生的 shortlist，是兩個不同介入，必須分開。新收集前驗證 initial-only encoding 與評估輸入路徑；若無法一致，當作獨立因素紀錄，不能靜默混用。

## 3. 實驗 A：在同一批 cross query 上量完整的預測到排序路徑

**目的：先把目前分散在不同面板的線索，放到相同資料上。** 這一步不重訓。

使用 D0 的全部 100 題、固定 shortlist。主要 checkpoint 是預先指定的原始 R0；四個 A/B 微調模型作重複性檢查，不根據新結果挑一個贏家。主要 context 是共同 base 的第一個真實失敗。第一步已成功和候選池無解的題目仍列入 coverage／總體報告，與條件式排名分析分開。

K=1 是主要比較；K=4 是次要比較，只用確實累積四個失敗的共同歷史。不填入假失敗、不延續成功後的軌跡，也不把不同 K 的不同場景直接比較來宣稱更多 memory 有害。

### A1. 同一條 query 的四種輸入

| 條件 | 輸入 scorer 的來源 | 用途 |
|---|---|---|
| A1-GT-geometry | 真實模擬的物體幾何 | 檢查接觸規則與真正成功條件是否一致。 |
| A1-GT-latent | 真實影片 latent → 固定 probe | 檢查在真實 latent 上，讀出幾何損失多少資訊。 |
| A1-base | Base 預測 latent → 同一 probe | Base 基準。 |
| A1-memory | 同一 query＋固定 context 的 memory 預測 → 同一 probe | Memory 的成對效果。 |

所有條件使用相同候選、相同 scorer 和幾何單位。GT-geometry 同時檢查原 32 個評估時間點，以及較密的真實模擬時間序列，以辨別取樣是否漏掉關鍵接觸或持續時間。較密取樣時使用相同物理時間範圍和按時間間隔加權的分數，避免只因多了幀數就放大接觸總分；這一支以 GT 幾何檢查時間取樣，不偷偷改模型 rollout 長度。真正成功以 simulator 的任務判定為準；不把「有一次接觸」自行當成成功定義。

不能把成功標籤直接當分數、得到 AUC=1，就宣稱 scorer 上限已驗證。GT-geometry 仍要經過實際要研究的接觸評分規則。

### A2. 每個 query 都存哪些指標？

- **Latent：** future latent L1、L2；不計相同的初始時間點。整體、成功、失敗分開。
- **幾何：** 綠球／動態目標位置誤差、表面距離誤差、是否接觸、接觸時間與持續時間。比較時間格點固定；不以整條軌跡平均掩蓋接觸附近的錯誤。
- **排序：** ranking AUC、第一個成功排名、前 10 個候選是否有解、成功／失敗配對的順序翻轉。真實失敗但分數最高的候選另列出來。
- **分數：** 分數分布、同分率、成功／失敗間的 margin、score 變化；log score 不稱為校準的成功機率。
- **Coverage：** query 是否含解、context 是否可用、實際 K、每類別的場景數與候選數。

固定 query 的主要 ranking 指標是場景平均 AUC；Hit@10 作決策相關的次要指標。閉環主要指標另定為 AUCCESS。所有數值均保留逐場景與逐模板結果，不只看總平均。

### A3. 可得到的結論與限制

- GT-geometry 已排不好：接觸分數／時間取樣不完全對應任務成功；先處理這層。
- GT-geometry 好、GT-latent 差：固定 probe 在真實表示上的資訊讀出存在限制。
- GT-latent 好、predicted latent 差：預測和它的讀出組合有問題。**這還不能單獨分清 world model 與 predicted-latent probe mismatch。**
- Memory 的 L1 下降，但成功 query 接觸誤差和排名惡化：在同一批資料上建立目標與決策不一致的證據；仍不等於某個訓練 loss 是唯一原因。

## 4. 實驗 B：直接測「把 prediction 變準」是否讓 ranking 變好

**目的：不只做 L1 與 AUC 的相關圖，而是明確改變預測內容。** 以下均為 oracle 離線診斷，不能部署，也不算新的模型性能。

### B1. 逐步把 latent 拉向真實值

對同一 query 定義：

`z_alpha = (1-alpha) × z_pred + alpha × z_true`

alpha 固定為 `0、0.25、0.5、0.75、1`。Base 和 memory 各自做一次，初始 latent 保持原值，其他一切不變。Future latent L1 應精確按 `(1-alpha)` 縮小；這也是實作檢查。

逐點量測幾何誤差、AUC 和 Hit@10：

- L1 必定下降、幾何卻不穩定：表徵距離和 probe 讀出的物理量不一致。
- 幾何改善、排序不改善：繼續檢查 scorer 重視的時間／物件，以及成功與失敗的相對 margin。
- 排序多數隨改善上升：支持這條特定修正路徑有用，不是證明任意降低 L1 都有效。

Latent 內插可能離開真實影片表示的分布。故需同時有 alpha=1 的真實端點，以及下列精確替換控制；不能把中途波動全部歸咎於 probe 本身。

### B2. 精確替換與幾何層的 oracle

1. 用固定 hash 隨機選一部分 query，把它們的完整預測 latent 換成真實 latent；替換比例同樣是上述五點，事先固定 5 組巢狀抽樣。這避免單條 latent 內插，但仍是人工混合的候選集合。
2. 在分數的輸入層，把預測的逐時表面距離逐步拉向真實距離。這直接、可控地降低 scorer 所見的距離誤差；不假裝中間值是一條物理可實現軌跡。
3. 分別只替換成功候選、只替換失敗候選，定位哪一類誤差在破壞排序。因為使用 query 標籤，這是類別條件的 oracle 分解，不能拿它挑 action。

單一「prediction quality」數字不足以界定介入。B1/B2 要報告哪種改變改善了哪種排序，而不是宣稱已證明「prediction → ranking」的一般定理。

## 5. 實驗 C：固定 query，改 support，測修正的適用範圍

**目的：判斷 memory 是否根據合適的經驗做出合適的修正。** 不改 checkpoint、query 或 scorer。

### C1. Support 對照表

| 條件 | 改變什麼 | 能歸因什麼 |
|---|---|---|
| C0 空 context | K=0、empty gate | Base 身份檢查，不當成與 K>0 的資訊量匹配對照。 |
| C1 同場景、base 政策失敗 | 主要 context | 正確場景經驗下的修正效果。 |
| C2 同場景、隨機政策失敗 | 只換 context 來源，匹配實際 K | 收集政策造成的經驗組合差異。 |
| C3 同模板、其他場景 | 換完整 support tuple，匹配 K 與失敗結果 | 同模板但不同場景的訊息能否取代正確經驗。 |
| C4 不同模板、其他場景 | 同上 | 更強的場景替代控制，與 C3 分開。 |

每個受測場景的 C3/C4 各預先固定 5 個 donor（若合格 donor 不足就報實際數），不可事後挑最差 donor。整份 action、軌跡、residual、結果一起替換，保持 support 內部一致。抽 donor 不使用 recipient 的 query 成功標籤或修正效果。

C1/C2 必須來自真實的「成功即停」政策 trace；只在兩邊都有相同實際 K 的場景做成對比較，並報告排除了哪些場景。這是條件式機制比較，不冒充所有部署場景的平均效果。

### C2. Support–query 關係怎麼描述？

先分清兩種分析：

- **可部署關係：** 初始幾何、action 參數、base 預測的碰撞對象／路徑，搭配已觀察 support 的軌跡。所有特徵在 query 執行前可取得。
- **Oracle 關係：** Query 真實碰撞對象、路徑與所需修正方向。只用於事後解釋，不用於選 support、訓練驗證集上的 gate 或形成部署成績。

不只用 action 座標距離。主要檢查「碰撞對象相同／不同、繞障礙物的路徑相同／不同、所需修正方向一致／相反」。特徵定義、時間對齊與分組規則在訓練／D0 上鎖定，D1/D2 不重新找有利分組。

在 K=1，以 query q 的預測幾何向量表示：

`需要的修正 u(q) = 真實幾何(q) − base 幾何(q)`

`實際修正 delta(q,s) = memory 幾何(q | support s) − base 幾何(q)`

分別計算距離、位置的修正方向與幅度，不把不同單位任意混成一個向量。報告誤差是否實際下降；方向相同不代表幅度合理。近零所需修正時，cosine 記未定義，另報 memory 是否破壞原本準確的預測。

### C3. 如何區分「修正錯了」和「修正太強」？

對相同 context 用現有 adapter scale `0、0.25、0.5、1` 做劑量比較。這是 hidden-state 介入，後續網路非線性，所以不能把 scale=0.5 說成最終 latent 修正剛好一半。

如果縮小修正有幫助，只能說完整介入過強或不適用；需要搭配 u 與 delta 的方向、幅度，才可進一步描述局部機制。最佳 scale 只在訓練／校準資料選，不能逐題用 query 真實結果挑選後宣稱可部署。

Residual 歸零、打亂 residual/action 對應可作壓力測試，但會製造訓練時沒見過的輸入。這些控制不單獨證明模型「不需要 residual」；要做方法層級的結論，需見實驗 E 的重新訓練控制。

**C 的主要對比：** C1 與 C2 的成對 AUC／Hit@10，以及 C1 與 C3/C4 的正確經驗優勢。真實物理關係分組中的異質效果是解釋性結果；沒有隨機操弄真實碰撞關係，不能把組間差異稱為「相似性本身的因果效果」。

## 6. 實驗 D：把固定歷史效果與閉環回饋分開

**目的：知道 memory 的一次預測問題，是否會透過後續選擇被放大。**

### D1. 每個歷史都交給所有模型評分

先由凍結的 base、原始 R0、隨機政策各收集真實 trace。對每個尚未成功的 prefix，讓 base／memory 在相同未嘗試 query 上評分；只允許讀取該 prefix 已觀察到的經驗。

組成「歷史來源 × 評分模型」矩陣。這測的是在共同歷史下，哪個模型排序較好。**模型在別人的歷史上選了一個好 action，不等於它自己的閉環就能達到那個狀態。** 各種歷史的結果分開，不當成互相獨立的自然樣本。

### D2. 在同一個失敗狀態分叉

以 base 的第一個失敗後狀態為主要分叉點；後期 prefix 為次要。從完全相同的已嘗試 action、context、剩餘預算開始，分別接續 base 和 memory，讓它們自行選擇後續 action。保存第一次分歧、各次成功候選排名與最終結果。

這是控制相同起點後，替換後續政策的總效果；它包含選擇與之後累積的經驗，不能再聲稱已單獨隔離其中一項。

### D3. 相同選擇預算下，操弄 memory 更新

所有 memory 分支都在相同第一個失敗後開始，設三種後續模式：

- **凍結：** 保留第一次 context，不再吸收新經驗。
- **正常更新：** 每次失敗後按原協定更新。
- **同場景隨機保留：** 只從該分支真正已觀察的失敗中選相同 K，與正常保留最近 K 比較。

不得把尚未執行、其他分支知道的 action 結果偷偷提供給目前分支。這些模式會走不同路徑，比較的是各種更新規則的總效果，不能用相減方式宣稱解出了「純 error amplification 比例」。

### D4. 候選池的獨立實驗

數值修正可能也影響前面的 shortlist。對相同原始有效 actions，做 `候選池來源（舊／穩定 scorer top1024）× 池內 scorer（舊／穩定）` 的 2×2，先用 base 與預先指定的 R0。

池內 scorer 的對比固定候選池；候選池對比允許成員改變，報告解存在率、重疊率與解題率。真正成功標籤只作事後 coverage 稽核，不用來建池。這一步與 memory checkpoint 介入分開；不能同時重選池、換 probe、換 memory 再只報一個提升。

## 7. 實驗 E：用訓練介入辨別資料與目標問題

只有 A–D 的量測和控制路徑通過後，才開始這一階段。目的不是再比較五個混合配方，而是讓每一軸的改變可被解釋。

### E1. 先做 matched 2×2

| 因素 | 水準 1 | 水準 2 |
|---|---|---|
| Support 來源 | 隨機政策的真實失敗前綴 | 凍結 base 政策的真實失敗前綴 |
| Loss | 普通 latent L1 | 成功／失敗分別平均後等權的 latent L1 |

共 4 組配方，每組 **5 個獨立 memory 初始化 seeds**，同一 seed 在四組間成對。共 20 次 memory 訓練／每個確認用 base 切分。這五個 seeds 仍以同一個凍結 base／probe 為條件，不涵蓋 base 訓練隨機性；若要主張整套訓練流程的穩健性，需另外重複 base／probe 初始化。先在單一開發切分完成，不自動展開全部 D2。

保持一致：base／probe、architecture、query IDs 與抽樣順序、actual K、場景排程、empty 比例、optimizer、步數、有效 batch、資料量。成功 query 不額外補進某一組；balanced loss 只改權重。單類別 episode 使用現有類別平均，單獨報其比例。

只使用兩種來源都有真實 prefix 的共同場景，匹配實際 K。加入相同的預先固定 empty episodes。重新訓練 memory，不共用一個已偏向某配方的 R0 初始化；若另做 warm-start，視為獨立因素，不混稱獨立 seeds。

具體初始協定沿用可重現的 R0：residual memory、16 個時間位置、10×10 空間池化、per-attempt encoder、empty gate；每 step 同場景／同 context 下 8 個 query，AdamW lr=3e-4、weight decay=0.05、OneCycle 3% warmup、gradient clip=1、bf16 forward。每個 seed 的空 context 比例固定 10%；其餘依預先凍結、共同可用的 K=1/2/4 排程抽樣，不把不足的 prefix 補成指定 K。若某 K 沒有足夠共同場景，在註冊中取消該層並縮小結論，不靜默更換比例。

先固定 20K steps 作同預算比較，保留訓練曲線與 training/validation 指標。若仍未收斂，只能得出「此預算下的配方效果」；加長預算須事先作另一個共同實驗，不只替落後組延長。效果可以包含最佳化難度，不能全部稱為資料的物理資訊量差異。

完整 2×2 可估計 support 來源效果、class weighting 效果，以及兩者是否互相依賴；不強迫用主效果忽略交互作用。

### E2. 再單獨檢查 decision-aware loss

在 E1 的普通 L1 下，兩種 support 來源各新增一組 `L1 + ranking loss`，仍 5 seeds，共新增 10 次訓練。Query 抽樣與其他因素完全不變，不同時改成功樣本比例。

Ranking 使用穩定分數與成功／失敗配對。溫度與權重只在 training/calibration 調整，使用共同小網格與相同預算；先驗證梯度不是數值歸零、且總 loss 沒有被單項尺度支配。所有方法享有相同模型選擇預算，最後配置在 D2 前鎖定。

若 A 階段發現 scorer 在真實幾何上已不可靠，E2 暫緩；先修好評分代理，再研究它能否當訓練目標。不能把一個壞代理的負結果當成所有 decision-aware training 都無效。

### E3. 證據使用與 residual 的必要性（條件式追加）

若 C 顯示正確 context 與 donor 差異很小，追加一組訓練控制：同一配方、seed、K、query 和預算，只把訓練 context 換成事先隨機指定的完整他場景失敗經驗。評估時對兩個模型都給相同的真實 context。它測試正確 support–query 配對在此訓練配方的價值；不把所有微小差異都解釋成模型完全忽略 memory。

若要聲稱 residual channel 有害／無用，再做成對重新訓練：兩組使用同一結構和輸入維度，只有 residual 通道在訓練與評估期間始終真實／始終為零；其餘真實 video/action/outcome 相同。這個比較的結論是該通道在該配方的增量價值。單次推論歸零不代替此控制。

每個追加控制使用 5 個成對 memory seeds。只在預先寫明的觸發條件下啟動，不把所有可能組合一次排滿。

## 8. 實驗 F：Probe 的 predicted-latent 適配

若 A/B 顯示 latent 改善沒有被讀成幾何改善，或 oracle latent 接近真值時讀出不穩定，進行此分支。Memory／base 的輸出全部凍結。

在 training-only 場景上，收集平衡的 base／memory 預測 latent 及真實幾何目標。訓練與原 probe 相同架構的適配 probe，另給原 training 分布的 probe 對照組相同更新預算。調參只用獨立校準場景，不使用 D1/D2 的 query 未來。

評估矩陣為 `latent 來源（base／memory／true）× probe（原本／predicted-latent 適配）`，固定 scorer。Probe 有獨立訓練 seeds，報告效果對它的敏感度。

若同一批 memory latent 經新 probe 後，在新場景同時改善真實幾何誤差和 ranking，可歸因為替換 readout 能救回一部分效能；**不代表 memory latent 本身已經物理正確**。若只改善 ranking、幾何不改善，可能是新讀出直接學了 task shortcut，須另外報告。

不存在一個可直接取得的「predicted latent 所代表的真實物理軌跡」作 probe oracle；真實執行的軌跡是 action 的結果，未必是該 latent 的語義。此分支提供可操作的讀出對照，不宣稱將 world model error 與 probe error 完美且唯一地分解。

## 9. 統計、主要比較與停止規則

### 9.1 事先鎖定什麼

執行前保存機器可讀的 registration：task IDs、模板切分、曝光紀錄、action IDs／順序、query/support 分配、donor IDs、checkpoints、scorer 版本、全部 seeds、主要比較、指標與分析程式 hash。

主要比較每階段控制在少數幾項：

- A：同 query/context 的 memory−base，AUC 與成功 query 的幾何誤差。
- B：預先指定的 oracle 修正曲線；不逐題挑最佳 alpha。
- C：C1−C2、C1−C3、C1−C4，actual K 固定。
- D：同起點分叉後的 memory−base AUCCESS；其他歷史與更新規則列次要。
- E：2×2 的來源、loss 與交互作用；E2/E3 另列。
- F：同 latent 換 probe 的成對效果，以及幾何／ranking 是否同時改善。

同一階段若進行多個正式檢定，使用預先指定的 Holm 校正；不要只摘出一個未校正的正結果。探索性關係圖和 subgroup 全部標示為探索性。

### 9.2 不確定性怎麼算

報告每個 template、每個 memory seed 的成對差異，並按模板／場景的群聚結構做成對重抽樣。Training seed 也是變異來源；不能只在同一模型裡重抽幾千個 action 就宣稱誤差很小。重複使用的 donor 一起重抽或單獨報 donor 敏感度。

目前只有 4 個 cross 開發模板，任何區間主要反映這些模板下的不確定性，不足以保證新模板的涵蓋率。不同 fold 若共享模板、base 訓練資料或 checkpoint，也不當成完全獨立。報告資料限制，不用複雜 bootstrap 包裝不足的獨立樣本數。

「沒有顯著差異」不等於兩個方法相同。若要談沒有實用效果，需要事先訂可接受範圍，例如 AUC ±0.02、AUCCESS ±0.01，並檢查區間是否足夠窄。這兩個數字只是本設計建議的決策門檻，不是已知科學常數；應在新結果出現前鎖定。

### 9.3 進入下一階段的條件

- 輸入對齊、legacy 重現、empty identity、scorer 精度或 action trace 稽核失敗：先修實作，不解讀模型機制。
- A 的資料與指標對齊通過：進入 B/C，不必等到某個方法顯著贏才分析機制。
- Scorer／probe 的基礎對照不可靠：先走 F 或修正代理，暫緩 E2 的 ranking 訓練。
- 訓練介入的變因未匹配，或可用場景太少：報告無法辨別，不用補挑成功題來救比較。
- D2 解盲後，不再用同一批結果選超參數；若更改方法，後續結果改稱開發結果，另保留新的確認資料。

## 10. 執行順序、成本與產出

| 階段 | 工作 | 是否重訓 | 產出 |
|---|---|---|---|
| 0 | 曝光／資料分割、數值與欄位稽核、鎖定 registration | 否 | 可比較資料清單與排除原因。 |
| 1 | A：同一批 cross 的 latent→geometry→ranking 完整量測 | 否 | 確定失效在哪些環節。 |
| 2 | B/C：oracle 修正曲線、support 替換與 scale | 否 | 預測改善是否能救 ranking；context 的適用範圍。 |
| 3 | D：歷史交叉評分、同起點分叉、更新規則 | 否 | 一次修正問題與後續政策效果。 |
| 4 | E1，視前面證據追加 E2/E3；需要時先做 F | 是 | 可歸因的資料／loss／readout 訓練比較。 |
| 5 | 鎖定配置，在 D1/D2 確認，再做完整閉環 | 視切分而定 | 新場景／新模板的效能與限制。 |

先用 R0 完成主要診斷矩陣，四個 A/B 模型重複主要對比；不把所有 donor、scale、oracle alpha、checkpoint 無差別做笛卡兒乘積。Base prediction、true latent、真實幾何和相同 context 的 memory 編碼可重用。

實測舊閉環中，一組 memory 的 100 題評估約需 3 GPU 小時，只作資源規劃參考。新矩陣會增加 probe 和模擬需求；先用預先指定的 8 個場景量吞吐與記憶體，再估總成本。這 8 題只作工程 smoke，不用來挑方法。每階段有獨立完成紀錄與可恢復輸出。

所有結果至少保存：初始 action IDs、實際支持歷史、latent／geometry／score／rank、各指標的分母、模型與資料 hash。大的 latent arrays 可對預先固定的 query 子集保存，完整候選保留可重算所需索引、幾何與分數；不能只保存事後挑出的失敗例。

最先執行的具體工作應是 **階段 0＋A**：補上同一批 cross query 的完整 latent、接觸幾何和排序紀錄。這能檢驗目前關鍵缺口——「這批 cross 裡，是否真的也是 L1 下降、但決策重要部分變差」。在這件事確認前，不直接把另一組 12 場景的 L1 結果套到 cross。

## 11. 哪些結果會改變我們的判斷？

| 結果組合 | 可以支持的判斷 | 仍不能說什麼 |
|---|---|---|
| 同 query 的 L1 下降，成功接觸誤差上升；oracle 接觸修正救回排序 | 整體 latent loss 沒捕捉到關鍵決策品質。 | 單靠此結果不能說 class balance 就能修好。 |
| 固定 query，換 support 後修正方向與排名有可重複差異 | 某些經驗的修正適用範圍不同，context 選擇有影響。 | 真實碰撞相似性的因果效果仍需另外控制。 |
| 同 latent，只換適配 probe 就改善新場景幾何與排序 | Readout 是可修復的限制之一。 | 不代表原預測 latent 或物理軌跡已完全正確。 |
| 同共同歷史 memory 有益，自行閉環卻差；分叉實驗確認差異 | 後續選擇與經驗累積是重要限制。 | 不可把幾個總分相減當成精確的中介效應。 |
| E1 固定 K/query 後，只有 support 來源介入有效 | 支持該收集政策的資料在該訓練配方有價值。 | 不代表所有 deployment mismatch 都已消除。 |
| E1 的 balanced loss 有效，來源因素無明確效果 | 支持類別權重是此配方的可改善因素。 | 不代表所有 latent loss 都不適合。 |
| D0 有效果，D1/D2 不穩定 | 效果可能依賴既有場景／模板；泛化證據不足。 | 不能把失敗一律歸因於 capacity 或 residual。 |

目前先設計並凍結以上比較；本文件本身不代表實驗已執行，也不代表所有條件式訓練分支都應一次啟動。
