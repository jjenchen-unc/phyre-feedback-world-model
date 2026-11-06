> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# PHYRE memory 實驗報告：目前結果、失敗線索與下一步

更新日期：2026-09-29。對象：未參與前面實驗討論的團隊成員。

本報告整理近期 cross-template memory 分析、R0–R4 重訓、部署資料 A/B 小規模實驗，以及已完成的 scorer 診斷。已完成的數字與尚待驗證的解釋分開呈現。文末提供詳細紀錄與原始資料入口。

## 先讀這段

我們想讓模型利用「剛剛嘗試失敗的經驗」，更準確地預測其他 action，進而更快找到解法。目前測過的 memory 模型，在這組 cross 開發資料上仍沒有超過不使用 memory 的 base。

目前有三個主要發現：

- **預測的平均誤差下降，不代表更容易選到成功 action。** 多組實驗都看到：失敗候選的平均預測誤差下降，但成功候選的誤差上升，排序也可能變差。
- **改用更接近部署時的失敗經驗訓練，這次尚未帶來穩定改善。** 最新 A/B 的兩個微調 seeds，一個變好、一個變差；驗證樣本很小，不能據此否定這個方向。
- **scorer 有一個可重現的數值弱點。** 很小的分數可能被算成完全相同的 0，丟掉候選間的差異。最新固定預測重算已確認：穩定計算會恢復部分排序資訊，使這個小型驗證集的 memory 平均 AUC 高於同樣修正後的 base；完整 cross 尚未重跑。

我們尚未確認 memory 失敗的單一主因，也沒有足夠證據支持直接重寫架構。下一步優先分清楚：是預測本身變差、讀取預測的 scorer 有問題，還是兩者都有。

## 1. 我們在解什麼問題？

PHYRE 是二維物理任務。這裡使用的 action 是放入一顆球的位置與大小，目標是讓場景中的物體滿足接觸條件。一次 action 對應一次完整模擬；如果失敗，再從相同初始場景嘗試另一個 action。

因此，本報告的「action sequence」是**一連串重新開始的嘗試**，不是在同一次模擬中連續控制物體。

目前系統的流程是：

```text
初始場景 + 候選 action
          ↓
Base world model：預測接下來的視覺特徵
          ↓
Probe：從預測特徵讀出物體位置等資訊
          ↓
Scorer：根據預測的接觸情況給候選打分
          ↓
選擇 action → 真實模擬 → 成功則停止
                         ↓ 若失敗
              把 action、結果與軌跡放進 memory
                         ↓
              重新預測、排序剩餘候選
```

Base world model 預測的是 **latent**，也就是影片經編碼器轉成的數值特徵，不是直接輸出物理座標。Probe 負責把這些特徵轉成可用來計算接觸分數的資訊。因此，world model、probe、scorer 三個環節都可能影響最後排序。

Memory 的目標，是利用已觀察的嘗試，修正其他候選的預測。後面提到的 memory 訓練與微調，**base world model 和 probe 都保持凍結**。

### Memory 實際看到什麼？

| 名稱 | 白話意思 |
|---|---|
| Support / context | 已經嘗試過、可以拿來參考的經驗。包含 action、成功／失敗結果、實際軌跡特徵。 |
| Query | 尚待預測或排序的候選 action。模型不能先看它的真實未來。 |
| Residual | 某次已觀察嘗試的「真實軌跡特徵 − base 預測特徵」。 |
| K | Memory 裡實際有幾次嘗試的經驗。 |
| Empty gate | 還沒有經驗時，關閉 memory 修正，讓輸出等於 base。 |

目前 memory 使用 residual 作為輸入的一部分，也保留實際軌跡特徵、action 與結果。訓練時可以用 query 的真實未來計算 loss，但不把它當作 query 預測的輸入。

**Residual 是某個 action 的預測錯誤，不等於整個場景共用的物理參數。** 例如，一個失敗 action 需要把預測往右修正，不代表另一個成功 action 也該往右修正。模型需要學到「這份經驗適用於哪些候選」。

現有架構已經根據 query 的場景與 action 讀取 memory。因此，「加上 query conditioning」並不是目前缺少、尚未實作的能力。

## 2. 怎麼讀後面的數字？

### 評估資料分成三種，不能混在一起看

| 評估範圍 | 測試什麼 | 能回答什麼、不能回答什麼 |
|---|---|---|
| 訓練場景診斷 | 在 memory 訓練用過的場景與既有候選上評估 | 可看模型在已見資料上有沒有利益；不能證明泛化。 |
| Memory 保留驗證集 | 新 memory 訓練沒用過的模板，但既有 base／probe 看過 | 可檢查 memory 在這些模板上的表現；不是整套系統都沒見過的 cross 測試。 |
| Cross 開發集 | 與原始訓練模板不同的任務；本次是 fold 1、4 個模板、100 題 | 可比較這批模型在跨模板任務的表現；這批資料已反覆用於分析，不是未使用過的最終 test。 |

「模板」是同一類任務的生成規則；同模板下的不同場景可能共享很多結構。因此，100 題不等於 100 種獨立的新物理機制。

### 三個主要指標

| 指標 | 白話解釋 | 方向與限制 |
|---|---|---|
| AUCCESS | 把不同嘗試次數下的解題成功率整合成一個分數，較早解出會得到較高分。 | 越高越好。用於完整嘗試流程，不等於下面的 AUC。 |
| Success@100 | 在最多 100 次嘗試內解出多少比例的題目。 | 越高越好。 |
| Ranking AUC | 隨機拿一個成功候選和一個失敗候選，成功候選被排在前面的比例；同分算一半。 | 越高越好；全部同分是 0.5。候選全成功或全失敗時無法計算。 |

另外會看 **latent L1**：預測特徵與真實特徵的平均絕對差，越低越好。但它平均了很多時間、位置與特徵，沒有特別保證「影響成功判斷的部分」更準確。

固定候選的 Ranking AUC，與重新選 action、更新 memory 的完整流程，是不同評估。下文不把它們互相替代，也不直接比較不同候選集合的 AUC 高低來宣稱進步。

## 3. 最早的問題：加入 memory 反而退步

在同一組 100 題 cross 開發資料上，每題有 1,024 個候選，最多嘗試 100 次，最多更新 10 次，memory 保留最近 4 次失敗。以下是歷史模型的結果：

| 方法 | AUCCESS ↑ | Success@100 ↑ |
|---|---:|---:|
| Base，不使用 memory | 0.4541 | 62% |
| 舊 memory，不提供 feedback | 0.3632 | 47% |
| 舊 memory，提供失敗 feedback | 0.3937 | 49% |
| 重新訓練、有 empty gate 的 memory | 0.4065 | 50% |
| 上一列同一模型，把修正強度降到 0.25 | 0.4502 | 60% |

這裡確認了兩件事。

第一，舊模型即使沒有經驗，也會透過學到的預設 memory token 改動 base。因此，「不提供 feedback」並不等於「不使用 memory」。後來加入 empty gate；新模型的空 memory 輸出已驗證與 base 相同。

第二，**修正空 memory 的行為後，收到真實失敗經驗仍然可能退步**。縮小修正可以追回大部分損失，但不能據此判斷是修正太大，還是修正方向本來就錯、縮小後只是比較接近 base。

舊 memory 相對 base 少解出的 13 題中，12 題集中在模板 `00014`。因此，問題並非平均分散在所有模板上。

另有歷史 TTA 結果達到 AUCCESS 0.5075、Success@100 67%。TTA 是在測試時用失敗經驗做梯度更新；它更新 base，且最多使用 10 次失敗，與凍結 base、保留 4 次經驗的 memory 並不公平對齊。它提示 feedback 可能有用，但不能用來直接判定哪種架構較好。

## 4. 兩個失敗案例：memory 把成功候選往不適合的方向修正

我們選了兩個已知退步的 `00014` 場景，固定同一份失敗經驗與同一批 query，檢查歷史 gated memory 的影響。這是針對問題案例的診斷，不代表整個資料集的失敗比例。

只加入第一次失敗後，排除這個已嘗試 action，剩下候選的結果如下：

| 場景 | Base 的第一個成功候選排名 | 加 memory 後排名 |
|---|---:|---:|
| `00014:273` | 2 | 350 |
| `00014:407` | 4 | 141 |

這兩題的整體 latent L1 下降，但成功候選的 latent 誤差、綠球位置誤差與排序都變差。

位置分析提供了一個更具體的線索：第一次失敗的 support 需要把 base 預測的終點向右修正；成功 query 的真實終點卻需要向左修正。加入 memory 後，成功 query 的預測仍平均向右移約 31／29 像素，離真實結果更遠。

這支持一個**局部解釋**：對已觀察 action 可能適合的修正，被用到了不適合的候選上。但這些位置來自 probe 讀出，且只測了兩個刻意挑選的舊模型案例；不能說已證明所有新模型都複製 support residual，或所有 query 都收到同一個偏移。

同兩題若把真實影片 latent 交給相同 probe／scorer，AUC 都是 1.0。這說明它在這兩題的真實影片上能讀出有用訊號；仍不能保證它能可靠處理模型預測的 latent。

## 5. 第一輪重訓 R0–R4：調整資料與目標，仍未超過 base

為了檢查失敗經驗不足、成功候選被平均 loss 忽略等可能性，我們訓練了五個新的 memory 模型。它們共用資料來源、凍結的 base／probe，以及 empty gate；每個模型訓練 20,000 steps，只有一個訓練 seed。

新 memory 使用 12 個模板、1,200 個訓練場景。另留 4 個模板給 memory 驗證。R0 是這輪新建立的參考模型，不是上一節的歷史 checkpoint。

| 模型 | 主要做法 | Cross AUCCESS ↑ | Success@100 ↑ |
|---|---|---:|---:|
| Base | 不使用 memory | 0.4541 | 62% |
| R0 | 成功／失敗混合 context，普通 latent L1 | 0.4420 | 61% |
| R1 | 改用失敗 context | 0.4277 | 58% |
| R2 | 調整 query 抽樣，並平衡成功／失敗的 loss 權重 | 0.4201 | 60% |
| R3 | 合併 R1 與 R2 | 0.4233 | 57% |
| R4 | R3 再加上成功候選應排在失敗候選前面的 loss | 0.3962 | 51% |

**觀察到的結果是：這五個模型都沒有超過 base，R1–R4 也沒有超過 R0。** 但實驗不能支持「失敗資料有害」或「ranking loss 沒用」這種一般結論，原因如下：

- R1 同時改變 context 來源與實際 K。平均 K 從 R0 的 2.10 降到 1.54，沒有單獨隔離失敗標籤的影響。
- R2 同時改了 query 抽樣和 loss 權重，無法判定各自的貢獻。
- R4 只測了一組 ranking 權重與設定。單 seed、固定訓練預算也不足以確立一般方法優劣。
- 結果很受模板 `00014` 影響；一些方法差異在排除該模板後會改變正負。這是敏感度提醒，不是刪除難題的理由。

事後在固定模板內重抽題目得到的差異區間，大多仍包含零；它又沒有涵蓋重新訓練或新模板的不確定性。因此，目前只能報告這次模型的實測排序。

在另一組 64 個 memory 保留驗證場景中，29 個有非空失敗 context。這 29 個場景的平均 AUC，base 是 0.6697，R0–R4 使用正確 context 時為 0.6397–0.6571。成功 query 的 latent L1 在 R0–R3 的 29/29 場景、R4 的 28/29 場景上升。

把 context 換成其他場景的控制組，與正確 context 的差異仍不明確；而且大多數替代場景來自同模板。這不足以證明 memory 忽略場景資訊。

## 6. Memory 至少在訓練資料上有用嗎？

我們接著從 memory 訓練場景選 60 題，檢查舊候選池上的表現。其中 22 題有可用的非空失敗 context。

| 同一批訓練場景的條件 | 平均 Ranking AUC ↑ |
|---|---:|
| Base | 0.7425 |
| R0，正確 context | 0.8151 |
| R0，其他場景的 context | 0.8086 |

R1–R4 使用正確 context 的 AUC 也高於 base，介於 0.8031–0.8255。

所以，「memory 在任何資料上都無法產生利益」太強；已見資料上確實有改善。但其他場景的 context 也能改善很多，且這些是訓練場景與既有候選，**不能把這個改善當成學會了可泛化的、場景專屬的物理修正**。

## 7. 最新 A/B：用 base＋scorer 收集更接近部署的資料

### 為什麼做這個實驗？

原先 memory 訓練常從小候選池抽 support。實際使用時，模型會在較大的候選集合裡挑它認為最有希望的 action；失敗後看到的，是「模型以為會成功、實際卻失敗」的經驗。它與隨機抽到的失敗不一定相同。

因此，我們要問的是：**保持其他條件一致，改用 base＋scorer 真正選出的失敗 context 訓練，是否比隨機政策選出的失敗 context 更好？**

這輪微調的是 memory，**沒有 post-train base world model 本身**。也還沒有用更新後的 memory 政策反覆重收資料。

### 怎麼收集、怎麼保持比較公平？

固定選 60 個訓練場景、16 個 memory 保留驗證場景。在每個場景，從前 5,000 個 cache actions 過濾無效 action，再用 base＋scorer 建立 1,024 個候選的 shortlist。

- **A 政策：** 在相同 shortlist 中以隨機順序嘗試。
- **B 政策：** 依 base＋scorer 的排序嘗試。
- 兩者都遇到第一次真實成功就停止，最多嘗試 10 次；不先查成功標籤來挑失敗。
- 只有兩種政策都實際累積了失敗的階段，才建立成對 context。兩邊實際 K 相同，最多保留最近 4 次失敗。
- 每個場景共用 32 個 query，且不與任何一邊的失敗 support 重疊。

Query 包含最多兩個已收集到的終止成功 action、部分高分候選和隨機候選。成功 action 在這裡是共同的監督目標，不是事先提供給政策的資訊。這個 query 組合不是純粹按部署政策抽樣，因此本輪主要測的是 **context 來源**。

總共收集 3,555 條真實模擬軌跡。51 個訓練場景與 12 個驗證場景有符合條件的成對失敗 context。訓練場景中，base 政策選出的 404 次失敗有 **98.76% 不在先前編碼過的資料集合裡**。這確認收到了不同的經驗，但不代表所有部署分布差異都已消除。

### 訓練與評估設定

A、B 都從同一個 R0 checkpoint 開始，各跑兩個微調 seeds、2,000 steps。兩邊共用 query 抽樣、實際 K 與訓練安排；一半抽樣機會使用新資料，另一半重播相同的舊 R0 資料。Loss 維持普通 latent L1，base／probe 不更新。

主要比較時，**A、B 都使用同一份 B 政策 context**。這樣才不會把「訓練方式不同」與「評估時拿到不同經驗」混在一起。先在每個場景內平均不同失敗階段，再平均場景，避免失敗階段較多的場景佔較大權重。

### 完成結果

| 模型 | 相同 base-policy context 下的 Ranking AUC ↑ |
|---|---:|
| Base，不用 memory | 0.8244 |
| 原始 R0，未微調 | 0.7850 |
| A，seed 0 | 0.7825 |
| B，seed 0 | 0.7773 |
| A，seed 1 | 0.7763 |
| B，seed 1 | 0.7815 |

B−A 在 seed 0 為 **−0.0052**，在 seed 1 為 **＋0.0052**。因此，本次沒有看到 B 的穩定優勢；四個微調模型也沒有超過原始 R0。

這張表有幾個必要限制：

- 12 個驗證場景中，只有 **7 個同時有成功與失敗 query，可以算 AUC**。另外 5 個全是失敗，仍保留在 latent 誤差統計中。
- 兩個 seeds 共用同一個預訓練 R0，只重複了微調過程，不是兩次完全獨立的訓練。
- 評估只涵蓋兩種政策都產生失敗 context 的場景，不代表所有部署場景。
- 這是固定候選評估，**不是新的 cross AUCCESS 或 Success@100 結果**。
- 收集時的「只編碼初始畫面」與歷史資料的「完整影片第一個 latent」存在數值差異。A、B 使用相同慣例，但不能宣稱完全重現了歷史部署路徑。

## 8. 最新發現：排序崩潰與 scorer 的數值問題

### 一個場景的全部候選被打成 0 分

在驗證場景 `00012:076`，加入正確 policy context 後，32 個候選的 scorer 分數全部是 0。這出現在原始 R0 和四個微調模型，而且十個失敗階段都一樣。

這個場景有一個成功 query。Base 的 AUC 是 0.871；memory 因為全部同分，AUC 變成 0.5。

這是本驗證集唯一出現整組候選全部同分的場景，但其他場景也有個別候選得到 0。不能只因失敗候選得零分就認定模型有錯；問題是同分是否抹掉了成功與失敗的區別。

這一題的影響很大：事後若把它拿掉，五個 memory 模型相對 base 的平均 AUC 差異都會由負轉正。正式結果仍保留它。這提醒我們，小樣本的整體平均可能被一個特殊失效模式主導。

另有一個容易誤讀的數字：全零時，成功 query 的「第一個成功排名」是 4，看起來比 base 的 5 好。但它本來就放在輸入清單第 4 個；同分排序只是沿用這個順序，**不是模型選得更準**。

### 為什麼可能被算成完全相同的 0？

目前 scorer 先估計每個時間點的物體表面距離，再把「距離越近、分數越高」的小分數相加。程式使用 `0.5 × (1 + tanh(x))`；當有限精度計算把 `tanh(x)` 捨成 −1，這一項就會直接變成 0。

使用 checkpoint 的參數做本機 float32 測試，在距離約 22 像素後就可重現歸零。距離 23 像素時，原式為 0，而數學等價的 sigmoid 寫法仍有約 `5.6e-9` 的分數。

上述合成測試證明計算方式會丟失小分數；現在實際模型距離也已重跑完成，結果見下一節。不同硬體／版本的確切歸零界線可能不同。Scorer 這段是 float32，不能直接把它歸因於 world model 使用 bf16。

**保住小分數，可能恢復排序，也可能只是更清楚地顯示錯誤排序。** 如果 memory 已經預測錯誤的接觸幾何，換一個數值穩定的公式並不會自動修好預測。

### 成功候選的預測問題，也不限於這一題

對最新驗證資料，分開計算成功與失敗 query 的 latent L1，得到：

| 條件 | 失敗 query 的平均 L1 ↓ | 成功 query 的平均 L1 ↓ |
|---|---:|---:|
| Base | 0.08750 | 0.07782 |
| 原始 R0，正確 policy context | 0.08559 | 0.08053 |
| 四個微調模型，正確 policy context | 0.08586–0.08608 | 0.08072–0.08203 |

兩欄都先在場景內平均，再平均場景；失敗欄有 12 個場景，成功欄有 7 個，因此兩欄的樣本範圍不同。

所有五個 memory 模型都在 **6/7 個有成功 query 的場景**增加成功候選的 L1。失敗候選則只有整體平均改善，不能說每個場景都改善。

這再次顯示平均預測 loss 與選出成功 action 之間有落差。不過，latent L1 不是幾何誤差；這個統計還不能單獨證明排序退步的原因。

## 9. 目前哪些解釋有證據，哪些還太早？

| 可能解釋 | 目前證據支持到哪裡？ |
|---|---|
| 空 memory 就改壞 base | 歷史模型確實存在；新模型已用 empty gate 排除，不能繼續拿它解釋所有新結果。 |
| Support 的修正被用在不適合的 query 上 | 兩個歷史案例有方向與幾何證據；新模型是否沿相同路徑失敗，仍待檢查。 |
| 訓練經驗與部署經驗不同 | 資料與選擇方式確實不同；最新 A/B 尚未證明換 context 來源足以改善。 |
| 普通 latent loss 沒保護成功候選 | 多組結果符合這個疑慮；尚未證明哪一種新 loss 可以解決，也不能否定所有 ranking loss。 |
| Scorer 丟掉細微排序資訊 | 公式與合成數值測試已確認這個弱點；已在同一份預測距離上確認排序資訊流失，穩定重算使小型驗證集的平均比較翻轉；完整 cross 效果仍未知。 |
| 模型完全不讀 memory，或架構一定不行 | 目前沒有足夠證據。錯場景控制並不乾淨，現有架構也已經根據 query 讀取 memory。 |

這些因素可以同時存在。例如，memory 先把預測距離推到過遠的區域，scorer 再把那些很小但不同的分數全部捨成零。現在的證據還不能把各自的影響拆開。

## 10. Scorer 診斷已完成：需要修正前面的解讀

這次沒有重訓模型，也沒有改變 query、context 或預測距離，只比較最後一步計算分數的方式。所有 1,400 筆重跑結果都精確重現原分數；再從保存的距離獨立重算穩定分數與場景平均 AUC，核對通過。

| 模型 | 原本 scorer 的 AUC | 穩定計算後的 AUC |
|---|---:|---:|
| Base | 0.8244 | 0.8391 |
| 原始 R0，正確 policy context | 0.7850 | 0.8541 |
| A，seed 0 | 0.7825 | 0.8676 |
| B，seed 0 | 0.7773 | 0.8624 |
| A，seed 1 | 0.7763 | 0.8440 |
| B，seed 1 | 0.7815 | 0.8474 |

這是同樣的 7 個 AUC 可計算場景，base 也做了相同數值修正。**原先「memory 在此驗證集的平均 AUC 低於 base」的排序，受 scorer 數值影響而翻轉。** 原始表格保留作紀錄，但不能再把它當成 memory 在這個候選集沒有排序價值的證據。

A/B 的資料來源比較仍沒有穩定優勢：B−A 為 −0.0052／＋0.0034。對原始 R0 的比較也變成 seed 0 微調較好、seed 1 較差。這些都是小樣本的觀察，不是一般方法排名。

在 `00012:076`，memory 的 AUC 從全零同分的 0.5 回到 0.9355，成功 query 排第 3；base 仍為 0.871、排第 5。**但物理預測沒有因此變準。** 原始 R0 在第一個失敗 context 後，把成功 query 的最近表面距離預測為 91.8 像素；base 為 5.2，真實為 2.6。這時 memory 的分數只是大約 `10^-79` 到 `10^-35` 之間仍有可用的相對差異。

跨 7 個有成功 query 的場景，成功候選的平均表面距離誤差也從 base 的 26.0 像素升到 R0 的 31.7，四個微調模型為 31.9–32.6。因此，數值問題與幾何預測問題可以同時存在；排序較好不代表模型已正確預測物理軌跡。

真實模擬幾何、真實 latent 經 probe 的 AUC，在這 7 題都是 1.0。這排除了此處普遍讀不懂真實軌跡的解釋，仍不能保證 probe 可靠讀取所有預測 latent。錯場景 context 的穩定 AUC 也仍高於正確 context，場景專屬經驗是否被有效利用仍未建立。

實作上，單純換 float32 sigmoid 還不完全足夠：原始 R0 得到 0.8407，使用 float64／log-space 才是 0.8541，需要保留精度直到排序。

**完整閉環評估已完成，結果見 §10.1**。原先設定為：base、原始 R0、四個 A/B 模型都成對比較原 scorer 與穩定 scorer，共 12 組；小規模檢查與主評估均已完成。使用獨立評估程式，保留歷史輸出。詳見 [本輪設定](STABLE_SCORER_CROSS_RUN.md)。既有候選收集、部分訓練目標也用過舊 scorer，不能把本次事後重算當成整條流程已修好；更不能直接宣布 cross AUCCESS 改善。

## 10.1 完整閉環結果（2026-09-30 已完成）

12/12 組與彙整核對均完成。目前沒有執行或排隊中的實驗。所有 1,200 筆逐題結果、逐次 action 選擇、context 更新與停止條件，已在遠端及本機通過核對；舊 scorer 的 base／R0 逐題成功次數重現歷史結果。13 個結果檔案 SHA256 一致。

| 模型 | 舊 scorer AUCCESS | 穩定 scorer AUCCESS | Success@100（兩者相同） |
|---|---:|---:|---:|
| base | 0.4541 | 0.4541 | 62% |
| initial_R0 | 0.4420 | 0.4419 | 61% |
| A_s0 | 0.4127 | 0.4119 | 53% |
| B_s0 | 0.4042 | 0.4062 | 52% |
| A_s1 | 0.4166 | 0.4173 | 55% |
| B_s1 | 0.4099 | 0.4110 | 55% |

穩定 scorer 沒有改變任何模型的已解題集合，也沒有改變 Success@100。各模型的首次成功次數只在 2–5/100 題改變；AUCCESS 最大增加約 0.0020。這裡的「改變題數」不是 action trace 有變的題數，不能據此說中間決策完全相同。

使用穩定 scorer 時，全部 memory 模型仍低於 base。A/B 微調都低於原始 R0；B 在兩個微調 seeds 都低於 A（差值約 −0.0057／−0.0062）。這是這批固定模型與重複使用的開發資料的結果，不是一般方法優劣或統計顯著性的判定。

需要區分兩個結論：數值缺陷確實存在，且曾影響固定 query 的 AUC；但只修正最後的 scorer，沒有解決本次 cross 解題退步。先前 AUC 診斷的 memory 保留模板為 00001/00004/00012/00022，這次 cross 是 00000/00002/00007/00014，候選數與閉環過程也不同；不能把兩者當成同一批題目上的矛盾結果。

保留原本小型固定 query 的結果，不以本次 cross 負結果抹去它。後續可用已保存的 traces 檢查分數改變是否影響前排候選，以及 memory 修正在哪次更新開始傷害成功 action；本次沒有再啟動訓練或其他新實驗。

## 10.2 第一次 memory 更新的失效位置（事後 trace 分析）

使用已保存的穩定 scorer 分數，檢查第一次失敗前後的排序。先確認各模型與 base 的初始分數完全相同，選到的第一個 action 也相同，再從兩邊比較的 query 中排除這個已嘗試失敗。因此，下面的變化發生在共同 context 下，尚未混入後續政策分歧。

100 題中，有 57 題第一次失敗且候選池內有解；另外的第一步成功題與無解候選池不進入此分析。

| 模型 | 成功候選排名改善 | 不變 | 變差 |
|---|---:|---:|---:|
| 原始 R0 | 22 | 8 | 27 |
| A seed 0 | 17 | 5 | 35 |
| B seed 0 | 17 | 5 | 35 |
| A seed 1 | 20 | 6 | 31 |
| B seed 1 | 20 | 5 | 32 |

例如 `00014:273`，排除第一個失敗後，base 的第一個成功候選排第 2；A seed0 更新後排第106，B seed0 排第141。`00014:407` 是第4→122／145。這些數字來自新微調模型與穩定 scorer，不是重用舊模型案例的排名。A_s0／B_s0 最後在這兩題都未能於100次內解出，而 base 原本能解。

目前可以定位的失效是：非空 context 引起的修正在第一次更新就會損害一些成功候選的相對排序。這些案例不需要先經過多輪錯誤累積才開始退步；但後續更新仍可能放大或修復，第一次排名下降本身不等於最後必定失敗。

它為何被學出仍未隔離。結合其他實驗，support 經驗對 query 的適用範圍學得不夠好，以及 latent loss 未保護決策重要部分，是值得優先查的解釋。此處保存的是分數，不能據此宣稱新模型把所有 query 往同一個物理方向修正、已證明 residual 複製，或資料不平衡就是唯一原因。

分析不新增訓練或推論。重現：`pipeline/analyze_first_memory_update.py`；完整逐題結果：`results/stable_scorer_cross_20260930/FIRST_UPDATE_ANALYSIS.json`。

## 11. 資料與詳細紀錄

本報告的表格來自已保存的結果；主要指標已有獨立重算。R0–R4 的 500 筆逐題 AUCCESS 已核對，最新 pilot 的 1,750 筆固定評估紀錄也已重算 AUC、排名與平均值，並檢查 A/B query、context 數量和 base／empty 一致性。這些檢查確認數字與比較設定一致，不代表因果解釋已成立。

| 想追查的內容 | 詳細文件／資料 |
|---|---|
| 歷史模型的 cross 結果 | [Cross failure analysis](CROSS_MEMORY_FAILURE_ANALYSIS.md)；表中 base 使用 `results/cx_clm_base.json`。 |
| 兩個案例的排名、位置與 residual 分析 | [固定 support 診斷](CROSS_MEMORY_DIAGNOSTIC.md)、[失敗原因重分析](MEMORY_FAILURE_REANALYSIS.md)。 |
| R0–R4 設定與結果 | [實驗設定](MEMORY_REDESIGN_RUN.md)、[結果](MEMORY_REDESIGN_RESULTS.md)、[解讀限制](MEMORY_REDESIGN_ANALYSIS_REVIEW.md)。 |
| 最新部署資料 A/B | [實驗設定](MEMORY_DEPLOY_PILOT_RUN.md)、[完整結果](MEMORY_DEPLOY_PILOT_RESULTS.md)、[原始 summary](../results/memory_deploy_pilot_20260929/SUMMARY.json)。 |
| Scorer 診斷與目前狀態 | [診斷紀錄](MEMORY_DEPLOY_SCORE_DIAGNOSIS.md)、[已完成的輸出稽核與合成測試](../results/memory_deploy_score_diagnostic_20260929/OFFLINE_AUDIT.json)。 |

共享專案位置：`oprime:/path/to/phyre-unc/`。本報告同步放在其 `docs/` 下。

Scorer 診斷 `33987` 與彙整 `33988` 已完成；輸出位置為 `results/memory_deploy_score_diagnostic_20260929/`，含 `ANALYSIS.json` 與獨立核對紀錄 `VERIFICATION.json`。後續穩定 scorer 閉環比較已提交：檢查 `34088`、主評估 `34089`、彙整 `34090`。本輪 12 組與核對已完成，結果見 §10.1。

## 12. 後續歸因實驗設計

完整方案見 [Memory 歸因實驗設計](MEMORY_CAUSAL_ATTRIBUTION_PLAN.md)。先在同一批 cross query 同時量 latent、幾何與排序，再用固定 query／context 的介入分清 readout、經驗適用範圍與閉環效果；最後才做匹配的 support 來源 × loss 訓練比較。方案已完成；階段 0/A 已完成，見 [執行設定](ATTRIBUTION_STAGE_A_RUN.md) 與 [完整結果](ATTRIBUTION_STAGE_A_RESULTS.md)。

100 個 cross 場景全部通過核對。主要非空 K1 比較中，R0 的整體 latent L1 由 0.115010 降至 0.114210（73 場景），但成功 action 的幾何誤差由 32.42 增至 41.23 pixel，AUC 由 0.5654 降至 0.5052（57 個含成功候選的場景）。四個 A/B 微調模型同樣呈現整體 L1 下降、成功幾何與排序變差。

需修正先前解釋：R0 成功 action 的平均 latent L1 也略降（0.131889→0.131602），所以不能只歸因於失敗 action 比例高或成功 action 的 L1 必然惡化。平均 latent 誤差與決策重要幾何之間的落差，需要獨立追查。真實 latent／真實幾何經相同讀出流程的 AUC 為 0.9769／0.9942，支持繼續檢查預測 latent 與 probe 的交界，但沒有排除 probe 對預測 latent 失準。這是重複使用的四模板開發資料上的量測，尚未完成 support 替換與 oracle 修正等因果介入，也沒有新的閉環成功率。


## 13. Within classifier 比較的公平性核對（2026-10-01）

400 場景 within fold0 的固定 initial-scene classifier AUCCESS0.5604，高於
base0.4789 與舊 memory+feedback0.5362。已重算這些方法的逐題 AUCCESS，
確認400個 task ID 一致；world-model各版本的 shortlist 解數亦一致。
這是共用512候選池的結果比較，不能直接稱為等資料、等資源的受控實驗。

「matched training data」準確指 classifier 的 s16r16 訓練配對對齊 base
predictor 的52,349條 rollout。Classifier 結果記錄25,527正例、26,822負例，
matched manifest 的檔案集合與 base pack 完全一致；歷史 classifier checkpoint
只存 state_dict，完整 launch args 未保存在該結果中，故此核對依據程式、結果
數量、既有實驗紀錄與 manifest，沒有重新訓練重現。

Memory另用 clusters pack：51,161列、50,982個唯一 rollout，其中47,474個
不在base的52,349條內；base+memory聯集99,823個。Probe使用另份6,382條的
sub400_div16 manifest，其中2,813個不在上述聯集內；該manifest同時供probe
fit/calibration分割使用，不能把全數說成實際fit資料。故整個memory系統與
classifier並非相同總訓練資料。

其他未控制差異：兩邊使用相同LeVJEPA模型revision，但classifier保留1024維
initial-scene tokens，world model使用PCA128的action-conditioned initial
frame；classifier學成功標籤，world model學完整latent軌跡，probe另外使用
模擬器位置/方向監督。架構、optimizer更新量與FLOPs未匹配。推論時memory
接收失敗rollout，0.5604的fixed classifier不接收feedback。

兩邊的512候選是由base world-model分數預篩，而不是中立的共同random池。
因此0.5604代表「WM預篩後由classifier重排」的表現，並非classifier從原始
候選池獨立解題的400場景結果。這仍是有效的共同候選重排比較，但不是完整
pipeline在獨立候選選擇下的公平性證明，也不能據此歸因為classifier只靠記憶版面。

額外修正：0.5459的matched-classifier online版本只在初始化時matched；
`closed_loop_dqn_online.py`的replay正負例由完整train status cache取樣，
沒有套matched manifest過濾。0.5604是fixed模式，不執行這段更新，故不受
這個額外replay來源影響；online版不能稱為全程同資料預算。

核對來源：`pipeline/{scene_classifier,closed_loop_dqn_online,closed_loop_memory,
make_shortlist,probe_rule}.py`；遠端base/memory/probe checkpoint metadata、
within訓練pack與manifest；`results/w400_*.json`。本次只有讀取與文件修正，
沒有新的訓練或評估。下一個受控比較需明確匹配總資料聯集、representation、
候選來源與feedback條件，或把差異清楚標成各方法本身的資源。


## 14. Probe 適配診斷已啟動（2026-10-01）

為區分memory修正問題與讀出不匹配，固定base/R0，只適配probe；比較真實latent
續訓、base/R0預測latent各半，以及initial-only控制，每組3個抽樣seed。331個
training場景與29個calibration場景都不包含cross的4個模板。重用階段A相同
100個cross場景/K1/query，先檢查原probe精確重現，再比較幾何與ranking。
本輪不重訓memory、不執行新閉環。工作已提交，尚無新結果；詳細預先設定、
資料限制與job ID見 [PROBE_ADAPTATION_RUN.md](PROBE_ADAPTATION_RUN.md)。

## 15. 已授權接續：latent L1 與幾何監督（2026-10-01）

Probe 適配已完成九組訓練，評估仍未全部結束。使用者授權其後繼續驗證
training objective，工作已透過 afterok 接續排程。新比較從同一 R0 出發，
只改「latent L1」與「latent L1＋0.1×幾何 loss」，各三個 matched seeds，
同 support/query／2,000 步，base 與原 probe 不更新。評估會同時看原 probe
與三個適配 heads，避免只用訓練提供梯度的 head 判定有效。

目前 CPU 前置检查通過；GPU smoke、正式訓練／評估尚待上游完成，沒有新
科學結果或 AUCCESS。若幾何與排序的改善能跨 readout、跨 seed 重現，接續
設計 task-centric latent；這一支已授權但尚未實作／排隊。
完整設定與限制：[MEMORY_GEOMETRY_FOLLOWUP_RUN.md](MEMORY_GEOMETRY_FOLLOWUP_RUN.md)。

## 16. Probe 與幾何監督已完成（2026-10-01 最新結果）

兩輪皆完成，當前沒有執行／排隊工作；第 14、15 節的執行中狀態已被此節
取代。完整解釋、模板差異、核對範圍見
[PROBE_GEOMETRY_RESULTS_20261001.md](PROBE_GEOMETRY_RESULTS_20261001.md)。

相同 57 個 cross 可算 AUC 場景，原 probe／適配 probe 的 AUC：base
0.5654／0.7101，原 R0 0.5052／0.6288，純 L1 續訓 0.5143／0.6413，
L1＋幾何監督 0.5560／0.6767。後兩列平均三個 memory seeds；適配 head
使用三個 seeds 全部結果，沒有挑最好者。不能把 3×3 組合當九次獨立完整訓練。

Readout 可改善，但 memory 與同樣適配的 base 的 AUC 差距反而擴大。
幾何監督相對 matched L1，在三 seeds／各 readout 的整體幾何與 ranking
都改善，latent L1 卻略變差；memory 仍沒有贏 base。00007 在原 probe
下仍出現幾何改善而 AUC 退步，不能說所有模板都改善。尚無新閉環 AUCCESS。

事前 task-centric latent 接續門檻通過，該分支已授權但尚未實作／排隊。
目前可支持訓練目標與 readout 都是可改善環節，不能宣稱唯一失敗根因已找到。

## 17. 閉環收益驗證已啟動（2026-10-01）

使用者授權先驗證離線改善是否增加解題收益，再接續 task-centric latent。
目前 22 組完整對照：原 probe base、三個適配 probe base、兩種 memory
各自全部 3×3 memory/probe seeds。固定 100 題、每題 1,024 候選與 100 次
嘗試預算，最多十次失敗 memory 更新、keep4；使用已保存 simulator 結果
做閉環重播，不新增候選／rollout 或訓練。

GPU smoke 34280 →核對 34281 →正式評估 34282 →彙整 34283 已提交，
上游成功才執行下游，最多兩個 GPU 工作。新的 AUCCESS 結果尚未完成。
設定：[ADAPTED_PROBE_CLOSED_LOOP_RUN.md](ADAPTED_PROBE_CLOSED_LOOP_RUN.md)。

第 17 節進度更新：smoke 與核對已完成（20 組設定×場景、1,124 次選擇），
完整 22 組評估 34282 已開始。Representation 最小實驗僅有準備稿，尚未啟動。

## 18. 閉環中途結果（2026-10-02，尚未全數完成）

22 組已完成六組並通過核對，另兩組執行、十四組排隊。Base 的三個適配
probe 已全部完成：平均 AUCCESS 0.4541，與原 probe 0.4541 幾乎相同；
Success@10 47%→46%，Success@100 62%→68%。新增成功多發生在第十次
之後，不能把 offline AUC 的提升直接說成更快解題。

僅兩個純 L1 memory 組合完成，目前均略低於同 probe 的 base；其餘 memory
仍在執行／排隊，幾何監督尚無完整閉環結果，不能下整體 memory 結論。
完整數字與核對範圍：
[ADAPTED_PROBE_CLOSED_LOOP_PROGRESS_20261002.md](ADAPTED_PROBE_CLOSED_LOOP_PROGRESS_20261002.md)。

## 19. 22 組閉環已完成（2026-10-02 最終結果）

全部評估與核對完成，目前沒有本輪執行／排隊工作。第 17、18 節的中途
狀態由此取代。完整結果見
[ADAPTED_PROBE_CLOSED_LOOP_RESULTS_20261002.md](ADAPTED_PROBE_CLOSED_LOOP_RESULTS_20261002.md)。

| 方法，全部 seeds 平均 | AUCCESS | Success@10 | Success@100 |
|---|---:|---:|---:|
| Base＋原 probe | 0.4541 | 47.0% | 62.0% |
| Base＋適配 probe | 0.4541 | 46.0% | 68.0% |
| 純 L1 memory＋適配 probe | 0.4527 | 47.8% | 62.9% |
| 幾何監督 memory＋適配 probe | 0.4622 | 47.0% | 63.8% |

幾何 memory 對同 probe base 的九個配對 AUCCESS 全為正，平均 +0.00817，
對匹配純 L1 也全為正。但 Success@100 九個配對都低於 base；第一次更新
平均 AUC 仍全為負，損害約從純 L1 的 -0.0567 減至 -0.0274。模板 00007
仍退步。因此支持小幅 AUCCESS 收益，不能宣稱 memory 已全面勝出，或
support–query 修正已穩定有用。所有結果仍是重用的四模板 D0 開發資料。

遠端與本機各重播 98,441 次選擇，本機重算 2,200 個逐題 AUCCESS，4,423
個下載檔案 hashes 一致。Task-centric latent 最小實驗已授權、尚未啟動。

## 20. 為何仍丟失 base 可解題：trace 歸因（2026-10-02）

詳見 [MEMORY_CLOSED_LOOP_LOSS_ANALYSIS_20261002.md](MEMORY_CLOSED_LOOP_LOSS_ANALYSIS_20261002.md)。
幾何 memory 平均丟掉 base 可解的 5.11 題、增加 0.89 題，淨損失 4.22 題。
九組共 46 筆丟失紀錄，涉及 11 個不同題目；44/46 在第一次更新後就把所有
成功候選排到剩餘預算外。不是主要等到 K4／多輪歷史才開始出錯。

用保存分數做事後、可精確重播的控制：沿目前策略前十次，若全失敗則第
十一次回到 base，平均 AUCCESS／S@100 為 0.4687／68.0%，優於持續使用
memory 的 0.4622／63.8%。只固定第一次 memory 排序則為 0.4614／64.3%。
此結果支持追查「修正適用性」與「持續信任錯誤排序」，不證明唯一訓練
根因。全部 cutoff（1／2／4／10）都保留；仍是同一 D0 上的事後分析，
不是新的確認性成績，也没有新增模型訓練／推論。
