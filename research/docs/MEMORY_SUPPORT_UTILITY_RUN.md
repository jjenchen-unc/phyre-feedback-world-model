> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Support 抽樣對照的執行設定

2026-10-04。使用者已授權執行。完整研究問題見[實驗設計](MEMORY_SUPPORT_UTILITY_EXPERIMENT_PLAN.md)。本輪先執行 K1 matched-training pilot，使用既有 cross D0；新場景 D1、K4 與閉環是後续工作，未在這一輪假裝完成。

## 2026-10-04 解讀範圍更正

使用者質疑 5 px 門檻後重新核對凍結程式：N 只是「真實失敗且取樣未來最小表面距離 ≤5 px」的操作定義，尚未驗證它代表差一點成功，更未驗證它代表 support–query 修正適用性。因此本輪不能用來否定「接近成功 action 的經驗可能有用」的假設。

三組模型雖有相同場景/query/初始化/預算，卻全部在同時具 N/H/F 的 171 題子集訓練。Uniform 也受 5／20 px 與高分分組造成的場景篩選影響，不是完整 1,200 題的 Uniform 基準。自然失敗評估本身不以 5 px 挑 support，但被評估的模型仍來自這個條件子集。Within 沿用同一批模型，也不能消除此限制。

本輪保留為「此門檻、此子集、此 loss 與 R0 起點下，三種抽樣配方的表現」之探索紀錄；計算與配對核對不等於 N 的物理意義得到驗證。原計畫有門檻敏感度分析，但本轮僅做 coverage 盤點，尚無跨門檻重新訓練／效果證據。不得把 coverage 敏感度當成效果穩健性。

較早的 probe adaptation、L1 對 L1＋幾何監督對照，沒有用此 5 px N 定義篩訓練資料；幾何 loss 為物件位置與方向誤差。原 scorer 的 soft 參數為 [4,1]，屬另一個固定讀出假設，不能與本輪 N 的 5 px 篩選混為一談。這些較早結果仍受各自資料和 readout 範圍限制。

## 本輪要回答什麼

相同模型、query、loss 和訓練預算下，改為只提供 near-success 失敗 support，或混合三種失敗 support，能否改善成功／失敗排序？舊模型是否善用 near-success 不作訓練門檻。

## 已完成的資料盤點

先按既有 cluster 整群切分；無 cluster 的 action 使用寬度 0.02 的正規化 action 網格。探索／support／query 的目標比例為 20／40／40，群以固定 hash 分配，實際比例因群大小不同。相同 task/action 額外去重。

1,200 個訓練場景中，322 個同時有 N 與遠端失敗；用固定 base＋adapted probe seed 0 評分後，171 個同時有 N/H/F。三組一律在這 171 個場景訓練，不能解釋為全部 1,200 場景上的比較。

| 項目 | 數量 |
|---|---:|
| 共同訓練場景 | 171，涵蓋原 12 個模板中的 10 個 |
| 共同 query | 3,777，其中 1,612 個真實成功 |
| N support | 525 |
| H support | 367 |
| F support | 1,758 |
| 全部失敗 support，含中間距離 | 2,889 |
| 有三類 support 的 calibration 場景 | 64 |
| 實際 calibration 評估場景 | 49，各模板最多 16 題，固定 hash 選取 |

每個共同訓練場景都包含成功 query，但每一步仍可能沒有抽到成功 query。實際訓練使用均勻 query 抽樣，不另外成功平衡。525/367/1758 是可用 support 數，不是各 arm 的預期抽樣次數；場景均勻抽樣會改變全域權重。

N 定義為真實失敗且未來取樣最小表面距離 ≤5 px。H 為最小距離 ≥20 px，且 base 分數位於完整 support 區前 20%；F 是不在前 20% 的其他 ≥20 px 失敗。高分名額包含成功 action，以穩定排序取 ceil(0.2×候選數)，同分保留 ID 順序。這是稀疏幾何代理，不等於完整的物理成功條件。

在全部 1,200 訓練場景的 support N 中，428/2,758 初始已在 5 px 內；這個數字的分母不是最後 171 場景。保留逐場景原始值，結果分析需檢查初始接近是否主導效應。

## 三組訓練

| Arm | 每一步的單條失敗 support |
|---|---|
| Uniform | 在該場景全部失敗 support 中均勻抽樣 |
| Near | 只在 N 中均勻抽樣 |
| Mixed | N/H/F 各 1/3 機率，再在該類內均勻抽樣 |

每組 seeds 0/1/2，共 9 組；相同原始 R0 起點，固定 base、原 probe。每組 2,000 steps、每步最多 8 query、AdamW lr 3e-5、weight decay .05、cosine、clip 1。Loss 固定為未來 31 幀 latent L1＋0.1×既有位置／方向 loss，每 query 使用固定 8 個幾何監督時點。只評估最後一步，不挑 cross 最佳 checkpoint。

同 seed 的場景/query/time 抽樣必須完全相同；support 使用獨立 RNG。檢查初始參數與相同輸入的輸出相同，但不要求不同 support 造成相同的第一步 loss。各組獨特 support 數、重複率、類別與缺少成功 query 的步數都保存。

訓練模板分布：00003=3、00005=4、00006=14、00011=13、00016=3、00017=32、00018=27、00019=18、00020=2、00021=55。樣本分布不均，必須報逐模板結果。

## 評估面板

主要 readout 固定 adapted probe seed 0；其餘兩個 adapted heads 和原 head 全部保留作敏感度檢查。每次比較 base 與 memory 都使用同一個 head。

**自然第一次失敗：** 原 cross D0 的全部 100 題、每題相同 1,024 shortlist。固定 base＋主要 head 的第一個 action 若失敗，就用該次經驗作 context，排除這個已試 action，其餘 1,023 候選作共同 query；74 題符合此條件。第一個 action 已成功的 26 題保留於流程統計和空 memory 身份檢查，不混入「第一次失敗修正收益」的平均。AUC 另外要求 query 同時含成功與失敗，分母在結果中另報。

**精選 support：** 將同一 D0 候選池依上述規則分區，固定 query 後分別提供 N/H/F。39 題同時具備三類：00000=23、00002=10、00007=6，00014=0。這個面板無法回答 00014 的三類差異；自然面板仍保留全部 00014 題目。Calibration 使用上述 49 題做同樣測試。

本 pilot 每場景每類固定抽 1 條 donor，而非把「最多 4 條」全跑滿，以控制成本。Donor 用固定 hash 選取，不看 query 標籤或新模型效果。這限制對同類 support 內部差異的判斷，不能宣稱完整 donor 平均。

為精確重現舊 base，cross 前向保持完整 1,024 候選與 batch 32；精選面板只在固定 query mask 上算指標。四個 heads 的 base 分數、位置、距離、latent 誤差都逐元素比對既有 probe 實驗 cache。Geometry 與分數輸出保留完整陣列，query 真實未來只作 target。

主要結果是自然面板的 Near−Uniform、Mixed−Uniform、Mixed−Near 場景平均 AUC。另報成功／失敗分數變化、第一個成功排名、Hit@10、latent／幾何誤差與精選面板結果。逐模板、逐 seed 完整保留；bootstrap 在場景層，按模板分層並對三 seed 差值先取平均，區間條件於這些模板與已訓練模型，不能冒充新模板或預訓練 seed 的不確定性。

## 執行與驗證

Run：`results/support_utility_20261004/`。凍結程式在 `code/`，準備階段原始版本保留在 `prep_code/`。資料、來源、模型與 head hashes 保存於 manifest，不能在檢視效果後覆寫。

準備工作：34366 資料盤點；34367／34368 支援區 base 評分；34369 凍結正式資料。四項 CPU 測試通過，涵蓋 cluster 重疊整組分配、top20 包含成功、抽樣 RNG 配對，以及缺類不能算可比場景。

34370 執行三組各三步的 GPU smoke、四個模板的 cross 小面板與原始陣列核對。34371 獨立重算實際資料的幾何分類、top20、cluster 分區、共同訓練集合與自然第一次失敗。正式訓練必須等待兩者成功。

正式訓練、評估最多各兩個 GPU 工作同時執行，後續評估和 CPU 彙整依賴前段成功。作業 IDs／實際狀態以 run/jobs.json 與 STATE.md 為準。Smoke 是工程檢查，不依它的模型分數決定是否訓練。

## 本輪的限制和接續

本輪未新增 rollout，未提供新場景確認，未學新 gate，也未測 K4 混合 support。配方效果混合了 support 内容、出現頻率和多樣性，不能把差異只歸因於「快成功」這一個物理屬性。

先完成九組 K1 matched training 與共同評估，再按原設計準備 D1 曝光稽核與收集、K4 適用性測試及必要的閉環比較。不因舊模型或這輪 smoke 的效果不好，就把 near-success 資訊判為無用。
