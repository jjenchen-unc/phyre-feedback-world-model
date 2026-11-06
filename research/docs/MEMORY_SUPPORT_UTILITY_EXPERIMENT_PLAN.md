> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Memory 如何從失敗經驗改善成功預測的實驗設計

2026-10-04。設計完成，尚未建立本輪資料、訓練或提交工作。本文供實作與 teammate review 使用；完成資料盤點後，才能凍結包含實際 IDs、檔案 hashes 與成本的執行設定。

我們要驗證的想法是：接近成功的失敗經驗，是否比其他失敗更能幫助 memory 預測成功 action；以及透過訓練資料的組合，能否讓 memory 學會採用有用的修正、忽略不適用的修正。

這次優先研究學習原因，暫不新增 fallback、gate、ranking loss 或 latent 架構。這些改動會讓資料配方的效果難以歸因。原 task-centric 方向保留，執行優先序由本設計取代。

## 要分開回答的三個問題

1. **現有模型會不會利用 near-success support？** 固定模型和 query，只替換 support。這是可選的現況診斷，不能把負結果當成資訊本身無用，也不是第二階段訓練的通過門檻。
2. **特別提供這種 support 訓練，能否學出更好的修正？** 固定 loss、query、初始化與訓練預算，只改 support 抽樣。
3. **模型學會選擇了嗎？** 在沒有 near-success、只有一條 near-success，以及混合干擾經驗的情況評估。平均效果改善不等於學會篩選。

重要反例是「base 原本給高分，但實際軌跡離成功很遠」的失敗。它可能透過降低其他失敗 action 的分數改善 ranking。因此測試的不是只把成功者分數拉高，也包含排除被高估的失敗者。

## 既有證據與本輪起點

目前 geometry memory 的 cross AUCCESS 高於同 head 的 base，但 Success@100 較低。丟失紀錄多數在第一次 support 修正就把成功候選推到嘗試預算外，見[損失歸因](MEMORY_CLOSED_LOOP_LOSS_ANALYSIS_20261002.md)。

現有 memory 已根據 query 初始場景與 action 讀取 support。Support 包含真實軌跡 latent、action、失敗結果，以及真實 latent 減 base 預測的 residual；它不是直接將一條 residual 複製到所有 query。最新訓練使用 latent L1 加 0.1 倍位置與方向 loss，見[既有訓練設定](MEMORY_GEOMETRY_FOLLOWUP_RUN.md)。

本輪先研究整條 support 經驗的效果。因為真實軌跡與 residual 同時輸入，若結果改善，不能單憑這輪宣稱 residual 通道是唯一原因。

## 資料定義

Support 是已執行、已知結果的 action。Query 是需要預測的其他 action。所有 support 都必須是真實失敗；成功 action 放在 query 監督或獨立的探索資料中，符合目前成功即停止的部署設定。

### 軌跡接近成功的定義

只在現有接觸任務適用的場景中分類；真正成功仍以 simulator 標籤為準。先沿用固定物件配對與表面距離計算，使用同一組未來取樣時間，不包含初始幀。以下 5／20 px 是本輪預先選定的分組門檻，不是已知物理界線。

| 類別 | 固定分類規則 | 想測的資訊 |
|---|---|---|
| N 接近成功的失敗 | Simulator 判定失敗，未來取樣中最小目標表面距離不超過 5 px | 接近接觸的經驗是否幫助預測成功 |
| H 被 base 高估的遠端失敗 | 最小距離至少 20 px，且 base 分數位於該場景 support 候選前 20% | 揭露 base 假陽性的經驗是否有用 |
| F 其他遠端失敗 | 最小距離至少 20 px，且不在上述前 20% | 一般遠端失敗的比較組 |

Base 分數用固定 adapted probe seed 0 和穩定 log scorer，在看 support 結果前計算；百分位母體是完整 support 候選，包含其中成功 action。最後只有失敗 action 可成為 support。N、H、F 互斥；5 至 20 px 的中間樣本保留在均勻抽樣和自然部署面板，不硬塞進任一類。

額外保留最小距離、近接觸時間比例、最長連續近接觸取樣段、初始是否已接近，以及 base 預測誤差。稀疏幀只能提供接觸代理，不能宣稱知道精確碰撞時間或持續時間；若補模擬取得密集序列，須另列一致的密集分類結果，不悄悄改主定義。

預先指定敏感度分析：N 門檻改為 2／10 px，F/H 邊界改為 10／30 px，逐次只改一個門檻。不依 memory 效果選最佳門檻；所有版本都報 coverage。若分類只剩極少場景，或 N 大多只是初始已接近，先報定義／資料不足。

### Action 接近與軌跡接近分開

Action 接近是另一個假設。只用獨立探索區的成功 action 作 anchor，記錄 support 到最近 anchor 的正規化 action 距離；不能用被評估 query 的成功標籤選 support。不同參數的尺度固定為各參數合法範圍。

先報 action 距離與 N/H/F 的交叉分布；近／遠二分僅用訓練資料距離的中位數固定門檻，再套到其他資料。沒有成功 anchor 的場景保留於主實驗，只不參與這個次要分析。Action 距離小不能替代軌跡接近，也不能當成 support 有用性的標籤。

## 場景與 action 如何切分

沿用現有 memory 的 12 個訓練模板與 4 個 calibration 模板。Calibration 模板曾被 base／原 probe 看過，不能稱為整個系統的未見模板。既有 cross 100 題是反覆使用的 D0，僅作機制探索。

新增 D1 規劃為同四個 cross 模板各 25 個未參與本系列分析的場景。先查曝光紀錄，再以固定 seed 20261004 抽取並鎖定 task IDs，不看 memory 效果挑題。這是新場景確認，不是新的模板集合。若曝光或資料可用性不足，明確報告，不能用 D0 補足後仍稱全新測試。

每個場景在看 support 類別／模型效果前，將 action 分為探索 20%、support 40%、query 40%。比例是目標，實際依 action 群分配。相同 rollout、重複 action、已知同一 anchor 的局部擾動群作為整群分配；任何 query 不得出現在 support。若無既有群 ID，使用固定 action 空間網格建立群：正規化後每軸寬度 0.02，並記錄跨區最近鄰距離；網格不保證完全隔開所有相近 action，不能過度宣稱去除了近鄰相關。

既有 packs 由多種收集方法產生，先按 task/action 去重並查群組來源。所有 arm 使用同一版新切分；既有 geometry checkpoint 只作參考，不替代新資料上的 matched control。

訓練 query 每步从固定場景 query 區均勻、不依標籤抽最多 8 個，不在本輪同時改成功／失敗比例。先盤點成功、base 高分失敗、其他失敗的數量與每步缺少成功 query 的比例。沒有成功 query 的場景仍可訓練 prediction，AUC 則未定義，不填 0.5。

先使用現有資料，不自動補到某個成功數量。若三類 support 覆蓋不足，另凍結一輪有上限的補資料設定，記錄隨機、base 政策和成功 anchor 擾動的各自預算。補資料不得看新 memory 的效果決定收集何時停止；每個 arm 使用同一新增資料 union。本文件不假設已具備足量 N/H/F。

## 第一階段 固定模型替換 support

此階段可省略。訓練前必要的是資料與實作檢查，不是證明舊模型已能利用新配方。使用者要驗證的「能否學會」必須透過第二階段的匹配訓練；第一階段即使完全負向，也不因此取消第二階段。

使用既有 geometry memory seeds 0/1/2；主要 readout 固定 adapted probe seed 0，其餘兩個 adapted heads 與原 head 作敏感度檢查，不選最好 head。

每個場景的 query 集完全固定。從 support 區的 N/H/F 各用固定 hash 選最多 4 條，不依 query 真實結果或模型改善挑選。三組 donor 數取可用數的共同最小值；K 始終是 1，每條 donor 各自評估，不能將 4 條合成 K4。Empty context 作 base identity 參考。

主要比較使用 N/H/F 都存在的同一批場景；另報各類覆蓋率、全場景缺失情況，以及 N-H、N-F 各自可用的配對面板。後兩者的分母不同，不能把平均數混在一起。Support 數多的場景不能因此占較大權重。

| 比較 | 回答的問題 |
|---|---|
| N 對 empty | 目前模型使用 near-success 經驗後是否真的改善 |
| N 對 H | 快成功的經驗是否比揭露假陽性的經驗更有用 |
| N 對 F | 快成功的經驗是否比一般遠端失敗更有用 |
| H 對 empty | 是否存在遠離成功、卻仍有價值的經驗 |

這是替換整條 support 的效果；它同時改變 action、軌跡與 residual。即使 N 勝出，也不能宣稱只改「距離成功遠近」便造成收益。再按 support 的 base 分數、residual 大小與 support-query action 距離分層；只在有共同範圍的樣本做匹配敏感度分析，不靠外推消除差异。

同時保存 query 真實碰撞／軌跡關係作事後解釋，明確標成 oracle；不能進入 support 選擇或模型輸入。

## 第二階段 固定訓練設定只改 support 配方

三個 arm 都在同一批具備 N/H/F 的訓練場景續訓；每一步使用相同場景和相同 query IDs。這個共同子集是條件式訓練面板，須報它占全部訓練場景的比例以及模板分布。不要讓 Near 組獨自丟掉較難場景。

| Arm | 每步的 K1 support 抽樣 | 用途 |
|---|---|---|
| Uniform | 从 support 區全部真實失敗均勻抽樣，包括中間距離 | 匹配本輪資料與計算預算的控制 |
| Near | 只從 N 抽樣 | 驗證集中提供接近成功的經驗是否有效 |
| Mixed | 以各 1/3 機率選 N/H/F，再在類內均勻抽樣 | 驗證保留有價值的反例是否優於只給 near-success |

三組都從同一原始 R0 初始化，每組 seeds 0/1/2，共 9 次續訓。沿用 2,000 steps、AdamW lr 3e-5、weight decay 0.05、cosine、clip 1；base 與 probe 固定。Loss 固定為未來 latent L1 加 0.1 倍既有幾何 loss，每 query 固定 8 個幾何監督時間點。

場景/query/time 抽樣序列須在同 seed 的三組完全相同，support 抽樣使用獨立 RNG，不能影響下一步 query。Support 不同，所以第一步 loss 不要求相同；應檢查初始參數相同，且給相同輸入時輸出相同。只比較最後一步 checkpoint，不在 cross 挑 checkpoint。

各組 support 的出現頻率和獨特樣本數不同，這正是抽樣介入的一部分。完整報出重複使用率與有效多樣性，不能宣稱頻率、內容、多樣性已被分別識別。若 Near 明顯過擬合少量 donor，再以單独的新實驗分解，不在本輪臨時替它加資料或改步數。

三組都在第一階段同一矩陣評估，也在自然政策面板評估：使用完整候選中固定 base＋adapted head 選出的第一次失敗，排除該 action 後，對相同剩餘 query 評分。第一次已成功、無解、沒有可用失敗的場景都列入流程統計。這個面板不強制提供 near-success support，不用 query 未來選 context。

## 第三階段 檢查是否真的學會忽略不適用的經驗

第一輪先保留 K1 訓練，在評估中做下列 K4 壓力測試；清楚標記 K1 到 K4 的分布差異，不把結果單獨解釋成「學會／不會篩選」。所有模型用同一組 query、同一組 support tuples。

- N＋3 條 F：是否保留單條 N 的收益。
- H＋3 條 F：是否也能利用揭露假陽性的經驗。
- 4 條 F：沒有 near-success 時是否仍做大幅有害修正。
- N＋H＋2 條 F：两種不同資訊同時存在時是否能使用。

每組都是真實、不同的 action，資料不足不重複填滿。成對比較只在兩條件都可構成的場景進行。固定數組 support 順序並完整報告，避免挑有利排列。

額外做 support removal：固定其他三條，移除 N、H 或一條 F，觀察 query 分數與誤差變化。不能用 attention weight 大小代替行為證據，也不能预先把 F 標成必然無用。

若需要聲稱「混合訓練教會篩選」，再開一個匹配的 K4 訓練對照：Uniform4 與 Mixed4，其他條件相同、相同 R0 起點與預算。Mixed4 的四種上述組合等機率，兩組使用同一批可構成全部組合的訓練場景。空 context 只作評估身份檢查；empty gate 本身使空輸入等於 base，這不能算學會拒絕非空但不適用的 support。此分支在 K1 資料和效應盤點後另凍結實際資料，不預設已排程。

## 評估指標與判讀規則

主要離線指標是場景平均 ranking AUC 的成對差；主要 readout 是固定 adapted seed 0。次要指標為第一個成功排名、Hit@10、成功／失敗分數改變、正確順序被改錯的比例，以及成功／失敗分開的 latent L1、幾何誤差和接觸附近誤差。Log score 不稱為校準成功機率。

尤其要區分「成功者分數下降」與「失敗者分數上升」。報告每個 support 對不同 query 的影響分布，而不只替 support 給一個總平均好壞標籤。

第一階段四個主要比較全部保留；第二階段主要比較是 Near−Uniform、Mixed−Uniform、Mixed−Near，在自然政策面板上計算。按場景成對重抽樣 10,000 次，先在 donor 和同場景內平均；报告每個模板與每個 seed。三個 seeds 共用 R0，不當三次獨立預訓練。四個 cross 模板下的區間是條件於這些模板的證據，不用來保證未見模板效果；若做正式多重檢定，階段內使用 Holm 校正。

實用效果門檻預先定為 AUC +0.02，作研究決策標準而非科學常數。點估計超過門檻但區間跨零，標為線索；區間涵蓋正負實用效果，標為無法分辨。區間上界低於 +0.02 只能說未支持此規模收益，不說完全無用。N−empty 與 N−H/F 都要看，避免「比較不差」被寫成「確實有幫助」。

| 結果 | 可以支持的判斷 |
|---|---|
| 舊模型 N 無益，但 Near 訓練後在新場景有益 | 舊模型未學會利用此類 support，訓練配方可改善 |
| N 在配對面板勝出，但 Near 在自然政策面板退步 | 精選支援下有收益，但資料配方不適合自然支援分布 |
| H 對 empty 有穩定正收益 | 不支持「只有接近成功的經驗有用」的強主張 |
| Mixed 在自然面板優於 Near，且干擾測試較穩 | 支持保留不同失敗類型；仍需 K 匹配對照才能談學會篩選 |
| N/H/F 都降低 L1 卻普遍破壞 ranking | 此配方下 support 分類尚不足以解決目標與決策不一致 |
| 幾何改善只出現在訓練用 probe | 可能是讀出特定效果，不能當成物理資訊普遍改善 |
| 有幫助與否更依賴 support-query 關係，而非 N/H/F | 後續應學 query 條件的適用性，而非只做 support 品質分類 |

負結果只限制本輪分類、資料、初始化、預算與模型；不證明某類經驗不含資訊，也不證明所有 memory 方法不可行。沒有直接的 learned gate 時，不把「修正比較小」當成已學會選擇。

## 新場景確認與閉環

D0 用於除錯和機制探索。鎖定三組訓練與分析後，D1 一次評估全部模型，不根據 D1 再選門檻或調 loss。D1 固定每模板 25 題是初始精度規劃，並非已達檢定力；以訓練／calibration 變異估算可分辨幅度，在解盲前若需改樣本數必須記錄，之後不依效果加題。

若進入閉環，base、Uniform、Near、Mixed 使用同一候選生成程序、1024 shortlist、100 次嘗試、最多 10 次更新與最近 4 次失敗；全部使用相同 adapted probe，不加入 fallback。完整測全部 arm 與 seeds，不只選離線贏家。主要看 AUCCESS 與 Success@100 的取捨，並保留第一次 support 更新的診斷。

固定 support 的機制評估可使用 simulator 資料離線選 N/H/F，需標為精選 support 條件；閉環只能依序觀察實際執行結果，成功即停。若沒有實際完整模擬而使用候選 cache，明確稱 cached simulator replay。

## 執行前檢查與產出

先完成資料盤點：各 split/template/scene 的 N/H/F/中間類數量、共同可比場景、成功 query 數、原始資料來源、action 群重疊、初始已接近比例、時間解析度與曝光紀錄。這些是真正決定能否執行的資料條件，目前尚未量測。

執行設定需保存 task/action IDs、群組、support/query 分割、donor 抽樣、初始化與所有 heads hashes、數值路徑、seeds、訓練步數、主要比較及程式 hash。先做預先指定的 8 場景工程 smoke，再估成本；不沿用舊閉環時間假裝知道本輪完整成本。

必要驗證包括：empty 等於 base、support 全為真實失敗、support/query 不重疊、query 未來只作 target、base/probe 權重不變、同 seed 的場景/query/time 序列一致、穩定分數與 action 標籤對齊，以及從逐 query 分數獨立重算全部排名與彙整。資料或數值检查失敗時先修正，不發布原因結論。

必要順序是 coverage 表與資料／實作檢查，接著 9 組 matched training 與自然政策測試。固定模型 support 效果矩陣是可選的現況診斷，不作訓練 gate。這輪的核心成果應是「哪種 support 在哪些 query 上有用、資料配方是否改變了這個關係」，不是再多一個總體 latent loss 數字。
