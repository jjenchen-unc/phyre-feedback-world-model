# Memory updater：夜間歸因實驗

日期：2026-10-19。**八輪調查已完成，包括四組新 updater 訓練與 1,800 條新增的完整閉環 rollout；結果已核對。** 這份紀錄不加入 milestone slides。

**最值得優先探索：保留 uniform query sampling，從訓練開始就對 residual 扣掉每個 channel 的時空平均值。** 目前只驗證了推論時的操作，尚未驗證這種新訓練方法。它在兩個訓練長度、cross／within 的四個 uniform 對照都小幅改善；最後的歸零控制也支持繼續保留 residual 內容。不過，這是下一個最有依據的小實驗，不是已找到普遍根因或穩定勝出的模型。

我們想回答的是：為什麼 memory 看了失敗經驗之後，未必更會選成功 action？這次把「模型收到什麼訊息」、「更新幅度與方向」及「訓練 query 的組成」分開檢查。沒有把以前失效的候選池實驗混進來。

## 共同設定

目前 pipeline 先由 initial-scene classifier 對 100,000 個 actions 排名，保留前 1,024 個。Q0 predictor 加上固定 probe／scorer 對這 1,024 個重新排名，選出第一個 attempt。失敗後，實際影片 latent 減去**當下 memory 狀態的預測 latent**，得到 residual；updater 以 residual、attempted action 和目前 memory 更新 tokens，再對剩餘候選重新排名。最多嘗試 9 次。

Q0 指沒有利用 trial feedback 更新 memory 的版本。S@9 指 9 次內至少成功一次的場景比例；它不是 latent loss，也不是所有 action pairs 的 ranking AUC。

這次沿用 cross fold 1、within fold 0，各 100 個 evaluation 場景，分別涵蓋 5 與 25 個 templates。Cross 沿用幾何 loss 權重 λ=0 的 Q0，within 沿用 λ=0.03 的 Q0，皆為 seed 0。這是 Q0 預訓練的設定；後續 updater 訓練的 loss 另固定為 latent L1＋0.1×geometry。兩個 protocol 使用各自固定的 Q0 起點，不能把 protocol 間的差異全歸因於 cross／within 本身。比較時候選池、第一個 action、預算、probe、scorer 都相同。

候選池本身含有至少一個成功 action 的場景是 cross 61/100、within 98/100。因此，固定候選池下的完美排序也無法解出剩餘 39／2 個場景。這是所有比較組共同的限制，不能用來解釋 residual 為什麼比 Q0 差。

## 已完成的來源結果

這張表全部來自相同的 Q0-first evaluation。Residual 與 zero control 各訓練 256 slots，每 slot 兩個 episodes，使用 64 個 updater 訓練場景；不是這晚新訓練的 128-slot 比較。

| 方法 | Cross S@1 / S@3 / S@5 / S@9 | Within S@1 / S@3 / S@5 / S@9 |
|---|---|---|
| Q0，不更新 | 3 / 7 / 8 / 11% | 29 / 42 / 47 / 50% |
| Residual updater | 3 / 4 / 7 / 8% | 29 / 40 / 49 / 57% |
| 另行訓練的 zero-residual control | 3 / 7 / 9 / 13% | 29 / 42 / 46 / 50% |
| AdaJEPA-style test-time adaptation | 3 / 5 / 6 / 9% | 29 / 45 / 51 / 58% |

Residual 在 cross 比 Q0 少成功 3 個場景，實際是救回 2 個、失去 5 個；within 多成功 7 個，實際是救回 12 個、失去 5 個。**不能概括成 memory 全部失敗，也不能因 within 多 7 個就宣稱方法已穩定有效。** 這是單一起點、單一 fold 的 pilot。

## 第一輪：是不是只讀 residual 的平均值？

固定原始 rollout 的歷史、memory、support action，分別保留完整 residual、只留每個 channel 的時空平均值、或扣掉平均值。每個 protocol 10 個場景，各看第一次與較後期更新，共 40 個狀態。這是單步反事實診斷，並不是新的 9-attempt evaluation。

第一次更新中，平均值只佔 pooled residual 約 4.3%／4.9% 的能量。扣掉平均值後，memory 更新與原始更新的相對差異中位數約 8.6%／9.8%；只留平均值的差異約 36.5%／85.9%。

**結果不支持「updater 主要只讀整體平均值」。** 結合先前打亂 residual 時空排列幾乎不改變更新的診斷，目前較合理的描述是：模型會利用 residual 的內容分布，但對內容與顯式時空位置的對應較不敏感。這不等於 latent 本身沒有幾何資訊，也沒有證明位置敏感度就是失敗根因。

## 第二輪：是不是更新太大，或零 residual 也會把 memory 推錯？

先將更新縮成原本的 1/4 或 1/2，再檢查單步 ranking。某些場景的 AUC 確實比較好，但最佳成功 action 的名次及下一次 attempt 成功率沒有一致改善。**AUC 變好不能替代 task success 的驗證。**

另一個實驗把 updater 拆成兩部分。令 U(m,r,a) 為更新後 memory：

- 原始更新：`U(m,r,a)`。
- 去掉零輸入漂移：`m + U(m,r,a) − U(m,0,a)`。
- 對照：保留原始更新方向，但把長度調成上述 residual-specific 更新的長度。

這輪使用每個 protocol 原有 10 個與新增 10 個場景。共 79 個不同狀態；其中一個新 within 場景只有一次更新，因此沒有重複計入「第一次／較後期」。474 組 score arrays 通過獨立驗證。

目前沒有一致證據顯示扣掉零輸入漂移就會更好。例如，在新增 within 場景的較後期狀態，原始 residual 下一個 action 成功 4/9，扣掉漂移後為 2/9。不同分組會有相反結果。因此還不能把「零輸入動態有害」當作通用根因。這些修改也可能讓固定 checkpoint 收到訓練時沒見過的狀態。

注意：將 residual-trained updater 的 residual 歸零，是輸入干預；它與前表**另外訓練**的 zero-residual control 不同。

## 第三輪：完整多次嘗試驗證（完成）

在相同的 200 個 evaluation 場景上，跑縮小更新至 1/4、縮小至 1/2、扣掉 residual 平均值、扣掉零輸入漂移四組，共 800 條新 rollout。每組都沿自己的選擇收集 observation，重新計算當下 residual，沒有沿用原始方法的後續 support。

第一個 action 與完整 first-score vector 均與原始 Q0 完全一致。兩個 canonical encoder 的影片 latent 重現、原始 policy 的逐步 replay 已通過。800 條 rollout 全部完成，下載後另行核對每一步選擇、候選 mask、成功標籤與 checkpoint 不變性，全部通過。

下表每格以 100 個場景為分母。括號為相對原始 residual「救回／失去」的場景數。

| 方法 | Cross S@9 | Within S@9 |
|---|---:|---:|
| Q0 | 11% | 50% |
| 原始 residual | 8% | 57% |
| 更新縮成 1/4 | 12%（5／1） | 56%（5／6） |
| 更新縮成 1/2 | 10%（3／1） | 54%（6／9） |
| 扣掉 residual 的時空平均值 | 10%（3／1） | 60%（6／3） |
| 扣掉零輸入漂移 | 12%（6／2） | 46%（4／15） |

這輪改變了兩個判斷：

1. Cross 確實對更新幅度敏感，縮成 1/4 比原始 residual 多解出 4 個場景。不過只比 Q0 多 1 個；within 也沒有同樣提升，因此不能說「更新太大」已解釋全部問題。
2. 零輸入動態不是普遍有害。扣掉它對 cross 有幫助，within 卻少解出 11 個場景。不能把沒有 residual 的部分一律當作應刪除的雜訊。

扣掉 residual 平均值在兩邊都比原始 residual 略好，但 cross 仍低於 Q0。Within 的增益分散在若干 templates；移除任一 template 後，對原始 residual 的成功率差仍為正（約 +1.0 至 +4.2 個百分點）。這只是敏感度檢查，不是信賴區間，也不構成跨 seed 證據。

Cross 分題型看，還有比更新幅度更大的瓶頸。每個 template 都有 20 個場景：

| Cross template | 候選池有成功 action 的場景數 | Q0 成功數 | 原始 residual | 1/4 更新 | 去平均值 |
|---|---:|---:|---:|---:|---:|
| 00008 | 17 | 0 | 0 | 0 | 0 |
| 00010 | 10 | 1 | 0 | 0 | 0 |
| 00013 | 15 | 5 | 4 | 6 | 5 |
| 00015 | 13 | 5 | 4 | 6 | 5 |
| 00023 | 6 | 0 | 0 | 0 | 0 |

例如 00008 的多數候選池明明有答案，這輪所有方法卻都沒有在 9 次內找到它；不能用「候選池沒有成功 action」解釋這個題型的失敗。1/4 更新與去平均值目前的增益仍落在原本較能解的 00013／00015。另一項干預「扣掉零輸入漂移」雖在 00023 解出 2 個場景，但同時嚴重傷害 within。這些分解支持繼續檢查題型間的預測／排序泛化，而不是宣稱一個全域更新係數已解決 cross。

## 第四輪：成功 query 是否在訓練時太少？

檢查的是最新的 online-selected support 訓練，不是以前的小候選池。每個 episode 的 supports 由當下模型選；queries 則是另外抽取、沒有嘗試過的 actions，用來計算預測 loss。

原始 residual 訓練中，有實際 memory update 的 episodes，cross 226/389、within 174/358 的 8 個 query 全部是失敗 action。成功 query 佔全部 active queries 的比例分別只有 10.6%／16.0%。這可能讓平均預測 loss 主要獎勵「把失敗 action 預測得更好」，沒有充分保護成功 action 的預測；但單靠這個統計還不能宣稱它造成 ranking 退步。

另外，cross 144/389、within 149/358 的 active episodes 實際只有一次更新就計算 supervision。因此「第一步更新從未被訓練」並不符合目前實作。

已從相同 fresh updater initialization 完成四組訓練：cross／within 各自比較原始 uniform query 與最多 4 個成功＋4 個失敗 query。某一類不足時，以剩餘候選補滿 8 個。沒有使用任意 near-success 門檻，也沒有用測試標籤選 deployment action。

兩組固定相同 Q0 reader、initial tokens、optimizer、scene/cap draws、`latent L1 + 0.1 × geometry loss`，各跑 128 slots；只改 query sampler。這是新 updater 的訓練，沒有沿用上一輪 updater 繼續訓練。模型學出不同 policy 後，後續 supports 和成功時機可以不同；但第一個 action 固定由 Q0 選擇，所以同一 episode 是否需要更新應完全一致。完整 audit 已確認兩組每個 episode 的 active 狀態與每個 slot 的 optimizer-step 狀態都相同。

128 slots 只有來源實驗的一半，所以主要因果比較是 **fresh balanced 128 對 fresh uniform 128**，不是拿 balanced 128 與舊 residual 256 的差直接宣稱 sampler 有效。四組都已完成相同 100 場景／protocol、9 次嘗試的 evaluation。128 與 256 slots 也使用不同長度的 learning-rate schedule，因此不能只看 fresh cross 比原始 residual 高，就判定原始模型是 overtraining。

完整 audit 已通過。兩組每個 episode 的 active 狀態與每個 slot 的 optimizer-step 狀態逐一相同；後續選中的 support 可以不同。

| 訓練指標 | Cross uniform | Cross balanced | Within uniform | Within balanced |
|---|---:|---:|---:|---:|
| Slots／episodes | 128／256 | 128／256 | 128／256 | 128／256 |
| Active episodes | 192 | 192 | 186 | 186 |
| Optimizer steps | 122 | 122 | 120 | 120 |
| 成功 query 比例 | 9.5% | 49.8% | 15.5% | 50.0% |
| 沒有成功 query 的 active episodes | 115 | 0 | 92 | 0 |

每個 protocol 的 64 個訓練場景都有被抽到，但實際產生 updater gradient 的不同場景只有 cross 48 個、within 44 個。其餘場景第一個 Q0 action 已成功，因此沒有 feedback update。這符合目前訓練流程；它也代表此次 query 平衡沒有增加獨立的 feedback 場景數。

Within 的完整結果已驗證：uniform 的 S@3／5／9 為 **45／50／56%**，balanced 為 **40／46／58%**。Balanced 到第 9 次淨增加 2 個成功場景，是救回 8 個、失去 6 個；移除 template 00020 後，淨增益為 0。因此，不能把這個小差異稱為穩定提升。更明確的是：成功 query 已經增加，前幾次嘗試卻沒有因此變好。

Cross 的 uniform／balanced S@3、5、9 分別是 **7／10／12%** 與 **9／11／12%**。第 9 次的成功場景雖有更換（救回 4 個、失去 4 個），總數沒有增加。

所以這次直接干預的結論是：**增加成功 query 的確改變了訓練資料，但尚未帶來一致的 task success 改善。** 這反對「只要成功 query 夠多就會解決」；它沒有否定所有 success-aware training，也沒有驗證直接 ranking loss。

## 第五輪：去平均值的效果是否依賴特定 checkpoint？

第三輪結果到齊後，新增一輪 evaluation：在第四輪的 uniform／balanced 新 checkpoints 上，各比較完整 residual 與扣掉平均值，額外 400 條 rollout，不新增訓練。開始前，已用第三輪 checkpoint 精確重現兩個場景的完整 centered-policy score traces；400 條新 rollout 也已逐條驗證。

這能檢查去平均值的效果是否在不同訓練長度／query 組成的 checkpoints 上持續存在；因為初始化與 evaluation 場景相同，**不能稱為獨立 seed／fold replication**。若只對原始 checkpoint 有效，就不應把它當作通用方法。

所有 fresh 128-slot 模型的完整結果如下；每格分母都是同一批 100 個場景。

| 方法 | Cross S@1 / S@3 / S@5 / S@9 | Within S@1 / S@3 / S@5 / S@9 |
|---|---|---|
| Q0，不更新 | 3 / 7 / 8 / 11% | 29 / 42 / 47 / 50% |
| Uniform queries | 3 / 7 / 10 / 12% | 29 / 45 / 50 / 56% |
| Uniform＋推論時去平均值 | 3 / 6 / 10 / 15% | 29 / 46 / 51 / 57% |
| Balanced queries | 3 / 9 / 11 / 12% | 29 / 40 / 46 / 58% |
| Balanced＋推論時去平均值 | 3 / 7 / 7 / 11% | 29 / 40 / 45 / 56% |

同一個 uniform checkpoint 去平均值，cross 救回 5 個、失去 2 個；within 救回 3 個、失去 2 個。但 balanced checkpoint 在 cross 救回 2 個、失去 3 個，在 within 救回 1 個、失去 3 個。

把原始 256-slot 和 fresh 128-slot checkpoints 一起看，**uniform 訓練下去平均值的四個對照都略好**：cross 8→10、12→15；within 57→60、56→57。這是目前較有延續性的訊號，但差異小、共用 evaluation 場景與初始化；balanced 下反而兩邊都變差。因此不能說「平均值一定是該移除的雜訊」。

此操作是在 residual pooling 後、送入 projection 前，對每個 channel 扣掉該次 residual 在時間與空間上的平均值。它保留相對變化，但沒有改變 checkpoint 的參數。目前測到的是**推論時干預**，還沒有訓練一個從頭就使用 centered residual 的 updater。

## 第六輪：差異是否已經出現在第一次更新？

Within 共有 71 個場景的第一個 Q0 action 失敗。比較這些場景時，各方法拿到完全相同的初始 memory、失敗 action 與 observation；第一次更新前的 residual 也由同一個 Q0 predictor 產生。因此，這裡不會混入後續探索路徑不同的影響。

| 第一次失敗後的下一次 attempt | 不更新 Q0 | Uniform updater | Balanced updater |
|---|---:|---:|---:|
| 成功場景數／71 | 4 | 9 | 6 |

在 69 個仍有成功候選的場景中，balanced 相對 uniform 把最佳成功 action 的名次改善了 23 次、惡化了 42 次，其餘相同。這支持「兩組學到的修正方式不同，而且差異已經出現在第一步」，不能只用後續 action selection 不同解釋。

它也反駁了一個過度概括：**不是所有 memory 的第一次更新都會破壞 ranking。** Uniform 在這批 within 場景的下一次成功數，比 Q0 的 4 個增加到 9 個。增加成功 query 並沒有自動得到更好的第一次修正。

Cross 有 97 個第一個 action 失敗的場景，其中 58 個還有成功候選、可定義 ranking AUC。下一次 attempt 的成功數：Q0 為 2，uniform 為 1，balanced 為 2，uniform 去平均值為 1，balanced 去平均值為 4。

值得注意的是，uniform 去平均值沒有增加下一次成功數；第一次更新後的靜態 top-8 含成功 action 場景數還從 10 降到 7，但完整閉環 S@9 卻從 12 增到 15。**一次更新的 AUC 或 top-k 品質，不能直接推論反覆更新的最終成功率。** 去平均值可能改變了後續探索路徑；這輪不能把它的 S@9 提升直接稱為「第一次物理預測更準」。

這是由完整 rollout 的 scores 做的事後單步分析，不是另一組 S@9 實驗。

## 第七輪：訓練與部署的 residual 分布是否不同？

使用所有不同的第一個失敗場景：cross 訓練 48／部署 97，within 訓練 44／部署 71，共 260 個。固定 Q0 初始狀態，只讀既有真實觀測快取；沒有新模擬、encoder 或訓練。所有 168 個部署場景的預測 score chunks 精確重現原始 Q0，另外 20 個場景的 residual RMS 重現之前的診斷。

下表量測的是 **support action 的 Q0 預測誤差**，不是 memory 更新後所有 query 的 loss。

| 第一個失敗 action 的誤差中位數 | Cross 訓練 | Cross 部署 | Within 訓練 | Within 部署 |
|---|---:|---:|---:|---:|
| Future latent L1 | 0.1139 | 0.1397 | 0.1202 | 0.1202 |
| Future residual RMS | 0.2711 | 0.2934 | 0.2627 | 0.2717 |

Cross 的 L1 增加約 22.6%，RMS 增加約 8.2%。以真實影片 latent 相對初始 latent 的變化量規一化後，cross 誤差比值仍增加約 14.6%；within 約增加 0.5%。因此，online selection 流程相同，不代表 residual 分布相同。

這是分布差異的證據，**不是已證明失敗由 overfitting 或 residual 太大造成**。例如 cross template 00008 的 support L1 中位數只有 0.1195，與 00013 的 0.1228 接近；00008 的 17 個有答案候選池卻仍全部沒有被 Q0 解出。單一誤差幅度還不足以解釋不同題型的排序困難，也不能代替對 query 預測誤差的測量。

## 第八輪：不用 residual 內容，是否也能得到同樣的改善？

這輪固定 fresh uniform 128-slot checkpoint，把每次傳入 updater 的 residual 全部設為零，仍提供實際 attempted action 與目前 memory。失敗後繼續更新、重排；成功就停止。候選池、Q0 第一個 action 和 9 次預算都不變。

它與「另外訓練一個 zero-residual updater」不同，目的是檢查**同一個已訓練模型在推論時有沒有需要 residual 內容**。因為零輸入可能偏離訓練分布，即使歸零後變差，也不能單憑這點證明模型理解了物理誤差。它仍然知道試了哪個 action，以及目前尚未成功，並不是完全沒有 feedback。

兩個 smoke 場景已精確重現原先零 residual 單步干預的 score vector，再開始完整評估。這輪不需要影片編碼或新模擬。

完整 200 條 rollout 已完成，下載後另行核對每一步 score、argmax、已嘗試 action mask、成功標籤、checkpoint 與 Q0 第一個 score vector，全部通過。

| 同一 fresh uniform checkpoint 的輸入 | Cross S@1 / S@3 / S@5 / S@9 | Within S@1 / S@3 / S@5 / S@9 |
|---|---|---|
| 原始 residual | 3 / 7 / 10 / 12% | 29 / 45 / 50 / 56% |
| 去平均值 residual | 3 / 6 / 10 / 15% | 29 / 46 / 51 / 57% |
| 全零 residual | 3 / 7 / 7 / 9% | 29 / 41 / 49 / 55% |

Cross 去平均值相對歸零救回 8 個、失去 2 個場景；within 救回 8 個、失去 6 個。原始 residual 相對歸零則分別救回 4 個、失去 1 個，以及救回 5 個、失去 4 個。

這使「完全拿掉 residual 也一樣」在 cross 上較不符合目前結果。但 within 歸零仍有 55%，比 Q0 的 50% 高，與原始 residual 的 56% 很接近。**Within 的部分增益可以在沒有 residual 內容時保留；不能把全部增益都歸功於讀懂預測誤差。** 仍需記住，這是 residual-trained checkpoint 的輸入干預，零輸入可能是 OOD，不能直接證明模型使用了正確的物理因果。

## 目前可以與不可以歸因的部分

| 可能原因 | 這次實驗說明了什麼 | 還不能說什麼 |
|---|---|---|
| Action 與 residual 融合後，residual 被完全忽略 | 改變／歸零 residual 會改變更新與排名；平均值也不是唯一主導成分 | 不能因此保證模型讀出了正確的物理原因 |
| 更新太大 | 1/4 更新幫助 cross，沒有同樣幫助 within | 不能用同一個全域係數解釋所有失敗 |
| 零輸入也會動，所以零輸入漂移有害 | 扣掉它會嚴重傷害 within | 不能把 action-conditioned dynamics 一律刪掉 |
| 訓練成功 query 太少 | 直接提高到約 50% 後，cross S@9 不變，within 小幅且脆弱地增加 | 不能說只需平衡標籤；也不能說所有 success-aware loss 都無效 |
| 訓練／部署 residual 不同 | Cross 的第一個 support 誤差確實變大，within 大致相同 | 不是 overfitting 的因果證明，也不代表所有 query 的誤差分布 |
| Feedback 訓練場景太少 | 真正提供更新梯度的不同場景只有 cross 48、within 44 個 | 沒有增加場景數的對照，不能確認增加資料就會解決 |
| Support 的修正不能轉移到 query | 同一個 first support 已會產生不同、甚至較差的 query 排序，值得追查 | 尚未直接測量跨 action 修正的適用範圍，不能稱為已證明根因 |
| Latent loss 與 success ranking 不一致 | 平衡成功 query 不自動改善排名；單步 ranking 也不等於閉環成功 | 這晚沒有做「同一 query 的更新前後 latent error」完整歸因，不能說已隔離 predictor／probe／scorer 的責任 |

現階段最準確的描述是：**memory 會使用輸入，也有能力改善部分場景，但學到的修正不是對所有 action、題型與嘗試階段都可靠。** 這比「implementation 一定錯了」或「只缺更多資料」更符合已有證據。候選池、Q0 第一個 action、當下 predictor residual、凍結參數、選擇 mask 與 label 的檢查均通過；這只支持已檢查流程的正確性，沒有證明整個系統完全沒有 bug。

## 最後推薦方向

**先試「uniform queries＋訓練時就使用去平均值 residual」，保持其他設定不變。**

理由不是平均值佔了大部分 residual；第一輪已不支持這件事。理由是目前實際閉環證據：uniform 訓練下，去平均值在 cross／within、128／256 slots 的四個對照都略好；同一 fresh checkpoint 的完整 residual／去平均值，也都比歸零稍好。它是成本小、可以單獨歸因的修改。

這個推薦有明確邊界。Fresh cross 從 12% 到 15%，仍只解出 61 個有答案候選池中的 15 個；within 從 56% 到 57% 只有一個場景差。Balanced 模型去平均值反而變差。這些比較共享 seed 與 evaluation 場景，並經過多次自適應分析，尚未建立跨 seed／fold 的穩定性。**不能宣稱「去平均值解決了 memory」或「平均值就是根因」。**

建議下一個受控實驗只新增兩個訓練：

1. Cross、within 各從相同 fresh updater 初始化開始，沿用本輪 uniform queries、128 slots、相同 scene/cap draws、固定 Q0／probe／scorer，以及 latent L1＋0.1×geometry loss。唯一改變是每次 update 的 residual 在訓練時就扣掉時空平均值。
2. 評估原始 residual、去平均值、全零三種輸入，對照本輪 raw-trained checkpoint 的同樣三種輸入。主比較是「raw-trained＋raw」與「center-trained＋center」；其餘對照用來區分訓練適配與單純推論干預。
3. 同時看 S@3／5／9 和每個場景的救回／失去，並在共同的第一個失敗 support 下比較相同 query 的 latent／geometry error 與成功 action 名次。這才能檢查改善的是預測、排序，還是後續探索路徑。
4. 若改善仍存在，再用新的 seed 或 fold 重現。這一晚尚未執行這兩個新訓練；它們是後續提案。

不優先擴大 balanced-query 訓練，因為已成功改變 query 組成卻未得到一致收益。也不把更多資料、ranking loss 或重做 architecture 說成必然答案：這幾個仍有道理，但本輪沒有直接驗證其效用。先用最小的變動檢查目前最一致的訊號，能讓下一輪結果比較容易解釋。

## 實際耗時

第三輪的 800 條 rollout 分為 32 個 GPU 分片，最慢分片 44 分 53 秒；每個場景平均約 cross 102–106 秒、within 61–73 秒，取決於方法及是否提早成功。這些時間包含反覆對 1,024 個候選排名，並大量重用既有真實觀測快取，不能直接當作新資料、冷快取或較少 GPUs 的耗時估計。Canonical observation encoder 由另外兩張 A6000 共用；需要編碼的工作完成後，兩個專用服務已停止。

新訓練的實際訓練迴圈耗時：cross uniform 約 **2 小時 58 分**、balanced **3 小時 24 分**；within uniform **2 小時 31 分**、balanced **2 小時 53 分**。四組各一張 GPU 並行，另共用 encoder。這些數字不含約一至兩分鐘的初始化／preflight，也不是從零收集資料的成本。

新 128-slot 模型的 raw／centered evaluation 各使用 16 個 GPU 分片／protocol。Cross 最慢分片分別 **23 分 34 秒／23 分 45 秒**，within **18 分 20 秒／19 分 19 秒**。這是實際分片執行時間，不含等待訓練完成的排隊時間；兩種 evaluation 可同時使用最多 32 張 predictor GPUs。

最後的 200 條零 residual 控制使用 32 個 GPU 分片，最慢分片 **12 分 44 秒**，不需 observation encoding。上述所有 GPU jobs 均已完成；沒有留下本輪排隊或執行中的訓練／評估。

## 證據與重現

- [原始 Q0-first 結果](../results/q0_first_feedback_20261018/SUMMARY.json)
- [本次計畫與 deadline](../results/memory_overnight_investigation_20261019/PLAN.json)
- [Residual 平均值與更新幅度](../results/residual_mean_step_diagnostic_20261019/SUMMARY.json)
- [更新方向與零輸入漂移](../results/residual_direction_diagnostic_20261019/SUMMARY.json)
- [最新訓練資料 audit](../results/memory_overnight_investigation_20261019/TRAINING_AUDIT.json)
- [訓練 active episode／optimizer step 不變量說明](../results/memory_overnight_investigation_20261019/QUERY_BALANCE_PLAN_CLARIFICATION.json)
- [完整閉環干預計畫](../results/memory_overnight_policy_20261019/PLAN.json)
- [Fresh query-composition 訓練計畫](../results/query_balance_training_20261019/PLAN.json)
- [完整新訓練 audit](../results/memory_overnight_investigation_20261019/FRESH_TRAINING_AUDIT.json)
- [Cross 新模型逐條驗證與配對結果](../results/memory_overnight_investigation_20261019/FRESH_PANEL_cross.json)
- [Cross 固定第一次更新分析](../results/memory_overnight_investigation_20261019/FIRST_UPDATE_cross.json)
- [新訓練完整結果](../results/query_balance_training_20261019/SUMMARY_128.json)
- [新 checkpoints 去平均值結果](../results/query_balance_centered_eval_20261019/SUMMARY.json)
- [Within 新模型逐條驗證與配對結果](../results/memory_overnight_investigation_20261019/FRESH_PANEL_within.json)
- [Within 固定第一次更新分析](../results/memory_overnight_investigation_20261019/FIRST_UPDATE_within.json)
- [Q0 訓練／部署 residual 分布](../results/initial_residual_shift_20261019/SUMMARY.json)
- [完整閉環干預結果](../results/memory_overnight_policy_20261019/SUMMARY.json)
- [Checkpoint centering 檢查程式](../results/query_balance_centered_eval_20261019/code/evaluate.py)
- [配對與 template 敏感度分析程式](../results/memory_overnight_investigation_20261019/code/compare_completed.py)

- [最後零 residual 控制結果](../results/fresh_zero_feedback_eval_20261019/SUMMARY.json)
- [實際 Slurm 執行時間](../results/memory_overnight_investigation_20261019/JOB_TIMES.psv)
