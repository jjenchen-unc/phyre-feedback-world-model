# 實驗流程、動機與 milestone 數字對照

整理：2026-11-06；實驗結果截至 2026-10-21。這次是文件整理，沒有新增訓練或修改投影片。給接手同學的研究問題與實作入口見 [bottleneck 與接手文件](TEAM_HANDOFF_BOTTLENECKS.md)。

## 先知道我們在測什麼

我們想知道：在同一個物理場景，一次失敗的實際觀測，能否幫助 world model 選出更好的下一個 action？每次嘗試都重設物理場景，只有模型的 memory 留下來。Within 使用訓練見過的 task templates 中的新場景；cross 測未見過的 templates。目前主要 panel 是官方 within fold 0 與 cross fold 1，各 100 個場景；不是每個 fold 只有五個場景。

目前候選流程是 initial-scene classifier 排序官方 100,000 個 action proposals，留下 1,024 個合法候選。Q0 預測候選的 future latent，probe 解碼幾何，再由 contact scorer 排序、選第一個 action。失敗後，memory updater 使用該次觀測更新 tokens，再重新預測、排序未嘗試的 candidates。Q0 是有 initial memory tokens、還沒做 feedback update 的 predictor；Base 沒有這些 tokens。

Support 指已執行、有實際觀測的 action；query 指要預測或排序的未嘗試 action。訓練可以取得 query 真值作為 loss，但部署時不能先看 query 的真實影片或成敗。詳見 [README 的模型說明](../README.md#model-and-training)。

## 實驗怎麼走到現在

| 階段與要回答的問題 | 設計／為什麼這樣做 | 得到的資訊與下一步 |
| --- | --- | --- |
| 公平 pretraining：比較是否被起點與 loss 混淆？ | Base、Q0 都從隨機 predictor 權重開始；配對 backbone seed、資料順序、8 epochs、latent L1 + λ geometry。Cross 52,276、within 52,349 條 trajectories；四種 λ，合計 24 runs。 | 建立可用的公平起點；不能把所有 pretraining dev loss 當成新的部署成功率。後續固定 cross λ=0、within λ=.03、seed 0。 |
| 候選池修正：測到的是否是部署時會遇到的 actions？ | 撤回用固定舊 Base 篩選 shortlist 的相關 ranking 比較；比較新版 Base、Q0、classifier 從 100k 選 1024 的覆蓋率。 | Classifier 的 pool 覆蓋率最高，後續固定此 selector。舊失效結果仍留存並標記，不拿來解釋新版失敗。 |
| Feedback：有 memory 是否優於固定排序？ | 先做 classifier-first；之後改成 classifier 選 pool、Q0 選第一個 action，訓練也使用當下 predictor/scorer 選 support。比較 Q0、residual、zero residual、AdaJEPA-style。 | 小規模多輪結果有改善，但 zero control 也能改善，尚不能歸因於觀測 residual。 |
| Oracle：是預測不好，還是下游完全讀不出成功訊號？ | 同 pool、同 probe/scorer，把預測 latent 換成各 action 的實際影片 latent；另有真實幾何對照。 | 小樣本 oracle 很強，顯示 predicted representation 到決策之間有很大缺口；但不能分開 predictor 本身誤差與 probe 在 predicted latent 上的失配。 |
| Residual／loss：誤差中哪些部分有用？ | 同 checkpoint 比 raw、去平均、zero；再比較 L1、MSE、scaled Smooth L1，保留 geometry loss。 | 首 seed 的 centered／Smooth 組合較好，但 raw Smooth 第二 seed 未重現改善；不能宣布新 loss 穩定勝出。 |
| Probe：重新適配下游讀取是否足夠？ | 對目前 predictor 的資料做 probe adaptation，先看小驗證集幾何誤差與 ranking。 | Mixed adaptation 降低部分 predicted geometry error，平均 AUC 未改善，未進入 closed-loop。尚不能說 probe 完全沒有問題。 |
| Success objective：直接教 updater 排名是否有效？ | 16 組短程 fits：原 prediction loss、BCE、pairwise、可學習 calibration、contact-distance target，搭配 real／zero 和部分第二 seeds。 | 沒有穩定 cross 贏家。縮小更新也沒有穩定救回 ranking；不支持「只需換 loss」或「只是 update 太大」的單一解釋。 |
| 獨立 ranking head：能否避開修改 latent？ | 固定 Q0/probe，對分數加小型 MLP correction；2 seeds × real／zero／query-only，共 6 fits。 | Support context 有訊號；real residual 沒穩定勝過 zero。接著拆解 support context。 |
| Support feature ablation：到底是 action 還是預測軌跡？ | 10 種 masks × 2 seeds，共 20 fits，全部 residual=0；加入完整 query distance 對照，避免相對特徵偷偷多帶 query 資訊。 | Support 預測距離特徵 + query–support 距離差值得研究；它們全都能在試 action 前計算，還不是觀測 feedback 有效的證據。 |

來源：[pretraining 表](../results/pretraining_dev.csv)、[失效結果清單](../research/results/INVALID_RESULTS_20261016.json)、[milestone 原始結果](../results/milestone.json)、[loss 比較](../research/results/loss_design_iteration_20261019/FINAL_SUMMARY.json)、[probe adaptation](../research/results/current_probe_adaptation_20261020/FINAL_SUMMARY.json)、[success objective](../research/results/success_objective_loop_20261021/FINAL_FINDINGS.json)、[head](../research/results/feedback_readout_loop_20261021/FINDINGS.json)、[feature ablation](../research/results/support_context_ablation_20261021/FINDINGS.json)。

## 三套結果不能直接放在同一排行榜

| 比較 | 第一個 action | 訓練／評估形式 | 正確指標 |
| --- | --- | --- | --- |
| 10/18 舊 milestone | Classifier top-1 | Updater 256 slots × 2 episodes；最多 9 次嘗試 | 多輪 S@9；`results/milestone.json` 的 `classifier_first_feedback` |
| 現有 slide 的主表 | Q0 top-1；無 memory Base 自選 first action | Updater 128 slots × 2 episodes；最多 9 次嘗試 | 多輪 S@1/3/5/9 |
| 10/21 最新 head 診斷 | 固定 Q0 top-1 | 只分析第一次失敗後的一次重新排序 | S@2，以及 97 個 first-failure 場景的下一批 top-k coverage |

因此 README 舊表的 within Q0 **53%** 與 slide 的 **50%** 都有來源，差別是 protocol。最新 head 的 **10/100 S@2** 也不能跟 slide 的 **15% S@9** 比大小來選方法。

同輪內控制比較的公平性，與不同輪訓練成本相同，是兩件事。最新 updater 是每步 8 queries、128 steps；head 是每步全部 424 queries、512 steps。AdaJEPA-style 也有額外 test-time optimization，並非同 compute 或官方 AdaJEPA 重現。

## 現有 slide 的數字是否 align？

**主要數字 align，但 slide 的研究進度比已完成的後續實驗舊。** 核對的是 `COMP669_Milestone_20261018.pptx` 與同名 PDF 的實際內容；不是依檔名猜測版本。簡報與 PDF 未在本次改寫，也沒有加入公開 repo。

### Slide 7 主表與 slide 11 完整表

下表單位都是百分比；每個 panel n=100，因此也是成功場景數。`w` 為 within fold 0，`c` 為 cross fold 1。

| 方法 | w S@1 | w S@3 | w S@5 | w S@9 | c S@1 | c S@3 | c S@5 | c S@9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base，無 memory | 26 | 36 | 39 | 46 | 3 | 4 | 7 | 7 |
| Q0，不更新 | 29 | 42 | 47 | 50 | 3 | 7 | 8 | 11 |
| AdaJEPA-style | 29 | 45 | 51 | 58 | 3 | 5 | 6 | 9 |
| Memory，L1，raw residual | 29 | 45 | 50 | 56 | 3 | 7 | 10 | 12 |
| L1，inference centered residual | 29 | 46 | 51 | 57 | 3 | 6 | 10 | 15 |
| Smooth L1，inference centered residual | 29 | 48 | 54 | 61 | 3 | 9 | 12 | 15 |
| Smooth L1，raw residual | 29 | 46 | 51 | 61 | 3 | 7 | 10 | 12 |
| Smooth L1，inference zero residual | 29 | 43 | 53 | 57 | 3 | 10 | 12 | 14 |

前六列的 S@9 對應 slide 7；全部八列對應 slide 11。來源：[Base/Q0/AdaJEPA/L1](../research/results/plain_base_comparison_20261019/SUMMARY.json)、[loss 與 input controls](../research/results/loss_design_iteration_20261019/FINAL_SUMMARY.json)。同一場景的 `first_success` 可獨立重算各欄，見 [核對程式](../scripts/verify_slide_results.py)。

這裡 centered 是**同一個 raw-residual 訓練 checkpoint 的 inference intervention**，不是「去平均後重新訓練」的結果。其訓練目標仍含 .1 × geometry loss；Smooth L1 使用訓練資料校準尺度。Slide 中的 Base 自選 first action，因此與 Q0 的差異也包含第一個 action 選擇。額外的 Base-with-Q0-first 對照為 within 48%、cross 10% S@9，已在同一原始 JSON，未列入 slide 主表。

### Slide 8 oracle 與 slide 13 候選覆蓋率

| Slide 8：靜態 oracle 診斷 | 場景數 | Q0 S@9 | 實際影片 latent S@9 |
| --- | ---: | ---: | ---: |
| Within | 25 | 60% | 96% |
| Cross | 10 | 20% | 100% |

所有診斷 pool 都有成功 action；不是主表的完整 100 scenes，也不是可部署的 policy。README 的 oracle 段落比較的是 **Base**（within 40%、cross 10%），slide 比的是 **Q0**（60%、20%），兩者不是矛盾。

| Slide 13：選出的 top-1024 至少含一個成功 action | Base selector | Q0 selector | Classifier selector |
| --- | ---: | ---: | ---: |
| Within，100 scenes | 94% | 91% | 98% |
| Cross，100 scenes | 54% | 51% | 61% |

來源均為 [milestone JSON](../results/milestone.json) 的 `oracle_1024` 與 `coverage_100k`。這張 coverage 表使用官方 cache labels；feedback 實驗使用重新模擬的 labels。精確計算後續成功率上限時應用該輪 pool 的 fresh labels，不能直接把不同 label 來源當成完全相同。

### 後續結果改變了哪些說法？

Slide 9 的「second seed running」已過時。Within 的 **raw-input** loss 比較已完成：

| Within S@9 | 第一 seed | 第二 seed |
| --- | ---: | ---: |
| L1，raw residual | 56% | 58% |
| Smooth L1，raw residual | 61% | 50% |

所以 raw Smooth 的優勢沒有重現。這不是 centered 組合的第二 seed 實驗，不能說已直接推翻或確認 centered 的 61%／15%。Cross 第二 seed 未通過預定啟動 gate，沒有完成該比較。投影片的首 seed 數字可保留為 pilot，但「Smooth L1 穩定比較好」超出證據。

之後的 head 實驗也讓問題更明確：目前改善可能不需要觀測 residual。最新 S@2 對照及建議實驗見 [接手文件](TEAM_HANDOFF_BOTTLENECKS.md)。

## 核對與保存

本次以公開 JSON 的 per-scene 記錄重算 slide 表；oracle 也從 first-success ranks 重算。PPTX／PDF 內容核對的檔案 SHA-256：

```text
PPTX 8336f484dcf5c8195c9eb3c286d1098b87f186af745e08e4a0993d87d0d1a144
PDF  297083c22567777d19e9e7ed3b1d2713969c02c96dbce67fa80a76cf81650d27
```

`research/` 保留原始計畫、結果與當時解讀；本文件補充後續證據，不回頭改寫實驗記錄。文件編寫日期不代表新的實驗日期。這次沒有新的 performance claim。
