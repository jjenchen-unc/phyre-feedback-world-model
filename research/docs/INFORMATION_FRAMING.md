# 研究重心：物理 attempts 的資訊內容、可提取性與價值

日期：2026-09-19。依使用者進一步釐清，本文更新 `PROPOSAL_REFRAMED.md` 的研究定位。原文的 predictor/scorer 診斷、資料切分與公平 baseline 仍適用；不再將「設計更好的 adaptation mechanism」或「提高 action ranking」當成唯一主要問題。本文沒有實驗結果，也未定案新的架構。

逐張操作與英文文案見 [SLIDE_UPDATE_GUIDE.md](SLIDE_UPDATE_GUIDE.md)，包含原 11 張投影片的編輯步驟、圖示資料流與講述提示。

## 1. 主要研究問題

> Previous attempts 提供什麼可被模型提取的資訊？哪些 attempts 會產生對其他 actions 有用的 evidence？這些資訊能否被整合、轉移，並在什麼條件下改善決策？

建議 working title：

**What Can a Physical Attempt Teach a World Model?**

*Information Extraction, Transfer, and Decision Value in PHYRE*

研究的主要對象是 **attempts 所提供的資訊與其可利用性**。Latent world model、memory updater、scorer 是提取、保存與衡量資訊的工具。Recurrent 是其中一種工具；固定 inference weights 仍可保留為方法範圍，但不是建立研究動機的唯一差別。

這更直接對齊原 slide 3：失敗影片可能揭露碰撞、支撐或物體運動，且其價值取決於模型和互動。不能把所有 empirical improvement 都稱為掌握了某種物理概念；要透過 evidence 内容的控制實驗建立支持。

## 2. 與 AdaJEPA：共享應用方向，不等於相同核心問題

| 比較層次 | AdaJEPA | 本研究的目前重心 |
|---|---|---|
| 最終用途 | 更可靠的 prediction / planning | 理解經驗，並探索其對 prediction / decisions 的用途 |
| 主要問題 | 如何利用線上 transitions 更新模型，使部署規劃更有效 | 哪些 attempts、哪些 observation 內容能提供可轉移的資訊 |
| 方法重點 | MPC 中的 gradient-based parameter adaptation | Controlled evidence extraction、累積與 transfer study；可使用 learned memory |
| 關鍵實驗變因 | Adaptation targets、steps、learning rate、buffer 等 | Evidence 內容、attempt 類型、歷史組合、query relevance；另控制 extraction mechanism |
| Decision improvement | 原文控制任務中的實驗結果 | 本研究要獨立測試的下游價值，不是預設前提 |

AdaJEPA 不只是一個 abstract prediction model：它在執行後的 transitions 上最小化 latent loss，並實驗評估 planning。其結果不構成「降低任意 prediction loss 必然改善任意 planning」的普遍定理，更不能直接推到 PHYRE 的 success ranking。[AdaJEPA §3–4](https://arxiv.org/html/2606.32026v1)

也不能說 AdaJEPA 完全不處理資訊選擇。原文有 recent-N / hard-N replay strategies，會選擇已取得的 transitions。這和「執行前選哪個 attempt 才能獲得有用資訊」不同；原文主規劃目標是 latent goal-reaching cost。[AdaJEPA §3.2](https://arxiv.org/html/2606.32026v1#S3.SS2)

正確定位：**AdaJEPA 是相關的 adaptation 方法與可用 baseline；本研究的問題範圍以經驗的資訊價值為中心。** 不必把主要貢獻限縮成用 recurrence 取代 gradient update、並且一定要贏它的成功率。

## 3. 三種「資訊」不能混在一起

1. **Observation 中包含什麼：** 成功／失敗 bit、接觸時間、碰撞後運動、未發生的接觸、支撐是否解除等。
2. **目前模型能提取什麼：** 在固定訓練資料和 context 預算下，encoder/updater 是否能利用上述內容影響未試 queries。
3. **提取後有什麼用途：** 其他 actions 的 outcome prediction、成功／失敗區分或排名有沒有改善。

第 2 層失敗不證明 observation 沒有資訊；第 3 層 prediction 改善也不保證 decision 改善。比較兩種 extraction mechanisms 有助辨識是 evidence 本身的限制，還是當前方法無法利用。

本任務不必預設 scene 具有未知隨機物理參數。Evidence 可以揭露有限模型對已知可見場景的預測錯誤。沒有識別性實驗時，不把 latent state 宣稱為真實 physical parameters 或 calibrated Bayesian belief。

## 4. 建議的研究問題順序

**RQ1 — What can be extracted?**

相同 attempt，依序提供 action-only、binary outcome、完整 outcome latents、prediction 加 observation、顯式 residual。以同一組未試 queries 評估增益。Prediction+observation 與 prediction+residual 資訊上可互相重建，因此效果差異是表示／學習效益，不能說 residual 憑空增加資訊。

可進一步遮蔽接觸前後或特定物體資訊，檢查相關 transfer 是否消失。遮蔽也會改變 input distribution，需提供適當對照或訓練時涵蓋，避免只把模型遇到異常輸入的退化當成機制證據。Attention heatmap 本身不足以支持因果解釋。

**RQ2 — Which attempts provide transferable evidence?**

固定 learner、更新預算、起始 context C 與 disjoint held-out query set Q，改變新加入的 probe attempt。比較有／無相關接觸、重複／互補、成功／失敗、大小 prediction discrepancy 等類型。這些是待測因素，不能預先認定「碰撞更多」或「residual 越大」一定更有用。

Probe pool 與 query pool 應分開，避免測某個 probe 時把它移出 Q，導致不同 probes 使用不同難度的 queries。不同 probe 的耗時／影片長度需記錄；context evidence 不能偷含 Q 的實際結果。

**RQ3 — How does evidence accumulate?**

同一固定 Q 下比較 C、C+A、C+B、C+A+B，檢查增量效益、冗餘、互補與干擾。以 experience bank 作資訊較少被跨次壓縮的參考，recurrent state 作 compact extraction/retention 方法。非線性 synergy 是要量測的效果，不保證存在。

**RQ4 — When does information help decisions?**

分開報 prediction utility 與 decision utility，並用原 framing 的 proposer/scorer/world-model 診斷定位落差。即使更準的 latent 沒改善 ranking，仍可能揭露 evidence 的適用範圍，但不能宣稱成功率提升。

## 5. 可操作的資訊效益定義

不要求一開始就估計 mutual information。可使用針對特定 learner 和 query distribution 的 **empirical transfer utility**。

令 Fₐ 為執行 probe a 後的 evidence，所有方法使用相同初始 context C 和固定 query set Q：

\[
\Delta_{\rm pred}(a\mid C,Q)
= L_Q(C)-L_Q(C\cup\{F_a\}),
\]

\[
\Delta_{\rm decision}(a\mid C,Q)
= J_Q(C\cup\{F_a\})-J_Q(C).
\]

L 是固定 Q 的 held-out outcome prediction loss；J 可以是同 pool 所選 action 的真實 success 或其他明確 ranking metric。兩個 delta 都可能為負，也可能一正一零。對多 scenes / seeds 取 paired estimates。

這些數字衡量 **當前模型可利用的資訊價值**，不是 action 的內在 Shannon information，也不構成模型無關的資訊上界。需要逐層 controls，才能從「有效」進一步解釋為「利用了哪種內容」。

失敗與資訊價值、surprise 與資訊價值、prediction utility 與 decision utility，都不應畫等號。

## 6. 「哪種 attempt 有用」與「主動選 informative attempt」的界線

**目前可完成的核心：** 執行／觀察後，在控制條件下分析哪些 attempts 提供有用 evidence。這是 retrospective characterization，不必先解 active exploration。

**若後續做主動取得資訊：** 執行前只使用 x、a、C，估計尚未知 outcome 的期望效益，選擇有助未來 queries 的 probe。不能用該 probe 的真實影片或 residual 去選它，再把結果稱為可部署的 information acquisition。若根據真實結果挑 context，只能報 oracle/probe-selection diagnostic。

當 attempts 同時可以解題與收集資訊，就要衡量立即成功與未來效益的取捨，並計入每次真實 interaction。這才進一步連到 active experimental design / dual-control 類問題。相關文獻需擴展，而不只比較 AdaJEPA。[Amortized Bayesian Experimental Design for Decision-Making, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c59f05d7ab3638b138cc61f32e1a7cd1-Abstract-Conference.html)

## 7. 實驗軸要隨研究問題改變

- **固定 evidence，比較 extractors：** learned context memory 與 AdaJEPA-style TTA 是否都能提取同一份 evidence 的價值？這仍然是公平且有用的 benchmark。
- **固定 extractor，比較 evidence：** 不同 attempts、影片片段、feedback 表示、歷史組合，有何增量效益？這是目前更應突出的一條主軸。
- **少量交叉驗證：** 若某類 evidence 只對其中一個 learner 有用，說明結論與機制有關。無須立刻跑大型全因子組合。

AdaJEPA 因而既是 competing adaptation baseline，也可以作為另一種 information consumer。如果兩種方法都受益於同種證據，能支持發現不只來自某個 recurrent architecture 的特性。

只比較成功率的 updater leaderboard，不足以回答「提取了什麼資訊」；同樣地，只畫幾張 memory attention 圖，也不足以回答哪些資訊促成跨 action transfer。

## 8. 可直接用於 slides / related work 的文字

**Problem slide：**

> We study what information can be extracted from previous physical attempts, which attempts provide evidence that transfers to untried actions, and when this information improves subsequent decisions.

**AdaJEPA comparison：**

> AdaJEPA studies how online parameter updates from executed transitions can improve planning. We focus on the information supplied by physical attempts: what can be extracted, which interactions provide transferable evidence, and how its value depends on the learner and the target actions. Prediction and decision gains are evaluated separately. We use AdaJEPA-style adaptation as an alternative mechanism for consuming the same evidence, rather than defining our contribution solely as a replacement for gradient-based adaptation.

**Method positioning：**

> Experience-conditioned latent models provide a test bed for studying information extraction and accumulation. Recurrent memory is one candidate representation, whose benefits and limitations are measured against explicit experience storage.

**Expected contribution：**

> A controlled study of the content, transfer, and downstream utility of execution feedback, together with a learned mechanism for extracting and accumulating useful evidence.

這些是問題與計畫，尚無效果宣稱。以資訊為核心的差異必須由對應實驗實現；若最後只比較 recurrent vs TTA 的總成功率，仍然主要是在研究 adaptation mechanism。
