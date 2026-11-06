> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Fixed-support cross-memory diagnostic

Registered and launched 2026-09-28; Slurm array 33762, two one-GPU tasks on
oprime.ib. This is a targeted diagnosis of known development failures, not a
held-out estimate of method performance. No training or simulator rollout.

## Frozen design

- Tasks: 00014:273 and 00014:407, selected in the prior failure analysis.
- Existing cross-f1 base, original memory, gated memory, and frozen probe.
- All 1024 saved shortlist candidates per task; use true query outcomes only
  for offline diagnostics. The model sees initial query latents and actions.
- First reproduce historical base first-success attempts (3 and 5).
- Support: base policy's failed prefix before success, capped at 4; use K=1
  and the full available prefix (2 and 4, respectively).
- At each K, remove exactly the same support candidates from every condition's
  reported query set. All conditions use the same support and query actions;
  this is a fixed-support ranking measurement, not a new closed-loop S@100.
- Conditions: base, observed latents, original empty, gated empty, original
  full context, gated full, gated scale=.25, no residual, no video, neither
  video nor residual (action/outcome retained), and coherent wrong-scene
  context from 00014:009 / 00000:000. Donors are the first K failed candidates
  in their saved shortlist order, chosen for offline intervention only.
- Zeroing is applied after computing the true support residual. Therefore
  no-video retains actual residuals; action/outcome-only zeros both channels.
  These are out-of-distribution interventions on the existing residual model,
  not newly trained action-only baselines. Wrong-scene replacement changes
  action, video, residual, and outcome coherently; it is not a clean isolation
  of video relevance alone.
- Readout and batch size match closed_loop_memory.py. Assert gated-empty scores
  match base scores before interpreting results.
- Save per-candidate scores, latent L1 by time, green trajectories/errors,
  surface-distance errors, labels/action IDs, support metadata and checkpoint
  hashes. Exclude initial time from future prediction-error summaries.

## Interpretation

Compare base/full on common queries: latent error, geometric/contact error,
success ranks, and errors specifically on successful candidates. A lower
overall latent error with worse success ranking is a prediction–decision gap;
larger latent and geometry errors indicate a prediction failure as well.
Observed-latent readout diagnoses probe limitations on this exact shortlist.
Wrong-scene and zero controls diagnose dependence, with the intervention
limitations above. Their ranking alone cannot establish that a trained
action-only system would suffice.

Implementation: [diagnose_cross_memory.py](../pipeline/diagnose_cross_memory.py).
## Completed results

兩個 Slurm tasks 均 COMPLETED / exit 0，耗時 4m41s / 4m46s。Base 的第 3/5
次成功均重現，gated-empty 分數與 base 相同。所有 candidate arrays、summary
已下載並以 SHA256 核對遠端版本；另從 arrays 重算排名、AUC、latent 平均值。

數據：[完整表格](../results/cross_memory_diagnostic_20260928/METRICS.md)、
[逐類別偏移](../results/cross_memory_diagnostic_20260928/class_conditional_shifts.json)、
[圖 PNG](../results/cross_memory_diagnostic_20260928/diagnostic.png)、
[圖 PDF](../results/cross_memory_diagnostic_20260928/diagnostic.pdf)。

### 一次失敗就把成功候選排到後面

以下是排除已觀察的第一個失敗後，剩餘 1023 candidates 的第一個成功排名。
Full feedback 指 gated checkpoint；不是額外閉環跑出的 attempts。

| 條件 | 00014:273 | 00014:407 |
|---|---:|---:|
| Base / gated-empty | 2 | 4 |
| Original empty | 288 | 311 |
| Original full feedback | 367 | 182 |
| Gated full feedback | 350 | 141 |
| Gated scale 0.25 | 1 | 15 |
| Video/residual 歸零，保留 action/outcome | 13 | 72 |
| Residual 歸零，保留 video | 312 | 147 |
| Video 歸零，保留 residual | 280 | 270 |
| 同模板、其他場景的完整失敗 context | 340 | 133 |
| 不同模板、其他場景的完整失敗 context | 317 | 83 |
| 真實影片 latent 經相同 probe/scorer | 1 | 1 |

更多失敗沒有修復：273 的 K=2，gated full 的排名 367（base 1）；407 的
K=4，排名 152（base 1）。這裡 support 仍固定為 base 的失敗前綴，不是各
memory 策略自行選出的 context。

### 平均 loss 改善，稀少成功例卻受損

| K=1、base → gated full | 00014:273 | 00014:407 |
|---|---:|---:|
| 全部 query 平均 latent L1 | 0.11663 → 0.11468 | 0.11902 → 0.11738 |
| 成功 query 平均 latent L1 | 0.13109 → 0.13900 | 0.13032 → 0.13388 |
| 成功 query 綠球誤差中位數 px | 77.39 → 109.30 | 119.28 → 133.92 |
| 同題 success/failure 排序 AUC | 0.7827 → 0.2275 | 0.7049 → 0.2905 |

成功候選分別只有 13/1023、20/1023。Memory 改善約 83% 的失敗候選 latent
L1，但讓 13/13、19/20 個成功候選的 L1 變差。平均 loss 因大量失敗例的改善
而降低，無法反映稀少解的預測受損。這不是只有 score 校準數值改變，實際
相對排序與成功候選的表示誤差都變差。

### 修正方向與物理含義

兩題第一次觀察到的失敗，綠球最終在右側 x≈242 / 227px；成功 actions 的
真實終點平均在左側 x≈17.5 / 16.3px。加入這次失敗後，所有成功候選的預測
終點都往右移，平均 +31.27 / +28.87px，遠離真實成功結果。這支持「失敗
方向的修正被錯套到成功 action」的局部機制。

但 base 自己也沒有準確預測成功軌跡：成功候選平均最終 x≈146 / 144px，
memory 後為 ≈178 / 172px。不能把結果寫成「原本精確的 physics 被破壞」；
較準確的是原本仍有用的相對 ranking 被進一步破壞。

在完全相同 shortlist 上，把真正 observed latents 交給相同 probe，兩題
AUC 均為 1.0，成功候選 green median error 為 1.30 / 0.99px。這排除了
probe 在這兩題真實影片上普遍讀不出成功的解釋；仍不能排除 probe 對
predicted/adapted latent 的分布失配。

### 消融支持的解釋與界線

- 移除 residual 後仍嚴重退步，故不支持「問題完全來自 residual 通道」；
  actual-video 通道加上 learned adapter 也足以呈現問題。
- 錯場景 context 的排名與正確 context 相近或更好；這組 diagnostic
  未呈現可靠的正確經驗選擇性。此控制同時改變 action 與影片，不能單獨
  歸因於影片語意或場景識別。
- 歸零兩個視覺通道明顯減輕退步，但仍落後 base；這不證明 trained
  action/outcome-only 模型足夠，因為歸零屬於既有 checkpoint 的 OOD 介入。
- 降低 scale 可減少損失，但 407 仍由第 4 名退到第 15 名，沒有通用修復。

本次最強的結論是：在兩個事先選定的 cross 失敗案例，memory 降低平均
預測 loss 的同時損害稀少成功 actions，且 feedback 的幾何修正方向不適合
這些成功 actions。尚未以訓練消融證明「failure-only context 分布」是原因，
也沒有證明這個機制涵蓋所有 cross templates。

下一步應先在更廣的未用場景驗證這種 success-conditional error，再比較
包含 policy-selected failure contexts 的訓練與 decision-aware objective；
目前沒有證據支持直接加大 memory 容量。
