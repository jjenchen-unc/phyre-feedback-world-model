> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Task-centric latent 最小實驗：準備稿

2026-10-01。使用者已授權此方向；執行順序是先完成
[閉環對照](ADAPTED_PROBE_CLOSED_LOOP_RUN.md)。本文件是準備稿，尚未實作、
凍結最終超參數或提交 job。不能當成已執行實驗的 registration。

## 最小問題

保持既有 world model、memory 架構與輸出 latent 不變，僅讓 memory 的
訓練目標更重視一個保留物件幾何的 compact 空間，是否比目前直接幾何監督
更好？先做 objective 介入，避免同時換掉動態模型、輸出空間與讀出方式。
這不是完整 TC-WM 重現，也不是直接證明 compact-space dynamics 更好。

## 分開兩個訓練階段

先從訓練資料的真實 latent 學一個小型線性投影與 embedding decoder，
以 reconstruction 保護輸入內容，搭配物件位置／bar 方向的 supervision。
物件描述只能取 query 初始狀態，未來幾何只當 target。
準備一個同容量、相同訓練樣本與預算，但不使用物理 supervision 的投影。

投影學好後完全凍結，再从相同 R0 初始化做 matched memory 續訓：

| 條件 | 要隔離的因素 |
|---|---|
| 現有純 latent L1 | 原始目標參考 |
| 現有 L1＋直接幾何 loss | 已完成的可行介入參考 |
| L1＋無物理監督投影空間誤差 | 單純壓縮／重新加權是否已足夠 |
| L1＋物理對齊投影空間誤差 | 物理 alignment 是否帶來額外效果 |

若要重用前兩組 checkpoint，必須確認續訓資料、抽樣、步數、optimizer、
初始化與數值路徑完全相同；否則重跑匹配控制，不能默默沿用不等預算結果。

## 凍結實作之前必須處理的歸因問題

- 投影可以任意縮小尺度，讓 projected loss 看似降低；須用訓練資料固定
  normalization，記錄 variance／有效維度／reconstruction，阻止把 collapse
  當成更好的 prediction。不跨不同空間直接比較 loss 數值。
- 幾何 head 可能僅用初始物件位置推測；用 heldout 的真實未來 latent 對
  初始 latent 替換檢查投影是否保留額外未來資訊，不把 head 的訓練 loss
  當足夠證據。查物件位置、方向、相對距離，而不是只看 pooled scene label。
- 提供 supervision 的投影 head 不作唯一評估 head。仍用目前固定 original
  與三個 adapted probes，在共同原 latent 上評估幾何、ranking。
- 新 probe、base、memory 的優劣比較始終使用相同 readout。成功標準包含
  與 base＋適配 probe 的比較，不能只報相對舊 memory 的改善。
- 00007 已顯示幾何與 ranking 可分離，必須保留逐模板結果，不能只看總平均。
- 同一 D0 持續用於機制探索。若方向有效，後續確認性場景／模板需另行
  事前鎖定，不能因為本輪贏就把 D0 重新命名成 test。

## 接續動作

閉環 34282／34283 完成後，先核對實際收益、第一次失敗修正方向，以及
memory 對相同 head 的 base 差距。再固定投影維度、loss normalization／
係數、訓練預算與先導檢查門檻，保存最終 registration，完成實作和 smoke。
這些步驟都在既有授權內；此處保留的是必要的實驗設計工作，不是待使用者
重新同意的 approval 流程。

## 閉環完成後的新證據（2026-10-02）

[損失歸因](MEMORY_CLOSED_LOOP_LOSS_ANALYSIS_20261002.md) 顯示，多數丟失題目
在第一次修正已超出預算；保留前十次策略、後續回 base 的事後重播能恢復
本批資料的 base 覆蓋率，AUCCESS 也更高。這提示先確認 fallback／gating
控制與新場景表現，再擴大 representation 改動。原 task-centric 最小實驗
仍在授權內，但需要保留該控制與第一次修正指標，不以 latent loss 改善
取代 support 適用性驗證。上述 counterfactual 是事後分析，不當作新測試。
