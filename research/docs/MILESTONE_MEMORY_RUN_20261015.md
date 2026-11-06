> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Milestone: matched Q0 後的 multi-turn update

本輪是小規模效果驗證。目的是回答：從公平重訓得到的 Q0 出發，實際失敗 action 的資訊能否讓後續選擇更好？使用 oprime GPU 0–5，共六張。

| 設定 | Cross | Within |
|---|---|---|
| 官方 fold | 1 | 0 |
| 初始 Q0 | scratch、完整 8 epochs、λ=0、seed 0 | scratch、完整 8 epochs、λ=0.03、seed 0 |
| 場景 | 原有 100 個 testing scenes | 原有 100 個 testing scenes |
| 候選與嘗試數 | 原有共同 1,024 action shortlist；最多 9 次 | 同左 |
| AdaJEPA-style | lr=0.001，每次 1 gradient step | lr=0.001，每次 3 gradient steps |

各比較五種方法：相同 λ 的靜態 base、固定 Q0、real-residual updater、zero-residual updater、Q0 + AdaJEPA-style。Base/Q0 重新計算分數並與已存結果逐值核對。所有動態方法的第一輪 1,024 個分數必須與固定 Q0 相同。

## Updater 訓練

共四組新訓練（兩個 protocol × real/zero），每組 seed 0、10,000 slots、每 slot 兩個 episodes。保留舊版配方，以便看出更換初始 Q0 的效果：

- 固定 predictor 和 initial tokens，只訓練新的 updater；real/zero 使用相同初始化和 episode draws。
- Scorer 在當下 tokens 下選 support action；遇成功停止。只把已選且失敗的 action 當作 update 資訊，不預先篩掉成功候選。
- Residual 是觀測減去「當下 predictor + tokens」的預測，每次更新都重新計算。
- 每 episode 的 support cap 從 1、2、4、8 均勻抽樣；query 與 support 的候選組分開。
- **只對最後一次更新後的 query 預測計算 loss：future latent L1 + 0.1 × geometry loss。** 初始模型的 λ 和 updater 的 0.1 是兩個不同階段的設定。
- Full BPTT，沿用原版逐 block activation checkpointing。AdamW lr=3e-4、weight decay=0.05、clip=1；OneCycle 3% warmup。沒有 support 的 slots 仍計入完整排程。

Zero updater 保留 action/history，只把 residual 設為零；它獨立訓練，不能拿 real 訓練出的權重臨時歸零來代替。

AdaJEPA-style 不做離線訓練或新的參數搜尋。沿用先前各 protocol 選出的設定，只更新 predictor 最後一個 block 與輸出 LayerNorm；Q0 initial tokens 固定。每次用最近五個已選失敗 actions 的 future latent L1，重建 Adam optimizer，模型權重在同一場景內累積更新、換場景時完全重設。這是我們在 PHYRE 上的實作，不代表原論文所有設定。

## 驗證與排程

正式開始前，各 protocol 都比較原版／加速版九個連續訓練 slots（其中一個包含兩條八次更新的壓力測試）：動作、loss、梯度及更新後權重一致；檢查 predictor/probe 固定、zero 輸入不變性、完整候選第一輪分數，以及 TTA 重設。嚴格數值模式只用在一致性測試；正式執行沿用歷史 CUDA 設定。

四張卡先訓練，兩張先評估；訓練完成後自動接續 updater 評估。每個評估方法拆成四個互斥的 25-scene shards。保存每一輪完整分數、action、更新紀錄與 tokens。最後自動核對原始結果並產生 `SUMMARY.json` 與 `RESULTS.md`。

主要呈現 S@1/3/5/9、相對 Q0 多解／少解哪些題，以及第一次失敗更新前後在相同剩餘候選上的 ranking。六張卡不是六組訓練；只有四組訓練，其餘容量加速評估。

執行紀錄位於 `results/milestone_memory_20261015/`：`PLAN.json` 固定設定、`code/` 固定程式、`STATUS.json` 即時排程、`attempts/job_*/` 日誌。每 500 slots 可恢復；每 1,000 slots 保存實測速度、support 深度與剩餘時間估計。舊的大型多 fold 計畫不在本輪執行範圍內。
