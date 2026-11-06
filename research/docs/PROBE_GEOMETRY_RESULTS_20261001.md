> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Probe 適配與 memory 幾何監督：已完成結果

2026-10-01。兩輪訓練、離線評估與原始陣列核對皆已完成。最後查詢 Slurm
沒有執行或排隊中的工作。尚無新閉環 AUCCESS，也沒有啟動 task-centric latent。

## 最重要的結果

Probe 對預測 latent 的讀出確實有可改善的地方，但只修 probe 沒有讓 memory
追上 base。Memory 訓練加入幾何監督，比同預算的純 L1 續訓更好；此改善能
轉移到沒有提供本輪訓練梯度的適配 heads，仍未超過同 head 的 base。

以下是同一批 cross 候選與 K1 失敗 context 的 ranking AUC。100 場景中有
73 個非空 K1，其中 57 個有成功候選，可以計算 AUC 與成功候選幾何誤差。
各指標先在場景內平均，再對場景等權平均。

| 預測模型 | 原 probe AUC | 適配 probe AUC |
|---|---:|---:|
| Base | **0.5654** | **0.7101** |
| 原始 memory R0 | 0.5052 | 0.6288 |
| Memory 純 L1 續訓 | 0.5143 | 0.6413 |
| Memory L1＋幾何監督 | 0.5560 | 0.6767 |

適配 probe 指在 base/R0 預測 latent 上等比例訓練的 predicted_mix。
Base、R0 的適配欄平均三個 probe seeds；後兩列原 probe 欄平均三個 memory
seeds，適配欄平均全部 3×3 memory/probe 組合，沒有挑最好 checkpoint。
這些組合共享 base、R0 與原 probe，不能當九次獨立完整訓練。

## 1. Probe 適配修好了什麼？

固定所有 base/R0 latent，只改 probe：

- Base AUC：0.5654 → 0.7101，平均增加 0.1447。
- R0 AUC：0.5052 → 0.6288，平均增加 0.1237。
- 同 query、同監督與同預算，但只用真實 latent 續訓的 true_control，
  base/R0 AUC 分別為 0.5617／0.5038，沒有同樣改善。
- R0 成功候選的表面距離 MAE：41.23 → 38.66 px。Base：32.42 → 32.13 px。
- 所有 probe 間的 latent L1 完全不變，因為 predictor/memory 沒有更新。

因此，有證據支持「原 probe 對模型產生的 latent 讀取不夠好」，而不只是
多訓練幾步就有效。但這不是 memory 特有問題：base 改善更大；memory
相對 base 的 AUC 差距，從 -0.0602 擴大到 -0.0812。不能說適配已消除
memory 的缺點，也不能唯一歸因為 memory 導致 latent distribution shift。

還要保留初始畫面對照的限制。只用初始 latent 訓練的 head，在相同初始
輸入上的平均 AUC 約 0.4243，低於 mixed head 讀 R0 的 0.6288；但將 mixed
head 的輸入直接替換為初始 latent，其 AUC 約 0.5860，且 seed 0 與 memory
輸入幾乎持平。這個輸入替換本身有分布改變，不能單獨證明捷徑；同樣也
不能只憑適配後 AUC 就宣稱模型已恢復正確的時序物理資訊。

## 2. Memory 的幾何監督有幫助嗎？

這輪兩組從同一 R0 出發、相同 K1/support/query、相同步數與 optimizer，
唯一介入是增加 0.1 倍的幾何 loss。Base 和提供梯度的原 probe 都固定。
幾何監督是位置與 bar 方向，不直接監督接觸距離、持續時間或成功標籤。

| 指標，三個 memory seeds 平均 | 純 L1 | L1＋幾何 |
|---|---:|---:|
| 原 probe AUC ↑ | 0.5143 | **0.5560** |
| 原 probe 成功候選 surface MAE ↓ | 40.17 px | **34.02 px** |
| 三個適配 heads 平均 AUC ↑ | 0.6413 | **0.6767** |
| 三個適配 heads 平均成功候選 surface MAE ↓ | 37.75 px | **34.56 px** |
| 整體 latent L1（73 場景）↓ | **0.114315** | 0.114966 |
| 成功候選 latent L1（57 場景）↓ | **0.131544** | 0.132279 |

三個 memory seeds 在原 probe 下都改善 AUC 與成功幾何；全部九個適配
head/memory 組合的場景平均也都改善。29 場景的 calibration 面板中，
28 個可算 AUC／成功誤差，改善方向也一致。這降低了「改善只存在於提供
梯度的原 head」的疑慮，但這些 heads 仍共享原始模型與架構，並非完全
獨立的物理驗證。

這輪還提供了與前一輪相反方向的介入證據：**latent L1 略變差，幾何與
ranking 反而改善。** 因此平均 latent L1 不適合作為唯一的訓練選擇指標。
這仍不是說 L1 完全沒用，也不是證明幾何誤差下降必然帶來更好的 ranking。

## 3. 模板差異不能忽略

幾何 loss 相對純 L1，在原 probe 下的 AUC 變化：

| Cross 模板 | 三個 memory seeds 的範圍 |
|---|---:|
| 00000 | +0.0047 ～ +0.0132 |
| 00002 | +0.0469 ～ +0.0578 |
| 00007 | **-0.0314 ～ -0.0155** |
| 00014 | +0.1106 ～ +0.1182 |

00007 在原 head 下成功幾何 MAE 降低約 10～14 px，但 AUC 仍退步；改用
適配 heads 後，該模板的 AUC 變化又全為正。表示 loss 與 readout 有交互
作用，不能把整體平均提升說成所有模板都改善，也不能當成已完成因果拆解。
Probe 適配本身的提升也不均勻，00007 改善最大，00002 的 R0 AUC 接近持平。

## 4. 目前能支持的結論與下一步

1. 預測 latent 的 readout 是可修復的限制之一，base 與 memory 都受影響。
2. 同預算下，給 memory 明確的幾何訓練目標比只追平均 L1 更有用；三個
   微調 seeds 的整體效果一致，且可轉移到其他固定 readout。
3. Memory 尚未贏過同 readout 的 base。不能用「新 memory＋新 probe」
   贏過「舊 base＋舊 probe」來宣稱 memory 有額外價值。
4. 幾何改善不保證每個模板的 ranking 改善。所有結果仍是四模板、重複
   使用的 D0 開發面板；沒有新的確認性跨模板或閉環解題證據。

事前記錄的 task-centric latent 探索門檻通過，而且兩面板都是 3/3 seeds
通過。下一步已獲授權：設計 compact physical alignment＋reconstruction
的受控 pilot，保留公平的 base、純 L1 與幾何 loss 參考。這個門檻只有方向
一致性，沒有顯著性或最小實用效果保證；架構實驗尚未實作／排隊。

## 核對範圍與原始紀錄

- Probe adaptation：129 場景，遠端核對 5,160 組原始條件陣列。
- 幾何 loss：6 模型×129 場景，遠端核對 3,096 組原始條件陣列。
- 本機独立從 903 份場景 summary 重算 1,704 個 macro 指標，核對 paired
  sampling parity；943 個下載 JSON 的 SHA256 與共享伺服器一致。
- 大型 NPZ 留在共享伺服器；本機沒有重新下載全部 NPZ。核對紀錄：
  `results/PROBE_GEOMETRY_VERIFICATION_20261001.json`。
- Probe run：`results/probe_adaptation_cross_20261001/{SUMMARY.json,RESULTS.md}`。
- Geometry run：`results/memory_geometry_followup_20261001/{SUMMARY.json,RESULTS.md}`。
- 設定：[PROBE_ADAPTATION_RUN.md](PROBE_ADAPTATION_RUN.md)、
  [MEMORY_GEOMETRY_FOLLOWUP_RUN.md](MEMORY_GEOMETRY_FOLLOWUP_RUN.md)。
