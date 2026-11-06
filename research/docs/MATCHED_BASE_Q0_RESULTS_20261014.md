> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。

# Base 與 Q0 相同 loss 訓練結果與下一步實驗

紀錄日期：2026-10-14。供下週組會與 teammate 閱讀。這輪 24 組訓練、24 組評估與彙整均已完成。

## 報告時先說的重點

這輪把 base 與 Q0 的訓練起點、資料量、loss 和訓練步數對齊，重新確認 initial tokens 本身有沒有幫助。結果是 **Q0 有局部改善，但沒有全面勝過 base；幾何監督的權重很重要，增加權重也不保證改善成功率**。

- 在 λ＝0.1 的三個 seeds 中，Q0 的平均 S@9 在 Cross 從 9.7% 到 11.3%，Within 從 46.7% 到 50.0%。但 Cross 的改善不一致；Within 的 S@1 與 S@20 反而較低。
- 在單 seed 的權重比較中，Within 的 λ＝0.03 對 base 和 Q0 都比 λ＝0 好。Cross Q0 則是 λ＝0 的 S@9 與 ranking AUC 最高，λ＝0.03 的 S@20 最高。
- 下一輪 Cross multi-turn 建議從 Q0、λ＝0、seed 0 開始。這是起始模型的選擇，不能推論 updater 的幾何 loss 權重也應該是 0。本輪沒有訓練或評估 residual updater。

## 背景與名詞

World model 接收場景和 action，預測未來的 latent，也就是 encoder 產生的畫面特徵。Scorer 再根據預測給候選 actions 排序，系統優先嘗試分數較好的 action。

**Base** 是沒有 memory prefix tokens 的 predictor。**Q0** 在同一個 self-attention transformer 中加入 32 個可學習的 initial tokens，並與 predictor 一起訓練。評估時這些 tokens 是固定的，沒有先前 trial 的觀測資訊。因此，Q0 可視為未讀取任何 support 前的 memory 模型起點。

後續的 **support** 是實際嘗試過、已有觀測結果的 action；**query** 是目前要預測的 action。Multi-turn 的預期流程是：依當下排名選 action，得到觀測，計算當下 predictor（含目前 tokens）的預測誤差 residual，更新 tokens，再重新預測與排序。這段流程尚未在本輪新 checkpoint 上實驗。

這裡的 base 都指本輪重新訓練的 base，不是歷史的 pretrained baseline。先前的 Q0 比較曾同時改變訓練起點、額外訓練量與幾何監督；本輪先把這些條件對齊。

## 實驗設定

| 項目 | 本輪設定 |
|---|---|
| Protocol | 官方 Cross fold 1、Within fold 0 |
| 初始化 | Base 和 Q0 的 predictor 都從隨機權重開始；同 seed 的 backbone 初始權重相同 |
| Q0 可學參數 | Predictor 與 32 個 initial tokens；updater 不參與訓練 |
| 固定元件 | Encoder、PCA、geometry probe、評估 scorer |
| Loss | Latent L1 ＋ λ × geometry loss；base 與 Q0 使用同一個函數 |
| λ | 0、0.03、0.1、0.3 |
| Seeds | λ＝0.1 使用 0／1／2；其餘 λ 只有 seed 0 |
| 訓練預算 | 全部訓練資料、8 epochs、batch 8；同 seed 使用相同資料順序 |
| Optimizer | AdamW，lr 3e-4，weight decay 0.05，gradient clip 1，OneCycle 3% warmup |
| Cross 資料 | 52,276 條軌跡，52,272 次 optimizer updates |
| Within 資料 | 52,349 條軌跡，52,344 次 optimizer updates |
| Checkpoint | 固定最後一個 epoch，不挑效果最好的 epoch |
| 排名評估 | 每個 protocol 固定 100 個場景，每場景相同的 1,024 個候選 actions 與實際模擬 labels |
| Scorer | Cross 使用 predicted_mix_s0，Within 使用原 scorer；同 protocol 全部模型一致 |
| 誤差評估 | 每個 protocol 另外固定 400 條 dev 軌跡 |

Geometry loss 經凍結 probe 讀出動態物體的位置與長條方向，對照真實幾何；每條軌跡取 8 個未來時間點。報告中的幾何誤差是這個未乘 λ 的 loss，不是像素距離，也不是物理接觸誤差。

配對檢查已確認 base／Q0 的初始 backbone 與訓練抽樣相同。24 組評估都通過原始結果核對。正式訓練與評估使用凍結程式；新增的加速 scorer 沒有套入本輪。

## 指標怎麼讀

- **S@k**：前 k 次嘗試內成功的場景比例，越高越好。100 個場景中多成功一題就是多 1 個百分點。主要指標是 S@9，並保留 S@1、S@20 判斷不同預算下的表現。
- **Ranking AUC**：每個場景內衡量成功 actions 能否排在失敗 actions 前面，再對可計算的場景取平均。它不是 AUCCESS。Cross 有 75 個、Within 有 95 個同時含成功與失敗候選的場景可計算；S@k 的分母仍是全部 100 題。
- **Latent L1 與幾何誤差**：越低越好，取最後 epoch 在固定 400 條 dev 軌跡上的結果。這些軌跡與 ranking 候選池不同，所以不能直接當成同一個 action 的誤差與排名關係。

模型固定且不在線更新，因此本輪以一次固定排序計算不同嘗試預算的結果；它不是 multi-turn 在線更新評估。

## 三個 seeds 的主要比較

以下固定 λ＝0.1。表中的 ± 是三個 seeds 的樣本標準差，不是信賴區間。三個 seeds 使用同一批 100 題，不代表 300 個獨立場景。

| Protocol | 模型 | S@1 % | S@9 % | S@20 % | Ranking AUC |
|---|---|---:|---:|---:|---:|
| Cross | Base | 0.3 ± 0.6 | 9.7 ± 2.5 | 18.3 ± 3.8 | 0.6310 ± 0.0462 |
| Cross | Q0 | 0.7 ± 0.6 | 11.3 ± 2.1 | 20.3 ± 3.1 | 0.6114 ± 0.0249 |
| Within | Base | 24.0 ± 2.0 | 46.7 ± 1.2 | 61.0 ± 1.7 | 0.8717 ± 0.0008 |
| Within | Q0 | 18.7 ± 5.7 | 50.0 ± 3.5 | 59.3 ± 2.1 | 0.8802 ± 0.0021 |

Cross 的 Q0 相對 base，三個 seeds 的 S@9 差距分別是 −1、＋6、0 個百分點，S@20 是 −3、＋7、＋2。平均成功率小幅提高，但不是每個 seed 都改善，而且平均 AUC 較低。

Within 的 S@9 差距是 0、＋2、＋8 個百分點，但 S@1 是 −5、−10、−1，S@20 是 −2、−3、0。Q0 的優勢依嘗試預算而變，不能只選 S@9 就稱它全面勝出。

## 權重比較與全部逐次結果

λ＝0.1 有三個 seeds，其他權重只有 seed 0。比較權重時應先對照各權重的 seed 0；其餘 seeds 用來觀察 λ＝0.1 的訓練變異。下表保留所有 24 組，避免只呈現最佳結果。

### Cross 結果

| 模型 | λ | Seed | S@1 % | S@9 % | S@20 % | AUC | Latent L1 | 幾何誤差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | 2 | 17 | 26 | 0.6724 | 0.1190 | 1.1130 |
| Q0 | 0 | 0 | 1 | 19 | 25 | 0.7245 | 0.1191 | 1.0408 |
| Base | 0.03 | 0 | 3 | 18 | 25 | 0.6740 | 0.1242 | 0.9754 |
| Q0 | 0.03 | 0 | 4 | 18 | 29 | 0.6117 | 0.1263 | 1.0009 |
| Base | 0.1 | 0 | 0 | 10 | 20 | 0.6832 | 0.1355 | 0.9936 |
| Q0 | 0.1 | 0 | 1 | 9 | 17 | 0.5865 | 0.1332 | 1.0300 |
| Base | 0.1 | 1 | 1 | 7 | 14 | 0.5951 | 0.1381 | 0.9938 |
| Q0 | 0.1 | 1 | 1 | 13 | 21 | 0.6113 | 0.1367 | 1.1080 |
| Base | 0.1 | 2 | 0 | 12 | 21 | 0.6148 | 0.1393 | 0.9467 |
| Q0 | 0.1 | 2 | 0 | 12 | 23 | 0.6363 | 0.1384 | 1.0252 |
| Base | 0.3 | 0 | 2 | 18 | 26 | 0.6407 | 0.1576 | 1.1418 |
| Q0 | 0.3 | 0 | 1 | 7 | 19 | 0.5853 | 0.1531 | 1.0886 |

### Within 結果

| 模型 | λ | Seed | S@1 % | S@9 % | S@20 % | AUC | Latent L1 | 幾何誤差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | 16 | 44 | 59 | 0.8330 | 0.0874 | 0.4201 |
| Q0 | 0 | 0 | 18 | 41 | 56 | 0.8470 | 0.0870 | 0.4194 |
| Base | 0.03 | 0 | 23 | 50 | 64 | 0.8890 | 0.0966 | 0.3247 |
| Q0 | 0.03 | 0 | 20 | 54 | 62 | 0.8851 | 0.0963 | 0.3363 |
| Base | 0.1 | 0 | 22 | 48 | 63 | 0.8725 | 0.1135 | 0.3675 |
| Q0 | 0.1 | 0 | 17 | 48 | 61 | 0.8816 | 0.1123 | 0.3701 |
| Base | 0.1 | 1 | 24 | 46 | 60 | 0.8709 | 0.1124 | 0.3617 |
| Q0 | 0.1 | 1 | 14 | 48 | 57 | 0.8778 | 0.1128 | 0.3712 |
| Base | 0.1 | 2 | 26 | 46 | 60 | 0.8718 | 0.1114 | 0.3585 |
| Q0 | 0.1 | 2 | 25 | 54 | 60 | 0.8810 | 0.1125 | 0.3663 |
| Base | 0.3 | 0 | 15 | 43 | 58 | 0.8546 | 0.1398 | 0.3945 |
| Q0 | 0.3 | 0 | 14 | 41 | 56 | 0.8587 | 0.1441 | 0.4119 |

## 結果能支持什麼

### Q0 有局部改善但還沒有全面超過 base

加入 initial tokens 不會自動讓 ranking 變好。λ＝0.1 的三 seed 結果顯示，Within 的 S@9 與 AUC 較好，但最前面的 action 與 20 次預算下的成功率較差。Cross 的平均 S@9 較好，但 seed 間方向不一致。這些結果還不能推論 residual 更新會有同樣效果。

### 幾何監督權重的影響依 protocol 與架構而變

Within、seed 0 的 λ＝0.03 比 λ＝0 更好：base 的 S@9 從 44% 到 50%，Q0 從 41% 到 54%；兩者的幾何誤差與 AUC 也改善。λ＝0.3 時這些成功率收益消失，因此現有結果不支持越大越好的權重設定。

Cross Q0、seed 0 的 λ＝0 在 S@9 與 AUC 最好，λ＝0.03 在 S@20 最好；λ＝0.1 與 0.3 的成功率較低。Base 的變化並不單調，例如 λ＝0.3 的 S@9 回到 18%。不能把 Q0 的反應概括成所有 Cross predictors 的共同規律。

這些權重只有 seed 0 的完整比較，尚未確立最佳 λ，也沒有做顯著性檢定。

### 更低的 latent L1 不是更高成功率的必要條件

Within、seed 0 從 λ＝0 改為 0.03，base 的 latent L1 從 0.0874 升到 0.0966，但幾何誤差從 0.4201 降到 0.3247，S@9 從 44% 升到 50%。Q0 也出現同方向的變化：L1 從 0.0870 升到 0.0963，幾何誤差從 0.4194 降到 0.3363，S@9 從 41% 升到 54%。

因此，整體 latent L1 較低不保證排名更好。這符合「成功判斷只依賴部分預測資訊」的可能解釋，但本輪沒有直接分離各種誤差的因果作用。

反過來，在 λ＝0.1 的三個 seeds 中，Q0 的 dev 幾何誤差都比對應 base 高，但平均 S@9 較高。因此也不能用單一平均幾何誤差完整解釋 ranking。400 條 dev 軌跡與排名池不同，是判讀這個現象的重要限制。

### AUC 和有限嘗試次數衡量不同部分的排序

Cross Q0、seed 0 的 λ＝0.03 有比 λ＝0 更高的 S@20（29% 對 25%），但 AUC 更低（0.6117 對 0.7245）。Within 的三 seed 平均也呈現 Q0 AUC 較高、S@1 卻較低。AUC 評估較廣泛的成功／失敗候選相對排序，S@k 更重視前 k 名是否包含至少一個成功 action，兩者不必同方向。

## 下一步 Cross multi turn 的起始設定

建議起點是 **Cross Q0、預訓練 λ＝0、seed 0**。它在本輪 Cross Q0 的 seed 0 比較中有最高 S@9（19%）與 AUC（0.7245），符合目前主要看 S@9 的目標。λ＝0.03 可作為後續第二個候選，尤其在目標是 S@20 時。

這個選擇依據靜態 ranking；尚未證明它比較容易學好 updater。Updater 的 loss、幾何權重、哪些參數可訓練，以及更新輪數仍需另行確定。本輪只完成結果紀錄，沒有提交新的 multi-turn 訓練。

建議驗證時從同一個 Q0 checkpoint 比較三種行為：不更新、用真實 residual 更新、把 residual 設為零再經過相同更新流程。Support 必須來自當下 scorer 實際選擇的 action，residual 使用包含目前 tokens 的 predictor 預測。取得 support 的每次嘗試都算入總 action 預算，才能把收益歸因到觀測資訊，而不是免費額外試了更多 actions。

零 residual 對照仍可能透過 updater 的 bias 或其他輸入改變 tokens；不能把它稱為「完全沒有更新」。它是檢查 residual 資訊是否有額外貢獻的對照。

## 報告可以採用的順序

1. 說明以前的比較同時改變多個訓練條件，所以這輪先做同起點、同 loss、同預算的 base／Q0 比較。
2. 用 λ＝0.1 三 seed 表說明 Q0 的收益取決於指標，並展示 seed 差距。
3. 用 seed 0 的權重表說明 Within 小幾何權重有收益、Cross Q0 則偏好較低權重。
4. 用 Within 的誤差例子說明 latent L1 不足以選模型，同時保留 ranking pool 與 dev 軌跡不同的限制。
5. 提出 Cross Q0 λ＝0 作為下一輪起點，下一個問題是「真實 residual 的多輪更新是否優於不更新與零 residual」。

## 資料與重現位置

- [原始實驗設定](MATCHED_BASE_Q0_WEIGHT_STUDY.md)
- [完整彙整 JSON](../results/matched_base_q0_20261012/SUMMARY.json)
- [全部 24 組的 CSV](../results/matched_base_q0_20261012/REPORT_METRICS.csv)，另含 S@3、S@5、訓練步數與實測訓練時間。
- 遠端專案：`oprime:/path/to/phyre-unc`。
- 起始 checkpoint：`results/matched_base_q0_20261012/models/cross/q0_g0_s0/model.pt`，模型檔留在遠端。
- 凍結訓練與評估程式：`results/matched_base_q0_20261012/code/`，排程擴充紀錄：同結果目錄的 `schedule_expansion_20261014.json`。
- `SUMMARY.json` 已確認 `complete=true`、`paired_training_verified=true`；24 組評估均 `raw_audit_passed=true`。

本報告使用的 SUMMARY.json SHA256：`3d0ee27f1752211168052f19f14c9e61d7822f09a0694788987e8b04516f0532`。
