# Residual 與 action 融合診斷

日期：2026-10-19。這份紀錄獨立保存，不加入 milestone slides。

目前結果顯示，updater 確實使用 residual：將 residual 歸零會明顯改變 memory 更新與下一個 action。但保留 residual 內容、打亂其時空排列，更新卻幾乎不變。因此，現有證據較支持「對明確時空對應關係不敏感」，不支持「residual 加上 action 後就完全消失」。這尚未證明是 cross 失敗的根因。

## 問題與實作

Residual 是同一個 attempted action 的「實際影片特徵 − 目前 memory 下的預測特徵」。目前 updater 依序執行：

1. 將 32 個時間點抽樣為 16 個，空間從 14 × 14 pooling 成 10 × 10。
2. 對 residual 做線性投影，加上 attempted action embedding、時間與空間 embedding。
3. 對相加結果做 LayerNorm。
4. 以現有 memory 為 cross-attention 的 query，融合後的特徵為 key/value，更新 memory。

這次要區分三個問題：residual 是否被其他 embedding 壓過、更新是否對 residual 有反應，以及這些反應是否影響 ranking。訊號幅度或敏感度本身不等於任務表現。

## 實驗設定

使用最新 Q0-first 實驗的 **residual-trained updater**，cross fold 1 與 within fold 0，各 10 個場景，均為 seed 0。只納入第一次 action 失敗、因此有 memory update 的場景。依 template 與 scene 的固定 hash 次序取樣，不依後續改善或退步挑選。

每個場景檢查第一次更新，以及第四次更新；若原始 rollout 較早結束，改用最後可取得的更新。總共 20 個場景、40 個更新狀態；同一場景的兩個狀態不是獨立樣本。

每次干預固定當下 memory、attempted action、原始已嘗試動作歷史與候選池。使用保存的實際觀測，對同一組 1,024 個 actions 重新排名，沒有重訓、額外模擬或新的閉環 rollout。

| 條件 | 改動 |
|---|---|
| Real | 原始 residual 與 action embedding |
| Zero residual | residual 歸零，保留 action 與位置資訊 |
| Shuffled residual | 打亂 pooling 後 residual tokens 的時空排列；位置 embedding 不動 |
| Scale 0.1 / 10 | residual 分別縮小或放大十倍 |
| No action embedding | 移除 action embedding，保留 residual 與位置資訊 |

**Zero residual 是對 residual-trained 模型的干預，不是先前另外訓練的 zero-residual control。** 兩者不能混為同一個比較。

## 驗證與執行

四個 GPU 分片全部成功，耗時 12 分 35 秒至 13 分 57 秒。40 個狀態的原始更新前後 score vectors 均與既有 evaluation 精確一致；拆解後的 real updater 與正式 updater 的輸出也精確一致。模型權重保持不變。

下載後另行從 240 組 score arrays 重算 next action、top-10 overlap 與 pairwise AUC，全部通過。四個 synthetic CPU tests 只驗證診斷實作，不代表模型效果。

## 主要結果

下表是**第一次更新**的場景中位數。更新差異定義為 `||干預後更新量 − 原始更新量|| / ||原始更新量||`，不是 success 的變化。

| 指標 | Cross | Within |
|---|---:|---:|
| Projected residual / action 的 centered RMS | 0.453 | 0.409 |
| Residual 歸零造成的 LayerNorm 輸出相對差異 | 34.9% | 30.6% |
| Residual 歸零造成的 memory 更新相對差異 | 40.8% | 86.4% |
| 打亂 residual 排列造成的 memory 更新相對差異 | 0.294% | 0.647% |

後期更新也呈現類似差異：歸零造成 34.8% / 79.0% 的更新差異，打亂排列僅 0.304% / 0.387%。

下表合計每個 protocol 的 20 個更新狀態，計算下一個 action 是否與原始 residual 條件不同。

| 干預 | Cross | Within |
|---|---:|---:|
| Residual 歸零 | 10 / 20 | 16 / 20 |
| 打亂 residual 排列 | 0 / 20 | 2 / 20 |

時間與空間 embedding 的 RMS 約 0.0021–0.0023，低於 projected residual 的約 0.110 / 0.084，以及 action embedding 的約 0.269 / 0.229。這提供位置訊號較弱的線索，但不能單靠幅度判定模型忽略位置。

## Ranking 是否改善

第一次更新的 pairwise AUC 相對更新前之變化，中位數如下。AUC 衡量成功 action 排在失敗 action 前面的程度；使用同一個已嘗試動作 mask。

| 條件 | Cross，6 個可定義 AUC 場景 | Within，10 個可定義 AUC 場景 |
|---|---:|---:|
| Real residual | −0.0784 | −0.0628 |
| Zero residual 干預 | −0.0656 | −0.0185 |

Cross 其餘 4 個場景沒有同時包含兩種 outcome，不能定義此 AUC。這些數字不能推出「關掉 residual 一定比較好」：後期 within 的 real residual 下一個 action 成功 2 次，zero residual 為 0 次。也不能把固定歷史下的單步結果當成新的 S@9。

## 可以與不可以下的結論

- **可以說：** residual 的內容確實影響 memory 與 action selection；目前沒有完全被 action addition 或 LayerNorm 消除的證據。
- **可以說：** 在這批狀態中，updater 對 residual 和顯式時間／位置 embedding 的對應關係敏感度低。
- **尚不能說：** 模型完全沒有幾何或位置資訊。Residual latent 本身可能攜帶位置資訊，而且 cross-attention 會聚合一組 tokens。
- **尚不能說：** pooling 損失了固定比例的有效資訊。約 0.73–0.74 的 pooling abs-retention 只反映數值抵消，不代表保留了多少任務資訊。
- **尚不能說：** fusion 是 cross 失敗的根因，或換成 concatenation 就會改善。歸零、打亂與放大也可能產生訓練分布之外的輸入。

下一步若要繼續歸因，可分開打亂時間與空間、移除位置 embedding，並加入「residual 與位置一起排列」的對照。先確定模型使用哪些對應關係，再決定是否重訓不同 fusion 架構。這些後續實驗尚未提交。

## 可重現資料

- [固定計畫與取樣](../results/residual_fusion_diagnostic_20261019/PLAN.json)
- [完整彙整數字](../results/residual_fusion_diagnostic_20261019/SUMMARY.json)
- [完成狀態](../results/residual_fusion_diagnostic_20261019/STATUS.json)
- [獨立驗證結果](../results/residual_fusion_diagnostic_20261019/VERIFICATION.json)
- [診斷程式](../results/residual_fusion_diagnostic_20261019/code/diagnose.py)

逐場景結果與 score arrays 位於同目錄 `cases/`。來源模型與原始 rollout：`results/q0_first_feedback_20261018/`。Sunflower jobs：36004 的四個分片、彙整 36008，皆 exit 0。
