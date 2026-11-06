# 原始 base、Q0 與 feedback 方法：完整比較

日期：2026-10-19。新補的原始 base evaluation 已全部完成，並在下載後獨立核對 200 個場景。

**原始 base 的 S@9 為 cross 7%、within 46%；Q0 為 11%／50%；目前去平均值 updater 為 15%／57%。** 所以，相對原始 base 的改善不能全部算成 feedback 的效果；Q0 本身已經改善部分排序。

## 比較設定

Classifier 先對 100,000 個 actions 排名、保留 top 1,024。所有方法使用完全相同的候選池、標籤與場景；cross fold 1、within fold 0 各 100 個場景，最多 9 次嘗試。固定 probe／scorer，不使用測試成功標籤挑選 action。

原始 base 不含 initial memory tokens，也沒有 updater。它以自己的 predictor＋scorer 排名選第一個 action，失敗後沿原排名繼續。Q0 則包含預訓練的 initial tokens，但不根據 feedback 更新。其餘方法從相同 Q0 第一個 action 出發，失敗後使用各自的更新方法。

因此，原始 base 與 Q0 的第一個 action **允許不同**；這是整個 policy 的比較。為了區分第一步與後續選擇的影響，另計算一個控制：先試 Q0 的第一個 action，之後按 base 靜態排序嘗試，排除已試 action。這個控制只需重用 base scores，不需要另一輪 GPU 推論。

Base 與 Q0 的預訓練已核對：都是 scratch、8 epochs、seed 0；cross λ=0，within λ=0.03。同一 protocol 的初始 backbone hash、資料抽樣 hash、樣本呈現次數與 optimizer steps 完全一致。Cross 為 418,176 次樣本呈現／52,272 steps；within 為 418,752／52,344。這些是呈現次數，並非不同樣本數。

Updater 另外接受 128-slot 離線訓練，AdaJEPA-style 則在 deployment 更新 Q0 predictor。這裡匹配的是 evaluation 場景、候選池與嘗試預算，沒有宣稱各方法總訓練／推論計算量相同。

## 完整結果

每格皆以同一批 100 個場景為分母。

| 方法 | Cross S@1 / S@3 / S@5 / S@9 | Within S@1 / S@3 / S@5 / S@9 |
|---|---|---|
| 原始 base：沒有 memory tokens | 3% / 4% / 7% / 7% | 26% / 36% / 39% / 46% |
| Q0：initial tokens，不更新 | 3% / 7% / 8% / 11% | 29% / 42% / 47% / 50% |
| Q0＋AdaJEPA-style adaptation | 3% / 5% / 6% / 9% | 29% / 45% / 51% / 58% |
| Uniform residual updater，128 slots | 3% / 7% / 10% / 12% | 29% / 45% / 50% / 56% |
| 同一 updater，推論時 residual 去平均值 | 3% / 6% / 10% / 15% | 29% / 46% / 51% / 57% |

去平均值 updater 對原始 base 的 S@9 差距為 cross ＋8、within ＋11 個百分點；對 Q0 則是 ＋4／＋7。前者包含 Q0 起點的改善，後者較接近我們要衡量的更新效果。

Q0 相對 base，cross 救回 7 個、失去 3 個場景；within 救回 13 個、失去 9 個。去平均值 updater 相對 base，cross 救回 12 個、失去 4 個；within 救回 20 個、失去 9 個。方法優勢不是每個場景都一致。

## 固定第一個 action 的控制

| 方法 | Cross S@1 / S@3 / S@5 / S@9 | Within S@1 / S@3 / S@5 / S@9 |
|---|---|---|
| Base 自己選第一個 action | 3% / 4% / 7% / 7% | 26% / 36% / 39% / 46% |
| 先試 Q0 第一個 action，再沿 base 排序 | 3% / 7% / 10% / 10% | 29% / 39% / 41% / 48% |
| Q0 自己排序，全程不更新 | 3% / 7% / 8% / 11% | 29% / 42% / 47% / 50% |

固定 Q0 第一個 action 後，base 的 S@9 變成 cross 10%、within 48%，與 Q0 的 11%／50% 更接近。這說明第一步選擇會影響對 base 的比較。Cross 的 base 與 Q0 雖然 S@1 都是 3%，也不表示它們選了相同 action 或解出了相同場景。

這個控制仍不代表 feedback update 沒有作用：去平均值 updater 相對「Q0 first＋base 排序」仍是 cross ＋5、within ＋9 個百分點。但這是單一起點、單一 fold 的小規模結果，不是跨 seed／fold 的穩定性結論。

## 驗證與成本

兩個 protocol 各一個 smoke 場景：base 快速推論的完整 1,024 個 scores 與原始較慢 scoring 路徑完全相同；Q0 的完整 scores 也精確重現已有 evaluation。確認 base 沒有 initial tokens／memory state，兩者使用相同固定 head 與 scorer rule。

完整 200 個場景逐一核對候選 IDs、成功標籤、有限 scores、stable ranking、固定第一個 action 的控制、S@1／3／5／9 統計與 checkpoint／head 不變性。所有核對通過，未使用影片 encoder、模擬或新增訓練。

16 個 GPU 分片的最慢執行時間為 **2 分 40 秒**，不含前置 smoke；所有 jobs 已完成。

## 證據

- [新比較完整結果與配對場景](../results/plain_base_comparison_20261019/SUMMARY.json)
- [數值重現驗證](../results/plain_base_comparison_20261019/SMOKE.json)
- [Base／Q0 預訓練匹配核對](../results/plain_base_comparison_20261019/PRETRAINING_MATCH.json)
- [評估程式](../results/plain_base_comparison_20261019/code/evaluate.py)
- [Slurm 執行時間](../results/plain_base_comparison_20261019/JOB_TIMES.psv)
- [先前夜間調查報告](MEMORY_OVERNIGHT_INVESTIGATION_20261019.md)

這份比較沒有修改 milestone slides；既有夜間報告與 completion audit 保留原始版本。
