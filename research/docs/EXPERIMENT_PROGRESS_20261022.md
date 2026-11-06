# PHYRE 實驗進度與續接紀錄

紀錄日期：2026 年 10 月 22 日。涵蓋 10 月 21 日完成的三輪小規模實驗；更早的完整 pretraining、updater 與 AdaJEPA 結果仍保留在原始紀錄中，不混入本次比較。

目前的主要線索是：**用 support 的預測距離軌跡作為 query 排序的參照，值得繼續驗證；目前尚未證明觀測 residual 能帶來額外改善。** 三輪訓練與評估均已完成，紀錄時 sunflower 上沒有本帳號的排程工作。本次只整理進度，沒有啟動新實驗。

## 共同設定與指標

這三輪沿用相同的 cross fold 1 Q0、probe、候選池與第一個 action。每個場景使用 initial-scene classifier 選出的 1,024 個候選，再由 Q0 選第一個嘗試。Q0 是含 initial memory tokens、尚未根據 feedback 更新的 predictor。

Support 是第一次嘗試；query 是接下來要評分的候選 action。Residual 是這次 support 的實際觀測 latent，減去當下固定 Q0 的預測 latent。因為只研究第一次 feedback，而且 Q0 固定，所以這裡可以重用快取；不能據此推論多次更新後也可以重用舊 residual。

訓練使用 53 個第一次失敗的場景，每場固定 8 個均勻抽取、尚未嘗試的 queries，共 424 個樣本，其中 32 個成功。另有兩組訓練模板內的驗證場景，第一次失敗的數量分別為 12、10；它們不是官方 within benchmark。

Cross 評估使用同一組 100 個場景：3 個在第一個 action 成功，其餘 97 個進入第二次選擇。

- **S@2**：100 個場景中，兩次嘗試內成功的數量。包括最初成功的 3 個。
- **下一批 top-k**：97 個第一次失敗的場景中，重新排序後前 k 個 action 至少有一個成功的場景數。排序只更新一次，不能當成每次嘗試都重新更新的多輪成功率。

各輪內的對照固定資料與訓練設定；跨輪的訓練目標、模型與樣本重複使用次數不同。Updater 每次用一個場景的 8 個 queries，跑 128 steps；小 head 使用全部 424 個 queries，跑 512 steps。因此跨輪表格用於追蹤假說，不代表相同訓練成本的排行榜。

## 第一輪比較 updater 的訓練目標

完成 16 組短程 updater 訓練，包含原 prediction loss、加入 BCE、加入 pairwise loss、可學習的 BCE 分數校準，以及加入目標間距離監督。搭配真實 residual 與訓練時就歸零的 residual 對照；原 loss 與 pairwise 另有第二個 seed。

目前沒有一種目標，在 cross 上穩定勝過 Q0 與自己的 zero-residual control。不能下結論說 pairwise 比 BCE 好，也不能說只要換成 success loss 就能解決問題。

BCE／pairwise 常產生較大的 memory 更新；將更新縮小到 10%–50%，仍沒有穩定恢復排序。加入距離監督可以改善部分驗證集的預測誤差，但沒有穩定轉成 cross 排序改善。

另有一個使用真值的理想化診斷：在 10 個場景，把 query 的預測目標距離朝真實值修正 10%，top-1 從 0/10 變成 4/10。它說明某些物理修正有價值，但沒有證明 support 足以提供其他 queries 所需的修正資訊。

完整設定、結果與核對資料：[success objective loop](../results/success_objective_loop_20261021/FINAL_FINDINGS.json)。

## 第二輪改用獨立 ranking head

固定 world model 與 probe，在原本的靜態 learned head 上加入小型 MLP 分數修正。兩個 seeds 各比較真實 residual、zero residual，以及完全不看 support 的 query-only head，共 6 組訓練。

| 方法 | Seed 8419 的 S@2 | Seed 8420 的 S@2 |
| --- | ---: | ---: |
| Q0 不更新 | 5/100 | 5/100 |
| 靜態 learned head | 6/100 | 6/100 |
| Query-only head | 5/100 | 6/100 |
| Support 加 zero residual | 9/100 | 8/100 |
| Support 加真實 residual | 8/100 | 8/100 |

Zero-residual head 仍看到 support action、support 的預測距離特徵，以及 query 與 support 的相對特徵；也處於已知第一次失敗的情境。因此它不是完全沒有 support 或沒有任何 feedback 的控制組。

Support-conditioned head 有一些改善訊號，但真實 residual 沒有穩定勝過 zero control。Residual 使用池化與訓練集擬合的 16 維投影，這個結果不能排除其他 residual 表徵可能有效。

Cross 的 residual 打亂對照確實改變了全部 97 個 support。另兩組驗證資料在各模板只有一個失敗場景，因此同模板打亂沒有作用，不能拿來支持消融結論。

完整結果：[feedback readout loop](../results/feedback_readout_loop_20261021/FINDINGS.json)。

## 第三輪拆開 support 的輸入資訊

完成 10 種輸入組合、各兩個 seeds，共 20 組小 head 訓練。全部不使用觀測 residual，分別提供 action、預測距離特徵、相對差異，以及它們的組合。

距離特徵描述每個時刻綠球與目標之間的預測表面距離，不是完整物體位置或真實觀測軌跡。

新增了重要對照：**只提供 query 自己的完整 32 個時間點距離**。因為 query–support 距離差提供了原始逐時刻距離資訊，而舊 query-only 的 37 維特徵包含轉換後的接觸分數與統計量，兩者資訊解析度不同。

| 額外輸入或對照 | Seed 8419 的 S@2 | Seed 8420 的 S@2 |
| --- | ---: | ---: |
| 上一輪完整 support，residual 歸零 | 9/100 | 8/100 |
| Support action | 4/100 | 6/100 |
| Query 與 support 的 action 差 | 4/100 | 3/100 |
| Support action 加 action 差 | 5/100 | 5/100 |
| Support 預測距離特徵 | 5/100 | 5/100 |
| Query–support 距離軌跡差 | 5/100 | 5/100 |
| **Support 預測距離特徵加距離軌跡差** | **8/100** | **10/100** |
| Support action 加預測距離特徵 | 5/100 | 7/100 |
| Action 差加距離軌跡差 | 5/100 | 8/100 |
| Query 自己的完整距離軌跡 | 6/100 | 8/100 |
| Query 完整距離加 support action 與 action 差 | 9/100 | 7/100 |

「Support 預測距離特徵加距離軌跡差」的下一批 top-5，在兩個 seeds 都是 11/97；query 完整距離對照都是 5/97。這提供了進一步研究參照軌跡的理由，但它沒有全面勝過上一輪完整 support control。

把 support action 與預測軌跡整組換成同模板另一個場景的 support，並重算相對差異後，這個組合的下一個 action 成功數平均從 5、7 降至 3.2、5.6；但 top-8 的影響不一致。五次打亂與這些小幅差異，還不足以證明特定 support 對應關係是成功的原因。

完整資料：[所有組合的比較表](../results/support_context_ablation_20261021/CROSS100_TABLE.csv)、[分析](../results/support_context_ablation_20261021/FINDINGS.json)、[獨立核對](../results/support_context_ablation_20261021/AUDIT.json)。

## 目前結論的範圍

1. Support action 單獨沒有重現完整 support 的改善；預測距離特徵與相對軌跡的組合比較值得追查。
2. 最新候選完全不需要觀測 residual，且特徵在執行 action 前就能算出來。因此它支持的是排序參照資訊的研究方向，尚未證明 observation-driven memory 有效。
3. 目前仍只有一個 cross fold、共享的 Q0、小訓練資料與多種探索過的設定。另兩組驗證的改善不一致，尚未得到可宣稱穩定泛化的新方法。
4. 靜態 learned head 在一個 confirmation 場景有約 1405 的極端 logit；cross 的 logit 範圍正常。不能用該 confirmation 的整體 BCE 單獨判斷泛化，也不能用這個異常解釋 cross 排序結果。原始結果保留於[校準診斷](../results/feedback_readout_loop_20261021/STATIC_LOGIT_RANGE_DIAGNOSTIC.json)。

## 下一步尚未啟動

比較兩種參照：第一次選中 support 的預測距離軌跡，以及同場景 1,024 個候選的平均預測距離軌跡。固定 query 輸入、模型、訓練預算、第一個 action 與失敗後的評估方式。

候選池平均軌跡不需要先執行任何 trial。如果它能提供相近效果，改善可能主要來自場景的預測資訊；如果正確 support 明顯更好，才有理由繼續追查 support 特有的資訊。這仍需另行設計對照，才能判斷「已知這次失敗」與「觀測物理誤差」各自的價值。

## 執行與驗證紀錄

三輪共完成 42 組新的神經網路訓練：16 組 updater、6 組 feedback head、20 組 support feature ablation；另有靜態 readout 的擬合與不需訓練的診斷。

最新一輪的準備工作為 Slurm 36952，20 組訓練與評估為 36953 的 array jobs，均已完成。舊 control 的 476 組分數精確重播一致；獨立重算 11,824 組排名，並以 NumPy／SciPy 重播 630 組 head 輸出，top-k 成功判定一致。2,400 個結果與權重陣列檔案的本機及伺服器雜湊一致。

本機專案：`/path/to/phyre-unc`。

伺服器專案：`sunflower:/path/to/phyre-unc`。各輪結果位於相同的 `results/` 子目錄；公開程式與資產需求請見 [research README](../README.md)。
