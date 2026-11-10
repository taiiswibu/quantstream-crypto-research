# Reference experiments

Imported evidence from the delivered MVP, executed before Git tracking. Git setup did not retrain models or reopen the final holdout; no new source commit is assigned to the old runs.

## ML and TCN

67,196 valid hourly BTCUSDT/ETHUSDT candles from 92 archives. Two partial-hour rows were quarantined. Gaps restart indicators and sequences.

| Fold | Validation start UTC | Validation end UTC |
|---|---|---|
| 1 | 2024-11-05 03:00 | 2025-04-02 23:00 |
| 2 | 2025-04-02 23:00 | 2025-08-29 20:00 |
| 3 | 2025-08-29 20:00 | 2026-01-25 17:00 |

Training labels crossing the next boundary are excluded; assets share the same boundaries. TCN best epochs were 3, 4 and 2. Final refit used 53,576 development rows and median 3 epochs.

Threshold: predicted return > 0.006. TCN mean validation Sharpe estimate was 0.4529 (124 trades), versus Ridge 0.4440 (20 trades). The small difference does not establish stable superiority.

Final TCN: -14.4567% net return, -17.0877% max drawdown, -2.0601 Sharpe estimate, 46 trades and 51.4183% directional accuracy. This is an unsuccessful strategy result after costs. The final test is already opened.

Original reports: [final report](../reports/reference/final_report.json), [model comparison](../reports/reference/comparison.json), [TCN metadata](../reports/reference/tcn_metadata.json).

Legacy `turnover` sums notionals in separate unit-capital asset sleeves. Divide by the sleeve count before interpreting it as normalized portfolio turnover. Return curves already average sleeves; a future accounting change should make turnover units explicit.

## Chronos-2 LoRA

Base `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. LoRA: 200 steps, context 128, batch 8, learning rate 1e-5, CPU. Same 512 sampled validation windows before/after; final holdout absent.

| Metric | Zero-shot | Fine-tuned |
|---|---:|---:|
| Return MAE | 0.007836 | 0.007916 |
| Directional accuracy | 50.20% | 49.80% |

Fine-tuning did not improve this comparison. Pretraining overlap remains unverified. Future entry open is approximated by a forecast of its preceding close. This experiment is exploratory and did not choose the champion.

[Original LoRA report](../reports/reference/chronos_report.json). No Kaggle/GPU run or competition score is claimed. New hypotheses should use [the experiment template](experiments/TEMPLATE.md) and a new independent future evaluation period.
