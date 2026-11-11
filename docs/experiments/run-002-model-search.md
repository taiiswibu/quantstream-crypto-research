# Run 002 — budgeted model search

Status: **completed; uploaded predictions and strategy metrics verified; research only**.

Hypothesis: a small, fixed-budget search across Ridge, XGBoost, TCN and GRU can identify lower normalized forecast MSE than the fixed Ridge baseline and zero-return forecast on development. This is an educational model-comparison exercise; profitability is not assumed.

Protocol: `quantstream.search.PROTOCOL = budget-search-v1`; notebook `04_Kaggle_Model_Search.ipynb`; up to 12 total Optuna trials, 8 neural epochs/fit, tuning folds 0/1, inner chronological 20% early stopping with label purge. Fixed threshold .006 and unchanged costs. Top 2 configurations evaluated across seeds 42/123/2026; selection uses seed-mean normalized MSE. Historically observed fold 2 is used once after freezing selection for a development diagnostic, with fixed cost stresses.

No independent final test, serving checkpoint refit, candidate promotion or live orders were performed. Do not warm-start historical folds from the already fitted full-development TCN.

## Actual execution and verification

- Uploaded artifact: `QuantStream_Research_Output.zip`, SHA-256 `c55779f1735ba0b1fa9b1517502c054c02c3b1ed42b0f391bf5ff11b841621ec`.
- Source: clean exported revision `7f4b8ec8eece36452782e1a8f1bc0b3d1160aeb7`.
- Snapshot manifest SHA-256: `c8e0112916e6cea733338ead4412d6e9553e4c5844a2d200d1fd38e31c747fd0`.
- Uploaded runtime reports Tesla T4 / CUDA, Python 3.12.13, PyTorch 2.10.0+cu128 and Optuna 4.9.0. Elapsed 211.83 seconds, approximately 3m32s. These are recorded metadata, not independent execution attestation; notebook URL/version was not supplied.
- All 12 planned trials completed: 5 TCN, 3 Ridge, 2 GRU, 2 XGBoost. No failed/pruned trials are recorded in this artifact.
- Source and snapshot hashes match the frozen export. Recomputed forecast and backtest metrics for all 12 saved trial prediction files; all agree. All trials have the same 14,270 symbol/timestamp/fold keys, targets match source, and labels stay inside the correct fold boundaries.
- Selection hash seal matches. Artifact timestamps place selection freeze before diagnostic start and completion; this consistency check is not a signed proof of execution order.
- Independently recomputed the diagnostic from 7,136 predictions and all three cost stresses. No final-holdout rows were used.
- Stability reports for seed 123/2026 do not include their underlying predictions. Checked their arithmetic and consistency with selection, but cannot independently recompute those seed backtests from this ZIP.

Machine-readable evidence is in `reports/experiments/run-002-model-search/`: original plan/selection/result/diagnostic, trial table and `verification.json`. Large predictions and SQLite state remain in the original uploaded artifact rather than Git.

## Selection and seed stability

Trial 2 TCN parameters: lookback 48, width 16, head dropout 0, AdamW learning rate 0.0013394334706750498, weight decay 0.0006358358856676254, batch size 256. Threshold remains fixed at 0.006.

| Candidate | Normalized MSE, seed 42 | Mean across three seeds | Seed standard deviation |
| --- | ---: | ---: | ---: |
| TCN trial 2 | 0.988080620 | 0.993059614 | 0.004287192 |
| Ridge trial 11, alpha 0.128443776 | 0.993823233 | 0.993823233 | approximately 0 |

Zero-return forecast has normalized MSE 1. The TCN's apparent 1.19% error reduction at seed 42 shrinks to 0.69% when averaging three seeds. Its mean error advantage over tuned Ridge is only about **0.077% relative**. Three-seed standard deviation is not a confidence interval or a statistical significance test; deterministic Ridge repeats are not independent evidence.

| TCN seed | Two-fold normalized MSE | Selected inner epochs, folds 1/2 | Two-fold trade count |
| --- | ---: | --- | ---: |
| 42 | 0.988080620 | 1 / 2 | 18 |
| 123 | 0.992552907 | 1 / 1 | 53 |
| 2026 | 0.998545314 | 3 / 2 | 109 |

The 18/53/109 trade-count range indicates substantial sensitivity of threshold crossings to initialization. Forecast MSE was the predeclared selection objective; this experiment did not impose the old strategy-research minimum-trade eligibility gate or promote a trading champion. In particular, the four trades in the first tuning fold at seed 42 cannot establish reliable strategy performance.

## Post-selection development diagnostic

Period: 2025-08-29 20:00 UTC to 2026-01-25 17:00 UTC. This is the historically observed third development fold, **not an independent final test**. Frozen candidate evaluated at the predeclared seed 42; selected inner epoch 3.

- Forecast normalized MSE: **1.009803034**, approximately 0.98% worse than the zero-return forecast.
- Forecast MAE: 0.0074602674; directional accuracy: 52.2982% across 7,136 rows.
- Strategy win rate at base costs: 26/49 = 53.0612%.

| Cost multiplier | Fee + slippage each side | Net return | Hourly Sharpe estimate | Maximum drawdown | Trades |
| --- | --- | ---: | ---: | ---: | ---: |
| 1x | 10 + 5 bps | 2.8873% | 0.7381 | -5.6181% | 49 |
| 1.5x | 15 + 7.5 bps | -0.7079% | -0.1222 | -5.9745% | 49 |
| 2x | 20 + 10 bps | -4.1242% | -0.9764 | -7.1195% | 49 |

Base-cost trades are concentrated in ETH (41), with 8 BTC trades. Mean net return per executed trade is +0.1823% for ETH and -0.1496% for BTC; these are trade averages, not asset portfolio returns. The legacy `turnover` field sums sleeve notionals and is not a normalized portfolio turnover rate.

Positive selected-tail strategy PnL can coexist with worse full-population forecast MSE. Neither metric cancels the other: they answer different questions. Here, the limited forecast gain, initialization sensitivity, cost fragility, historical data reuse and small trade sample do not support an independent alpha claim.

## Decision and next experiment

Keep **`research_only_no_promotion`**. Preserve all trials, fixed costs/threshold and existing final-test lock; do not rerun the diagnostic to pick a more favorable seed, threshold or model. Do not retrospectively expand the search budget to improve this report.

Next authorized learning experiment remains run 003: the already specified three-configuration Chronos LoRA lab. Use the same exported Dataset with notebook 05; compare steps/LR under its pinned protocol and return the adapter/evidence ZIP. Its results will also be exploratory development results, not an independent trading test.

For a later strategy-research cycle, register a new hypothesis and fresh prospective observation period before evaluation. Per-asset/regime diagnostics, block-based uncertainty and feature ablations remain possible follow-ups, not analyses completed by this run.

Environment caveat: both `xgboost` 3.2.0 and `xgboost-cpu` 3.4.1 distributions are listed in the uploaded environment. The loaded module version was not recorded separately, so do not infer which distribution actually supplied the imported code from this metadata alone.
