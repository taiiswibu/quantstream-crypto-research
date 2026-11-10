# Run 001 — first user-account Kaggle TCN run

Status: **completed, uploaded artifact verified**. No promotion or new final-test evaluation.

## Hypothesis and purpose
Verify that the Git-backed development snapshot and TCN training pipeline run on the user's Kaggle account. This is a reproducibility/environment exercise, not a claim of new trading alpha.

## Frozen protocol
- Branch: `feat/kaggle-tcn-run-001`.
- Starting main revision: `a7d7eac5cae8e747590d45cff03fe0be4bfedfa5`.
- Actual training source revision: read `manifest.json → source_control.commit` from the exported input.
- Historical dataset: BTCUSDT and ETHUSDT hourly candles, 2023-01 through 2026-10.
- Feature snapshot SHA-256: `57980ab33b0fd7af78da048073906cddd617c84d30a7fed7c6de57ead0d79c83`.
- Three walk-forward development folds; fixed maximum **6 epochs/fold**.
- Epoch selection: validation MSE. Final refit epochs: median selected fold epoch.
- Lookback 48; seed 42; unchanged TCN architecture, Huber loss and AdamW setup.
- Threshold grid: 0, 0.0015, 0.003, 0.006; selection uses development results.
- Costs: 10 bps fee and 5 bps slippage each side; one-hour execution delay; four-hour holding.
- Device: GPU if available, otherwise CPU. Preserve actual `device_used` from the output.
- The previous final holdout starts 2026-01-25 17:00 UTC and is **not included in the Kaggle export**. Do not reopen it for this candidate.

## Planned comparison
Compare fold losses, selected epoch counts, validation metrics and threshold with the existing CPU reference. Floating-point/device/package differences can change results; bit-identical weights are not required.

## Observed results
- Uploaded artifact: `TCN_Kaggle_Output.zip`, SHA-256 `b08165a1c4cb985c3db773956c0d60617729e0697aa9d571ce8bb975703a051a`.
- Source snapshot revision: `7aba48a1d49d0f5801676befa3e7f64c99343b4d`; checkpoint and manifest hashes verified.
- Actual device: `cuda`. GPU model, runtime package versions, notebook URL/version were not recorded; do not infer them.
- **Protocol deviation:** observed histories contain **8**, not the planned 6, epochs per fold. The exported code revision does not prove which notebook cell was executed. Preserve the actual history.
- Best epochs: 3, 4, 2; final development refit: 3 epochs, 53,576 rows.
- Recomputed all fold strategy metrics from 21,406 uploaded predictions and the hashed development bars; they agree with metadata. Prediction keys are unique and targets match the frozen development snapshot.
- Aggregate prediction MAE: 0.0083017141; directional accuracy: 52.8730%. Both are **development selection** metrics.
- Chosen threshold: 0.006; 124 trades; mean fold Sharpe estimate: 0.452915. These values match the existing CPU reference; this run establishes pipeline execution, not incremental alpha.

| Development fold | Return after costs | Hourly Sharpe estimate | Maximum drawdown | Trades |
| --- | ---: | ---: | ---: | ---: |
| 1 | -2.2528% | -0.2448 | -8.5716% | 64 |
| 2 | 6.5857% | 1.3928 | -5.4246% | 39 |
| 3 | 0.5204% | 0.2108 | -3.0390% | 21 |

Learning curves continue reducing training loss after the best validation epoch. This motivates an inner chronological early-stopping split in the next experiment. The original implementation uses each outer validation set for both epoch selection and reported validation performance, so these metrics have selection optimism.

Machine-readable evidence: `reports/experiments/run-001-kaggle-tcn/tcn_metadata.json`. Weights and large prediction data remain outside Git.

## Next step
Run the budgeted model-search notebook. Do not initialize historical folds from this final checkpoint: it has already trained on the entire development period. Candidate promotion needs separate review; independent assessment requires new future observations.
