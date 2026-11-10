# Run 001 — first user-account Kaggle TCN run

Status: **planned**. No Kaggle training result has been recorded yet.

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

## Results — fill after the run
- Kaggle notebook URL/version:
- Actual Git SHA / dataset hashes:
- Actual device and package versions:
- Fold and aggregate validation metrics:
- Output artifact: `TCN_Kaggle_Output.zip`.
- Decision and limitations:

## Next step
Download the real output and update this record in a new commit. Candidate promotion is a separate validation decision; any independent final assessment requires a new future period.
