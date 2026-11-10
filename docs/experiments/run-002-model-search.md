# Run 002 — budgeted model search

Status: **prepared, not yet run on the user's Kaggle account**.

Hypothesis: a small, fixed-budget search across Ridge, XGBoost, TCN and GRU can identify lower normalized forecast MSE than the fixed Ridge baseline and zero-return forecast on development. This is an educational model-comparison exercise; profitability is not assumed.

Protocol: `quantstream.search.PROTOCOL = budget-search-v1`; notebook `04_Kaggle_Model_Search.ipynb`; up to 12 total Optuna trials, 8 neural epochs/fit, tuning folds 0/1, inner chronological 20% early stopping with label purge. Fixed threshold .006 and unchanged costs. Top 2 configurations evaluated across seeds 42/123/2026; selection uses seed-mean normalized MSE. Historically observed fold 2 is used once after freezing selection for a development diagnostic, with fixed cost stresses.

All runtime parameters and checksums will be stored in `plan.json` from the actual run. No independent final test, serving checkpoint refit, candidate promotion or live orders are performed. Do not warm-start historical folds from the already fitted full-development TCN.

Output expected: `QuantStream_Research_Output.zip`. Actual trial count, runtime, parameter choices, metrics and decision remain unknown until that artifact is received. Preserve failed/pruned/interrupted attempts and any protocol deviations.
