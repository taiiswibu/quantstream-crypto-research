# Roadmap

## Completed local work
- [x] Real ingestion, checksum, quarantine and gap handling.
- [x] Causal features, purged splits and three-fold model research.
- [x] TCN and exploratory Chronos LoRA runs.
- [x] Cost-aware backtest and locked final holdout.
- [x] Local dashboard, paper worker and monitoring.
- [x] Baseline Git import and snapshot-contract regression fix.
- [x] Git provenance, CI configuration and contribution documentation.

## GitHub and next research cycle
- [x] Publish to the user's new repository.
- [x] Observe successful remote CI on Linux and Windows.
- [x] Verify the uploaded CUDA TCN output and record results/protocol deviation.
- [x] Prepare budgeted Ridge/XGBoost/TCN/GRU search with inner early stopping and multi-seed checks.
- [x] Prepare a controlled Chronos LoRA steps/learning-rate lab.
- [x] Verify all 12 model-search trial predictions, diagnostic/cost metrics and record a research-only decision.
- [x] Execute the predeclared Chronos-2 LoRA A/B/C lab; independently recompute exported development metrics and retain research-only status.
- [x] Commit notebook 05 torchao remediation, CPU adapter preflight and runtime evidence capture (full Kaggle GPU rerun still unverified).
- [x] Draft [Run 004 seed-stability protocol](experiments/run-004-seed-stability-protocol.md), with explicit training budget and separate prospective gates; not executed.
- [x] Implement Run 004 runner with independent fits, shared-sample checks, trainer/adapter evidence and weekly paired bootstrap.
- [x] Review notebook 06 Kaggle GPU preflight, adapter/step evidence and freeze the environment lock with source/data/budget identities.
- [ ] Record the actual Kaggle notebook URL and materialize the frozen per-run plan before the first research fit.
- [ ] Execute all three Run 004 seeds (or record an interrupted/partial run) and independently verify the output.
- [ ] Make sleeve vs portfolio turnover units explicit with accounting coverage.
- [ ] Add block-based uncertainty and per-regime validation.
- [ ] Define new future evaluation before assessing a revised model.
- [ ] Record a forward paper-trading observation period.

Kafka/warehouse extensions depend on measured needs. Profitable performance remains an open research goal.
