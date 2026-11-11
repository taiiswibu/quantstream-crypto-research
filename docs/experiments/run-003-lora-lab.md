# Run 003 — controlled Chronos-2 LoRA lab

Status: **completed; predictions and adapter integrity checked; research only (`research_only_no_promotion`)**.

## Predeclared experiment

Notebook: `notebooks/05_Kaggle_LoRA_Lab.ipynb`, exported source commit `7f4b8ec8eece36452782e1a8f1bc0b3d1160aeb7`. The same pinned `amazon/chronos-2` checkpoint (revision `29ec3766d36d6f73f0696f85560a422f50e8498c`), seed 42, context 128, batch 8 and 512 sampled development windows were used by the reported three experiments.

| Adapter | Steps | Learning rate | MAE | Relative MAE change vs zero-shot | Directional accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | — | — | **0.0078359965** | — | **50.1953%** (257/512) |
| A | 100 | 1e-5 | 0.0080378979 | +2.5766% worse | 48.6328% (249/512) |
| B | 200 | 1e-5 | 0.0078874549 | +0.6567% worse | 47.6563% (244/512) |
| C | 200 | 3e-6 | 0.0079001647 | +0.8189% worse | 48.2422% (247/512) |

Lower MAE is better. B has the lowest MAE of the **three tuned adapters**, but all three are worse than the **same zero-shot checkpoint** on these windows. B is not a promoted model or a proven trading strategy. A/B isolates the declared steps change; B/C isolates the declared learning-rate change only within this one-seed, one-development-sample experiment. No broad causal generalization is justified.

The 512 rows comprise 256 BTCUSDT and 256 ETHUSDT decision windows from **2025-08-29 20:00 UTC** to **2026-01-25 11:00 UTC**, both in the historically observed development period. On BTC and ETH separately, **each adapter's MAE is also higher than zero-shot**. This is not an unseen holdout, a new trading backtest, or a profitability claim.

## Artifact and independent checks

- Received Kaggle output ZIP: `QuantStream_LoRA_Output.zip`, SHA-256 `42ce81ff99b65b2db6d41a1e1215e5c38e36e56bd49adfc868d0a96ad85aeb9d`. Keep this 12 MB artifact out of Git; retain it for audit.
- `comparison.json`, `plan.json`, 3 per-experiment `chronos_report.json`, 3 Parquet predictions, 3 snapshot manifests, and all 3 serialized adapters exist. ZIP CRC passed.
- Parsed **all 512 prediction rows per adapter** directly from Parquet; recomputed MAE and directional accuracy to reported precision. Verified same 512 symbol/timestamp/target rows and identical zero-shot predictions across A/B/C, unique keys, finite values and 256 observations per asset.
- Report, plan and copied snapshot manifest agree on configurations, source commit, checkpoint revision and dataset hash `c8e0112916e6cea733338ead4412d6e9553e4c5844a2d200d1fd38e31c747fd0` (same exported development snapshot used in Run 002).
- Each serialized adapter is a readable PEFT LoRA safetensors artifact: 194 tensors, 1,206,912 stored parameters, all finite; adapters have distinct SHA-256 digests. Metadata reports Tesla T4, CUDA, PyTorch `2.11.0+cu130` and PEFT `0.21.2`.
- Audit summary: [verification.json](../../reports/experiments/run-003-lora-lab/verification.json).

**Verification boundaries:** the original complete Kaggle input, trainer step-by-step logs, notebook execution URL/version and original market bars were not supplied here. Therefore ZIP checks **do not independently attest** that the exact optimizer-step count executed, that target returns match the original exchange data, or that all notebook-side environment changes were committed. The earlier PEFT/torchao incompatibility was reported during execution; the successful artifact records a newer PEFT version, but does not establish which exact repair steps were applied. Reproducible runtime remediation remains an open task.

## Decision and next steps

Keep **`research_only_no_promotion`**. Do not choose B for deployment simply because it beats A and C: it still loses to zero-shot, and the single-seed sample is development-only. No final-test reuse or trading performance estimation was performed by this lab.

For a next *distinct* hypothesis, record a protocol and environment lock before re-running: verify the complete training environment, test alternative fine-tuning data objectives and horizons, add multiple random seeds and use fresh prospective windows for independent evaluation. Report both prediction error and trading results after costs only after the strategy is specified in advance. The three planned A/B/C outcomes should remain unchanged historical evidence.
