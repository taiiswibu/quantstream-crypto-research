# Run 003 — controlled Chronos-2 LoRA learning exercise

Status: **prepared, not run**.

Hypotheses: compare 100 vs 200 steps at LR 1e-5, then LR 1e-5 vs 3e-6 at 200 steps. Each adapter starts from base revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, seed 42, context 128, batch 8. Evaluate on the same fixed 512 sampled development windows; save zero-shot and adapter predictions.

Notebook: `05_Kaggle_LoRA_Lab.ipynb`; output: `QuantStream_LoRA_Output.zip`. Outcomes are unknown. This exercise does not tune LoRA rank/target modules, validate pretraining non-overlap, estimate strategy alpha, open the final holdout, or automatically promote an adapter. Any apparent gain is development-only and needs a separate future-data assessment.
