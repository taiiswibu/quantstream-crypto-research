# Run 005 — fixed-weight inference controls

Status: **protocol and runner implemented; full Kaggle diagnostic not executed**. No Run 005 score, model promotion or prospective evaluation is claimed.

Bootstrap revision: **`run-005-inference-v1-runtime-fix-001`**. The uploaded probe confirms duplicate NumPy metadata: first match/imported 2.1.3, legacy last match 1.26.4. The only legacy inventory differences are missing Chronos 2.3.2 and extra torchao 0.10.0. See the [verified runtime incident](run-005-runtime-incident.md). Preserve the historical lock and all prediction treatments; repair only these two package differences without dependencies, require the complete legacy inventory match and add an explicit loaded-module contract from the probe. No historical loaded NumPy identity is claimed. Full inference remains pending the corrected Kaggle run.

## Question and fixed controls

Run 004 seed 42 reproduces Run 003 B's adapter bytes exactly, but their stored forecasts/MAE differ. Run 004 explicitly uses eval mode; the historical artifact lacks a model train/eval flag and complete runtime inventory. This diagnostic measures mode and RNG effects under one controlled runtime. It cannot uniquely reconstruct the old unrecorded state or establish the complete cause of its historical score.

The adapter is selected **because of the historical byte match**, not because seed 42 has the best Run 004 MAE. There is no new fitting, optimizer, backward pass, seed selection, threshold tuning, portfolio operation or final-holdout access.

| Control | Fixed value |
| --- | --- |
| Model | `amazon/chronos-2` |
| Base revision | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| Adapter | Run 004 seed 42, identical to Run 003 B |
| Adapter safetensors SHA-256 | `fb1769324c59a682ef7b84a674fab1b0275e5ed0e408981cb0e27cc4a094182d` |
| Original Run 004 ZIP SHA-256 | `0c53a0dabcd2c67e3c36993628d9495c4027d950840be5a28673f9a0b849c333` |
| Loading | Explicit pinned base + `PeftModel.from_pretrained(..., is_trainable=False)`; **unmerged** |
| Context / prediction batch / horizon | 128 / 16 / 5 closes, preserving the 4-hour-return transformation with one-bar execution lag |
| Data | Same four Parquet bytes/data contract, 512 sample rows, contexts and training-series fingerprints as Run 004 |
| Runtime | Same full legacy Run 004 Python/package/GPU/CUDA/cuDNN inventory, plus imported-module validation from the reviewed probe; stop and preserve diagnostics on a difference |
| Tolerance | `rtol=1e-6`, `atol=1e-9`; also report exact equality and maximum/mean absolute prediction difference |
| Optimization | **0 optimizer steps**; all parameters frozen and every forecast under `torch.inference_mode()` |

When Kaggle automatically extracts an uploaded Run 004 ZIP, validate the exact bytes of six selected reference files against the hashes hardcoded before execution: plan, root/seed-42 statuses, prediction Parquet, adapter config and safetensors. Do not rebuild a ZIP and call its hash the original. Raw and expanded input forms are recorded separately; the original ZIP SHA is recomputed only when those ZIP bytes exist.

Snapshot export date/source-code metadata may differ only under the existing Run 003 scientific-identity validator. Actual Parquet files are rehashed. The sample/contexts/training identities must match Run 004; labels are used for scoring and alignment, never fed to the model.

## Predeclared pass order and budget

First run one **base zero-shot eval** pass and compare it with the stored Run 004 zero-shot forecasts. Then load the selected adapter without merging and run:

| Order | Case | Model mode | Inference RNG seed |
| --- | --- | --- | ---: |
| 1 | `eval_42_a` | eval | 42 |
| 2 | `eval_42_b` | eval | 42 |
| 3 | `eval_123` | eval | 123 |
| 4 | `dropout_42_a` | train mode, gradients disabled | 42 |
| 5 | `dropout_42_b` | train mode, gradients disabled | 42 |
| 6 | `dropout_123` | train mode, gradients disabled | 123 |
| 7 | `dropout_2026` | train mode, gradients disabled | 2026 |
| 8 | `eval_42_after` | eval, after dropout cases | 42 |

These are **inference RNG seeds**, not newly trained model seeds. The two same-RNG dropout passes are a reproducibility control, not independent samples. Total budget: **nine passes × 512 windows = 4,608 forecast rows**, including one base control; no additional passes to chase a historical MAE. This is a mechanism diagnostic on old development data, not additional out-of-sample evidence.

## Measurement and decisions

- Hash all model parameters and persistent buffers after loading and after every adapter pass. Abort on changed state, non-finite forecasts, enabled gradients or trainable parameters. Preserve partial files and the error. Do not merge adapters or save a newly named model.
- Record requested/observed model mode, count of active nonzero `nn.Dropout` modules, CPU/CUDA RNG fingerprints, state hashes and every forecast. Restore eval mode when the experiment exits.
- Compare same-RNG eval repeats, eval across RNG seeds, eval before/after dropout, same-RNG dropout repeats, dropout across RNG seeds, and mode change with RNG held at 42. Report exact equality and tolerance checks separately.
- Compare the first adapter eval pass with the saved Run 004 seed-42 forecasts; report this separately from within-runtime controls. Compare the separate base pass with its zero-shot reference.
- Export MAE, MSE and directional accuracy for **every** case, pooled and by BTC/ETH. Do not select the lowest-error dropout draw, create an ensemble or infer a new trading edge.

`status=completed` means the planned passes finished. `controls_passed` additionally requires mode treatments to hold, active dropout in the requested treatment, and reproducibility within the declared tolerance. `reference_reproduced` additionally requires both stored Run 004 references to match. Failed controls/reference are a diagnostic result to investigate, not a reason to relabel the run as successful reproduction. If dropout has no observed effect, report the null result; do not force one.

Full market inputs and numerical predictions are supplied by the user's snapshot/artifact. The environment lock is a version inventory, not a container/image or bitwise guarantee. Historical Run 003/004 scores and artifacts remain unchanged. Retain `research_only_no_promotion` regardless of the outcome; the separate prospective pilot has not started.

## Kaggle handoff and execution limits

Notebook: `notebooks/07_Kaggle_Inference_Check.ipynb`, **Retry001**. Add the original snapshot Dataset plus the completed `QuantStream_Run004_Output.zip` as another input (raw or automatically expanded), use Tesla T4 and Internet, start a fresh session and **Run All once**. There is no `MODE` variable or manual train-cell handoff. Output is `QuantStream_Run005_Retry001_Output.zip`; the new working folder preserves the earlier setup-failure evidence. Preserve/send partial evidence or traceback on failure.

Local checks use a small randomly initialized Chronos model with synthetic fixture adapter weights: reload through PEFT without merging, all eight mode/RNG passes with zero fitting, unchanged-state checks and partial-evidence guards. These checks do not claim a full pretrained Kaggle run or a Run 005 research result. The full Kaggle output and immutable saved notebook-version ID remain outstanding.
