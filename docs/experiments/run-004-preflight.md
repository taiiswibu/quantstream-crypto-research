# Run 004 — Kaggle GPU preflight

Status: **preflight evidence reviewed; environment lock frozen; three research fits not started**.

The supplied `QuantStream_Run004_Preflight.zip` records a completed synthetic-only Chronos-2 LoRA smoke on Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, Chronos 2.3.2 and PEFT 0.21.2. The trainer records one optimizer step, finite loss 0.18055895 and finite gradient norm 0.95477563. This is a sin/cos environment check, not a financial forecast score or a trading result.

## Evidence and frozen execution

- ZIP SHA-256: `1e1f37e8970e7f924cee940f254a0fc109ec46a5e4e8fa51ad714be637829eef`.
- [Reviewed lock](../../reports/experiments/run-004-seed-stability/kaggle-preflight-lock.json), preserved byte-for-byte from `candidate-lock.json`; SHA-256 `f44720e43fd6404e3b95b7fa8481f61b03ad96cb2ab6af34f86406974c52474d`.
- [Verification record](../../reports/experiments/run-004-seed-stability/preflight-verification.json).
- Keep execution source at **`bb47737b5bc400169a8bbd75f8c7534503ba18d5`**, as pinned by notebook 06. Its code-file hashes and protocol SHA-256 `32d6cb80433ca147d799e17ddd9d45ae9e383ff9b17b21dab1096202b6722441` match the Git objects. Later documentation updates do not change the pinned execution source or the scientific budget.
- The lock records all 937 installed package versions, GPU/CUDA/cuDNN, model revision, resolved LoRA configuration and data/sample/training identities. The environment inventory agrees with the lock and its exported version list.

ZIP integrity, both manifest byte hashes, model-config hash and adapter file hashes were recomputed. All 194 adapter tensors (1,206,912 stored parameters) are finite; 73 of the 97 LoRA B tensors contain nonzero values after the recorded step. The serialized adapter matches rank 8, alpha 16 and the declared target modules. TrainerState and TrainingArguments agree on one step, seed 0, batch 8 and learning rate 1e-5. The lock reports finite smoke forecasts and eval mode.

The 512-row sample hash was independently recomputed from the Run 003 B prediction artifact: **`c024cd77d9f37fca3909c97b682b560d466c3f203e77d263794d52bea445a28c`**, matching preflight. Current/reference scientific manifest fields and the four recorded Parquet hashes agree. Raw current Parquet bytes and numerical smoke forecasts are not included in this ZIP; their runtime checks are evidenced by the pinned code and recorded output, not independently rerun here. Base weights and live GPU execution were not independently observed.

## Start the research fits

In the same Kaggle session, keep the existing source checkout, snapshot and preflight directory. Run this complete cell on its own, replacing the notebook-URL placeholder with the actual URL. It calls the research runner directly and does not depend on `MODE` or on rerunning an earlier cell:

```python
from quantstream.common import sha256
from quantstream.seed_lab import run_seed_lab, archive_evidence
from IPython.display import display, FileLink

reviewed_locks = {
    'f44720e43fd6404e3b95b7fa8481f61b03ad96cb2ab6af34f86406974c52474d',
    'b8e06d61a24e3f42b067d0877f67dfd2f166a84e52a540539730a5ccad1a3900',
}
assert sha256(LOCK) in reviewed_locks, 'Preflight lock has not been reviewed'
NOTEBOOK_URL = 'DAN_URL_NOTEBOOK_KAGGLE_THAT_VAO_DAY'
TRAIN_ZIP = WORK.parent / 'QuantStream_Run004_Output.zip'
if OUTPUT.exists() or TRAIN_ZIP.exists():
    raise FileExistsError('Preserve existing research evidence for review')
try:
    result = run_seed_lab(SNAPSHOT, OUTPUT, LOCK, NOTEBOOK_URL)
    print('Completed all three research seeds')
finally:
    saved = archive_evidence(OUTPUT, TRAIN_ZIP)
    if saved:
        display(FileLink(saved))
```

The setup/source cells are not needed in the already initialized session. The runner validates the actual environment/source/data/LoRA against the reviewed lock, records the notebook URL and freezes `plan.json` before its first fit. If the session has been lost, restore the saved preflight evidence and pinned source before attempting a fit; never manufacture a replacement lock or erase earlier attempts.

Budget remains seeds **42/123/2026**, **200 steps each**, learning rate **1e-5**, context **128**, batch **8** and the same **512** development windows. Each seed reloads the pretrained checkpoint. Download `QuantStream_Run004_Output.zip` after the run, including partial evidence if it fails. No seed replacement or automatic retry is authorized by this preflight review.

The notebook URL/version is absent from preflight and must be recorded before the research fits. The lock is a version inventory, not a container image or a guarantee of bitwise reproduction. There is no new independent holdout, PnL result or model promotion; retain **`research_only_no_promotion`**.

## Repeated preflight — 2026-11-12

The second supplied ZIP again records `scope=preflight_only`, `global_step=1`, `research_fits_started=0`. Its source, environment, data/sample/training identities, LoRA configuration and research budget are identical to the first reviewed lock. The smoke safetensors bytes are also identical. Only the serialized ordering of the same `target_modules` names in `adapter_config.json` differs, changing that file's hash and therefore the raw lock hash. This is not another financial trial or an additional seed result.

The second [lock](../../reports/experiments/run-004-seed-stability/kaggle-preflight-lock-rerun-001.json) is preserved as received, SHA-256 **`b8e06d61a24e3f42b067d0877f67dfd2f166a84e52a540539730a5ccad1a3900`**. Its [verification record](../../reports/experiments/run-004-seed-stability/preflight-rerun-001-verification.json) retains the original lock hash and repeated ZIP hash. Both reviewed locks are accepted by the handoff cell; runtime validation still requires the exact environment, pinned source, data and scientific settings in the selected lock. The original lock and evidence remain unchanged.
