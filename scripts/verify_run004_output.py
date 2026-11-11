"""Audit the supplied Run 004 ZIP without training; emit inspectable calculations.

Usage: python scripts/verify_run004_output.py OUTPUT.zip --run003-zip RUN003.zip
The caller supplies original artifacts; no model weights are copied into Git.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
from safetensors.numpy import load as load_tensors

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sample_hash(frame):
    rows = frame[["symbol", "decision_time", "target_return"]].copy()
    rows["decision_time"] = pd.to_datetime(rows.decision_time, utc=True).astype(str)
    return digest(rows.to_json(orient="split", index=False, double_precision=15).encode())


def measures(truth, prediction):
    errors = prediction - truth
    mae0, mse0 = np.mean(np.abs(truth)), np.mean(np.square(truth))
    mae, mse = np.mean(np.abs(errors)), np.mean(np.square(errors))
    return {"rows": len(truth), "mae": float(mae), "mse": float(mse),
            "mae_ratio_to_zero": float(mae / mae0), "mse_ratio_to_zero": float(mse / mse0),
            "directional_accuracy": float(np.count_nonzero((prediction > 0) == (truth > 0)) / len(truth))}


def check_numbers(actual, expected):
    for key, value in actual.items():
        if isinstance(value, (int, float)):
            assert np.isclose(value, expected[key], rtol=1e-12, atol=1e-15), (key, value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])


def audit(archive_path, output, run003_zip=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        def read(name):
            return json.loads(archive.read(name))
        plan, comparison, status = read("plan.json"), read("comparison.json"), read("status.json")
        assert status["status"] == "completed" and status["completed_fits"] == 3
        assert status["total_optimizer_steps"] == 600
        assert plan["seeds"] == list(SEEDS) and plan["steps"] == 200 and plan["learning_rate"] == 1e-5
        assert (plan["context_length"], plan["batch_size"], plan["max_windows"], plan["inference_mode"]) == (128, 8, 512, "eval")
        lock_path = ROOT / "reports/experiments/run-004-seed-stability/kaggle-preflight-lock-rerun-001.json"
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        assert digest(lock_bytes) == plan["preflight_lock_sha256"]
        assert {key: plan[key] for key in lock} == lock
        assert plan["source"]["dirty"] is False
        commit = plan["source"]["commit"]
        for name, expected in plan["source"]["files"].items():
            raw = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + name])
            assert digest(raw) == expected, name
        protocol = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                            commit + ":docs/experiments/run-004-seed-stability-protocol.md"])
        assert digest(protocol) == plan["source"]["protocol_sha256"]

        frames, evidence, losses = [], [], []
        for seed in SEEDS:
            prefix = f"seed-{seed}/"
            state, args, report, fit_status = [read(prefix + name) for name in
                ("trainer_state.json", "training_args.json", "chronos_report.json", "status.json")]
            assert fit_status["status"] == "completed" and fit_status["seed"] == seed
            assert state["global_step"] == state["max_steps"] == args["max_steps"] == fit_status["global_step"] == 200
            assert args["seed"] == seed and args["per_device_train_batch_size"] == 8 and args["learning_rate"] == 1e-5
            assert args["use_cpu"] is False and args["logging_steps"] == 1
            assert args["output_dir"].endswith(f"seed-{seed}/trainer")
            assert report["seed"] == seed and report["steps"] == 200 and report["mode"] == "lora"
            assert report["inference_mode"] == "eval" and report["evaluation_windows"] == 512
            assert report["resolved_revision"] == plan["revision"] and report["model_id"] == plan["model_id"]
            assert report["snapshot_manifest_sha256"] == plan["inputs"]["manifest_sha256"]
            assert digest(archive.read(prefix + "training_snapshot_manifest.json")) == plan["inputs"]["manifest_sha256"]
            raw_predictions = archive.read(prefix + "chronos_validation.parquet")
            assert digest(raw_predictions) == fit_status["prediction_sha256"]
            frame = pd.read_parquet(io.BytesIO(raw_predictions))
            assert len(frame) == 512 and frame.groupby("symbol").size().to_dict() == {"BTCUSDT": 256, "ETHUSDT": 256}
            assert not frame.duplicated(["symbol", "decision_time"]).any()
            assert np.isfinite(frame[["target_return", "zero_shot_return", "finetuned_return"]].to_numpy()).all()
            assert sample_hash(frame) == plan["inputs"]["sample_sha256"]
            if frames:
                pd.testing.assert_frame_equal(frame[["symbol", "decision_time", "target_return"]],
                                              frames[0][["symbol", "decision_time", "target_return"]])
                assert np.array_equal(frame.zero_shot_return, frames[0].zero_shot_return)
            frames.append(frame)
            history = [row for row in state["log_history"] if "loss" in row]
            assert [row["step"] for row in history] == list(range(1, 201))
            assert all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"]) for row in history)
            losses.extend({"seed": seed, **row} for row in history)
            config = read(prefix + "adapter/adapter_config.json")
            for key in ("r", "lora_alpha", "lora_dropout", "bias", "use_dora", "use_rslora", "lora_bias"):
                assert config[key] == plan["lora"][key]
            assert sorted(config["target_modules"]) == sorted(plan["lora"]["target_modules"])
            for name, expected in fit_status["adapter"]["files"].items():
                assert digest(archive.read(prefix + "adapter/" + name)) == expected
            weights = archive.read(prefix + "adapter/adapter_model.safetensors")
            tensors = load_tensors(weights)
            assert len(tensors) == 194
            assert sum(array.size for array in tensors.values()) == 1206912
            assert all(np.isfinite(array).all() for array in tensors.values())
            tensor_digest = hashlib.sha256()
            for key in sorted(tensors):
                tensor_digest.update(key.encode())
                tensor_digest.update(tensors[key].tobytes())
            current = measures(frame.target_return.to_numpy(), frame.finetuned_return.to_numpy())
            assert np.isclose(current["mae"], report["finetuned_mae"], rtol=1e-12, atol=1e-15)
            assert current["directional_accuracy"] == report["finetuned_directional_accuracy"]
            assert pd.Timestamp(plan["created_at"]) < pd.Timestamp(fit_status["at"]) <= pd.Timestamp(status["at"])
            evidence.append({"seed": seed, "global_step": state["global_step"], "logged_loss_steps": len(history),
                "adapter_sha256": digest(weights), "adapter_tensors": 194, "adapter_stored_parameters": 1206912,
                "adapter_tensor_values_sha256": tensor_digest.hexdigest(),
                "prediction_sha256": digest(raw_predictions),
                "first20_mean_training_loss": float(np.mean([row["loss"] for row in history[:20]])),
                "last20_mean_training_loss": float(np.mean([row["loss"] for row in history[-20:]])),
                "training_seconds_recorded": state["log_history"][-1]["train_runtime"]})
        assert len({entry["adapter_sha256"] for entry in evidence}) == 3
        assert len({entry["adapter_tensor_values_sha256"] for entry in evidence}) == 3

        # Recompute every baseline, asset and seed row directly, without the production summary helper.
        reference = frames[0]
        records = []
        for asset in ("ALL", "BTCUSDT", "ETHUSDT"):
            mask = np.ones(len(reference), dtype=bool) if asset == "ALL" else reference.symbol.eq(asset).to_numpy()
            truth, baseline = reference.target_return.to_numpy()[mask], reference.zero_shot_return.to_numpy()[mask]
            for model, prediction in (("zero_return", np.zeros(len(truth))), ("zero_shot", baseline)):
                records.append({"asset": asset, "model": model, "seed": None, **measures(truth, prediction)})
            for seed, frame in zip(SEEDS, frames):
                row = {"asset": asset, "model": "lora_B", "seed": seed,
                       **measures(truth, frame.finetuned_return.to_numpy()[mask])}
                row["mae_delta_vs_zero_shot"] = row["mae"] - np.mean(np.abs(baseline - truth))
                records.append(row)
        assert len(records) == len(comparison["metrics"]) == 15
        for actual, expected in zip(records, comparison["metrics"]):
            check_numbers(actual, expected)
        csv = pd.read_csv(io.BytesIO(archive.read("metrics.csv")))
        numeric = ["rows", "mae", "mse", "mae_ratio_to_zero", "mse_ratio_to_zero", "directional_accuracy"]
        csv_keys = [(row.asset, row.model, None if pd.isna(row.seed) else int(row.seed)) for row in csv.itertuples()]
        assert csv_keys == [(row["asset"], row["model"], row["seed"]) for row in records]
        assert np.allclose(csv[numeric].to_numpy(), pd.DataFrame(records)[numeric].to_numpy(), rtol=1e-12, atol=1e-15)
        for expected in comparison["seed_summary"]:
            selected = [row for row in records if row["asset"] == expected["asset"] and row["model"] == "lora_B"]
            for key in ("mae", "mse", "directional_accuracy", "mae_delta_vs_zero_shot"):
                check_numbers({"mean": np.mean([row[key] for row in selected]),
                               "std": np.std([row[key] for row in selected], ddof=1)}, expected[key])

        # Independently resample actual row-index blocks, rather than using the production block-total implementation.
        truth = reference.target_return.to_numpy()
        delta = np.mean([np.abs(frame.finetuned_return.to_numpy() - truth) for frame in frames], axis=0)
        delta -= np.abs(reference.zero_shot_return.to_numpy() - truth)
        time = pd.DatetimeIndex(pd.to_datetime(reference.decision_time, utc=True))
        week = time.normalize() - pd.to_timedelta(time.dayofweek, unit="D")
        weeks = sorted(week.unique())
        blocks = [np.flatnonzero(week == value) for value in weeks]
        rng = np.random.default_rng(2026)
        bootstrap = []
        for _ in range(2000):
            picked = rng.integers(0, len(blocks), len(blocks))
            indices = np.concatenate([blocks[index] for index in picked])
            bootstrap.append(float(np.mean(delta[indices])))
        interval = np.percentile(bootstrap, [2.5, 97.5])
        assert len(blocks) == comparison["paired_mae_bootstrap"]["blocks"] == 22
        assert np.allclose(interval, comparison["paired_mae_bootstrap"]["percentile_95"], rtol=1e-12, atol=1e-15)
        assert np.isclose(delta.mean(), comparison["paired_mae_bootstrap"]["estimate"], rtol=1e-12, atol=1e-15)
        assert comparison["deployment_status"] == "research_only_no_promotion"
        assert comparison["diagnostic_decision"] == "keep_for_further_research"

        historical = {"checked": False}
        if run003_zip:
            with zipfile.ZipFile(run003_zip) as old:
                old_weights = old.read("B_steps200_lr1e5/adapter/adapter_model.safetensors")
                old_rows = pd.read_parquet(io.BytesIO(old.read("B_steps200_lr1e5/chronos_validation.parquet")))
            pd.testing.assert_frame_equal(old_rows[["symbol", "decision_time", "target_return"]],
                                          reference[["symbol", "decision_time", "target_return"]])
            assert np.array_equal(old_rows.zero_shot_return, reference.zero_shot_return)
            assert digest(old_weights) == evidence[0]["adapter_sha256"]
            historical = {"checked": True, "run003_B_adapter_same_bytes_as_seed42": True,
                "same_sample_targets_and_zero_shot_predictions": True,
                "run003_B_mae": float(np.mean(np.abs(old_rows.finetuned_return.to_numpy() - truth))),
                "run004_seed42_mae": records[2]["mae"],
                "interpretation": "The learned adapter did not change. Historical score differences concern inference/evaluation, not a better learned seed-42 adapter; exact causal effects were not isolated."}
        selected = [row for row in records if row["asset"] == "ALL" and row["model"] == "lora_B"]
        zero_mae, zero_shot_mae = records[0]["mae"], records[1]["mae"]
        mean_mae = float(np.mean([row["mae"] for row in selected]))
        verification = {"status": "completed_verified_development_research_only", "artifact_sha256": digest(Path(archive_path).read_bytes()),
            "execution_source_commit": commit, "preflight_lock_sha256": plan["preflight_lock_sha256"],
            "notebook_url_recorded": plan["notebook_url"], "plan_created_at_utc": plan["created_at"],
            "finished_at_utc": status["at"], "steps_evidenced": 600, "seed_evidence": evidence,
            "unique_evaluation_windows": 512, "forecast_rows_across_seeds": 1536,
            "first_decision_time_utc": str(reference.decision_time.min()),
            "last_decision_time_utc": str(reference.decision_time.max()),
            "rows_per_asset_per_seed": reference.groupby("symbol").size().to_dict(),
            "sample_sha256": plan["inputs"]["sample_sha256"],
            "checks": {"zip_crc_ok": True, "plan_matches_reviewed_lock": True, "source_hashes_match_git": True,
                       "all_rows_finite_unique_and_aligned": True, "baseline_identical_across_seeds": True,
                       "all_15_metric_rows_and_csv_recomputed": True, "seed_mean_std_recomputed": True,
                       "weekly_paired_bootstrap_independently_recomputed": True, "all_adapter_hashes_and_tensors_checked": True,
                       "all_600_loss_log_steps_finite": True},
            "pooled_seed_mean": {"mae": mean_mae, "mae_std_ddof1": float(np.std([row["mae"] for row in selected], ddof=1)),
                "mae_relative_change_vs_zero_shot_pct": (mean_mae / zero_shot_mae - 1) * 100,
                "mae_relative_change_vs_zero_return_pct": (mean_mae / zero_mae - 1) * 100,
                "directional_accuracy": float(np.mean([row["directional_accuracy"] for row in selected]))},
            "paired_mae_bootstrap": {"estimate": float(delta.mean()), "percentile_95": interval.tolist(),
                                     "calendar_week_blocks": len(blocks), "repetitions": 2000, "rng_seed": 2026,
                                     "scope": "Descriptive development CI conditional on the fixed three seeds, not independent alpha evidence."},
            "historical_comparison": historical, "diagnostic_decision": comparison["diagnostic_decision"],
            "deployment_status": comparison["deployment_status"],
            "limitations": ["Previously observed development sample; B was selected after Run 003.",
                "Raw market Parquet inputs, base weights and live GPU execution were not independently observed.",
                "Recorded Kaggle URL is an editable notebook path; an immutable saved-version ID is absent.",
                "Train loss alone and endpoint-only development scores do not establish overfitting timing.",
                "No new holdout, post-fee PnL or prospective result; pretraining overlap remains unverified.",
                "Three seeds and 22 resampled calendar weeks do not remove selection bias or characterize all seeds."]}
        (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        pd.DataFrame(records).to_csv(output / "metrics-verified.csv", index=False)
        pd.DataFrame(losses).to_csv(output / "training-loss-verified.csv", index=False)
        pd.DataFrame({"week_start_utc": weeks, "rows": [len(block) for block in blocks],
                      "mean_mae_delta": [float(np.mean(delta[block])) for block in blocks]}).to_csv(output / "weekly-blocks.csv", index=False)
        return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--run003-zip", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/experiments/run-004-seed-stability")
    args = parser.parse_args()
    result = audit(args.archive, args.output, args.run003_zip)
    print(json.dumps({key: result[key] for key in ("status", "steps_evidenced", "pooled_seed_mean", "paired_mae_bootstrap", "historical_comparison")}, indent=2))
