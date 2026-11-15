"""Fixed final experiment: balanced families, raw-return scoring and honest acceptance."""

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from .common import Config, git_provenance, sha256, utcnow, write_json, write_parquet
from .features import fold_data
from . import workflow


def conditions(epochs=96, minimum_epochs=24):
    result = []
    for family in ("ridge", "xgboost", "tcn", "gru"):
        for transform in ("none", "volatility"):
            common = {
                "family": family,
                "target_transform": transform,
                "forecast_horizon": 4,
            }
            if family == "ridge":
                params = {
                    **common,
                    "alpha": 1000.0,
                    "fit_intercept": transform == "none",
                }
            elif family == "xgboost":
                params = {
                    **common,
                    "max_depth": 2,
                    "learning_rate": 0.03,
                    "n_estimators": 200,
                    "reg_lambda": 100.0,
                    "min_child_weight": 500,
                }
            else:
                params = {
                    **common,
                    "lookback": 48,
                    "width": 32,
                    "dropout": 0.15,
                    "lr": 0.0003,
                    "weight_decay": 0.001,
                    "batch_size": 256,
                    "loss": "mse",
                    "lr_schedule": "cosine",
                    "schedule_epochs": epochs,
                    "epoch_cap": epochs,
                    "min_epochs": minimum_epochs,
                    "patience": 12,
                }
                if family == "tcn":
                    params["activation"] = "silu"
            result.append(params)
    return result


def weekly_interval(predictions, seed=42, repeats=1000):
    """Resample UTC week clusters jointly across assets, separately inside each fold."""
    data = predictions.copy()
    data["week"] = data.decision_time.dt.as_unit("ns").astype("int64") // (
        7 * 24 * 3600 * 10**9
    )
    data["difference"] = (
        data.prediction - data.target_return
    ) ** 2 - data.target_return**2
    data["zero_loss"] = data.target_return**2
    groups = [
        g.groupby("week")[["difference", "zero_loss"]].sum().to_numpy()
        for _, g in data.groupby("fold")
    ]
    if not groups or any(len(g) < 8 for g in groups):
        return {"available": False, "reason": "Fewer than eight week clusters per fold"}
    rng = np.random.default_rng(seed)
    values = np.zeros(repeats)
    for g in groups:
        sample = g[rng.integers(0, len(g), size=(repeats, len(g)))].sum(axis=1)
        values += sample[:, 0] / np.maximum(sample[:, 1], 1e-15) / len(groups)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "available": True,
        "lower": float(lower),
        "upper": float(upper),
        "repeats": repeats,
        "cluster": "UTC weeks joint across assets within each fold",
        "statistic": "Mean fold relative MSE difference to zero; negative is better",
        "limitation": "Approximate diagnostic after reused validation; not an independent significance test",
    }


def acceptance(reports, predictions):
    ratios = np.array(
        [[f["prediction"]["mse_ratio_to_zero"] for f in r["folds"]] for r in reports]
    )
    interval = weekly_interval(predictions)
    passes = bool(
        ratios.mean() < 1
        and (ratios.mean(axis=0) < 1).all()
        and (ratios.mean(axis=1) < 1).all()
        and interval["available"]
        and interval["upper"] < 0
    )
    return {
        "mean_seed_fold_mse_ratio": float(ratios.mean()),
        "worst_fold_mean_mse_ratio": float(ratios.mean(axis=0).max()),
        "worst_seed_mean_mse_ratio": float(ratios.mean(axis=1).max()),
        "beats_zero_on_mean_and_every_fold": bool(
            ratios.mean() < 1 and (ratios.mean(axis=0) < 1).all()
        ),
        "weekly_interval_seed42": interval,
        "signal_approved": passes,
        "usage": "paper_only",
        "test_is_independent": False,
    }


def _export(root, fitted, cfg, manifest, selection, test_report, context, role):
    Path(root).mkdir(parents=True, exist_ok=True)
    path = Path(
        workflow.export_model(
            root,
            fitted,
            cfg,
            manifest,
            selection,
            test_report,
            context,
            run_name="run008",
        )
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        artifact_role=role,
        deployment_status=(
            "paper_candidate_approved"
            if selection["signal_approved"]
            else "research_validation_failed"
        ),
        validation_scope="Configuration CV; weights differ between folds and deployment refit",
        forecast_contract_sha256=sha256(
            Path(__file__).with_name("forecast_contract.py")
        ),
    )
    write_json(metadata_path, metadata)
    write_json(
        path / "bundle_manifest.json",
        {
            "format": "quantstream-notebook-v1",
            "files": {
                p.name: sha256(p)
                for p in path.iterdir()
                if p.is_file() and p.name != "bundle_manifest.json"
            },
        },
    )
    return path


def run(root, epochs=96, minimum_epochs=24, seconds=10800, device="auto"):
    import torch
    from .search import with_context
    from .backtest import evaluate_strategy, simulate

    root = Path(root)
    if not 1 <= minimum_epochs <= epochs <= 120 or not 60 <= seconds <= 14400:
        raise ValueError(
            "Final budget: 1 <= minimum <= epochs <= 120, 60..14400 seconds"
        )
    if (root / "training_plan.json").exists():
        raise ValueError(
            "Final run already started; preserve evidence instead of overwriting"
        )
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected:
            raise ValueError("Dataset changed: " + name)
    if manifest["feature_code_sha256"] != sha256(
        Path(__file__).with_name("features.py")
    ):
        raise ValueError("Feature source differs from the frozen snapshot")
    cfg = Config(
        **{**manifest["config"], "symbols": tuple(manifest["config"]["symbols"])}
    )
    if cfg.horizon != 4 or len(manifest["split"]["tuning_folds"]) != 3:
        raise ValueError(
            "Final protocol requires the frozen four-hour target and three folds"
        )
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    candidates = conditions(epochs, minimum_epochs)
    plan = {
        "protocol": "run008-final-v1",
        "conditions": candidates,
        "seeds": [42, 123, 2026],
        "epochs": epochs,
        "minimum_epochs": minimum_epochs,
        "seconds": seconds,
        "dataset_manifest_sha256": sha256(root / "dataset_manifest.json"),
        "source": git_provenance(Path(__file__).resolve().parents[1]),
        "source_files": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "runtime": workflow.runtime_record(device),
        "test_is_independent": False,
        "rules": "Eight balanced conditions; one finalist per family; seed42 export; zero baseline retained; no test-driven selection",
    }
    with (root / "training_plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    write_json(root / "status.json", {"status": "running", "started_at": utcnow()})
    budget = workflow.Budget(seconds)
    started = time.monotonic()
    features = pd.read_parquet(root / "data/processed/features.parquet")
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    split = manifest["split"]
    dev = features[
        features.target_return.notna()
        & features.label_end_time.lt(pd.Timestamp(split["test_start"]))
    ]
    trial_rows, reports = [], []
    try:
        for number, params in enumerate(candidates):
            print(
                f"Condition {number + 1}/{len(candidates)} · {params['family']} · {params['target_transform']}",
                flush=True,
            )
            report = workflow.evaluate(
                dev,
                bars,
                split,
                params,
                42,
                epochs,
                cfg,
                device,
                budget,
                root / "trials" / f"{number:03}",
            )
            reports.append(report)
            trial_rows.append(
                {
                    "number": number,
                    "value": report["objective"],
                    "state": "COMPLETE",
                    "params_family": params["family"],
                    "params_target_transform": params["target_transform"],
                    "user_attrs_params": json.dumps(params),
                }
            )
            pd.DataFrame(trial_rows).to_csv(root / "trials.csv", index=False)
            write_json(
                root / "progress.json",
                {
                    "stage": "balanced_conditions",
                    "completed": number + 1,
                    "planned": len(candidates),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        leaders = []
        for family in ("ridge", "xgboost", "tcn", "gru"):
            ids = [i for i, p in enumerate(candidates) if p["family"] == family]
            best = min(ids, key=lambda i: (reports[i]["objective"], i))
            seed_reports = [reports[best]]
            for seed in (123, 2026):
                folder = root / "stability" / f"{best:03}-seed-{seed}"
                if family == "ridge":
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        root / "trials" / f"{best:03}" / "predictions.parquet",
                        folder / "predictions.parquet",
                    )
                    write_json(
                        folder / "report.json",
                        {
                            **reports[best],
                            "seed": seed,
                            "reused_deterministic_fit": True,
                        },
                    )
                    seed_reports.append(reports[best])
                else:
                    seed_reports.append(
                        workflow.evaluate(
                            dev,
                            bars,
                            split,
                            candidates[best],
                            seed,
                            epochs,
                            cfg,
                            device,
                            budget,
                            folder,
                        )
                    )
            predictions = pd.read_parquet(
                root / "trials" / f"{best:03}" / "predictions.parquet"
            )
            gate = acceptance(seed_reports, predictions)
            leaders.append(
                {
                    "trial": best,
                    "params": candidates[best],
                    "mean_objective": gate["mean_seed_fold_mse_ratio"],
                    "std_objective": float(
                        np.std([r["objective"] for r in seed_reports])
                    ),
                    "validation_gate": gate,
                }
            )
            write_json(
                root / "progress.json",
                {
                    "stage": "seed_stability",
                    "completed_families": len(leaders),
                    "planned_families": 4,
                },
            )
        accepted = [
            item for item in leaders if item["validation_gate"]["signal_approved"]
        ]
        selected = min(
            accepted or leaders,
            key=lambda item: (item["mean_objective"], item["trial"]),
        )
        cv = pd.read_parquet(
            root / "trials" / f"{selected['trial']:03}" / "predictions.parquet"
        )
        selection = {
            **selected,
            "shortlist": leaders,
            "seed": 42,
            "signal_approved": bool(accepted),
            "recommended_model": "learned_candidate"
            if accepted
            else "zero_return_no_signal",
            "validation": workflow.forecast_metrics(cv),
            "frozen_at": utcnow(),
        }
        write_json(root / "selection.json", selection)
        with (root / "retrospective-test.started.json").open("x") as stream:
            json.dump(
                {
                    "selection_sha256": sha256(root / "selection.json"),
                    "started_at": utcnow(),
                },
                stream,
            )
        audit_root = root / "audit_model"
        write_json(
            root / "progress.json",
            {
                "stage": "audit_fit",
                "note": "Selection frozen; fitting historical audit weights",
            },
        )
        fitted = workflow.fit_model(
            dev,
            selected["params"],
            42,
            epochs,
            cfg,
            device,
            budget,
            audit_root / "final-fit",
        )
        _, test = fold_data(features, split["test_start"], split["end"])
        context = with_context(features, test, split["test_start"])
        values = workflow.predict(fitted, context)
        mask = (context.is_validation & context.eligible_window).to_numpy()
        if not np.isfinite(values[mask]).all():
            raise ValueError("Missing retrospective forecasts")
        rows = context[mask].copy().assign(prediction=values[mask])
        write_parquet(
            root / "retrospective_predictions.parquet",
            rows[["symbol", "decision_time", "target_return", "prediction"]],
        )
        metrics = [
            {"model": "selected_audit_weights", **workflow.forecast_metrics(rows)},
            {
                "model": "zero_return",
                **workflow.forecast_metrics(rows.assign(prediction=0.0)),
            },
        ]
        write_json(root / "retrospective_metrics.json", metrics)
        write_json(
            root / "retrospective_asset_metrics.json",
            [
                {"symbol": s, **workflow.forecast_metrics(g)}
                for s, g in rows.groupby("symbol")
            ],
        )
        curve, trades, strategy = evaluate_strategy(
            bars, rows, rows.prediction, 0.006, split["test_start"], split["end"], cfg
        )
        empty = pd.DataFrame(columns=["symbol", "decision_time", "go_long"])
        hold, _, hold_metrics = simulate(
            bars, empty, split["test_start"], split["end"], cfg, buy_hold=True
        )
        ema, _, ema_metrics = evaluate_strategy(
            bars, rows, rows.ema_gap, 0, split["test_start"], split["end"], cfg
        )
        write_parquet(
            root / "retrospective_equity.parquet",
            pd.DataFrame(
                {
                    "timestamp": curve.index,
                    "model": curve.values,
                    "buy_hold": hold.values,
                    "ema": ema.values,
                }
            ),
        )
        write_parquet(root / "retrospective_trades.parquet", trades)
        write_json(
            root / "retrospective_strategies.json",
            {"model": strategy, "buy_hold": hold_metrics, "ema": ema_metrics},
        )
        _export(
            audit_root,
            fitted,
            cfg,
            manifest,
            selection,
            metrics[0],
            context,
            "historical_audit_weights",
        )
        del fitted
        # Refit once on all matured labels for fresh inference; old test scores do not describe these weights.
        all_observed = features[
            features.target_return.notna() & features.label_end_time.notna()
        ].copy()
        write_json(
            root / "progress.json",
            {
                "stage": "deployment_refit",
                "note": "Refitting once on all matured observations; test metrics will be empty for these weights",
            },
        )
        fitted = workflow.fit_model(
            all_observed,
            selected["params"],
            42,
            epochs,
            cfg,
            device,
            budget,
            root / "final-fit",
        )
        deployment = _export(
            root,
            fitted,
            cfg,
            manifest,
            selection,
            None,
            features,
            "all_observed_research_refit_no_independent_test",
        )
        zero = {
            "model": None,
            "params": {"family": "zero_return"},
            "epochs": None,
            "training_rows": 0,
            "device": "cpu",
        }
        zero_selection = {
            **selection,
            "signal_approved": False,
            "validation": workflow.forecast_metrics(cv.assign(prediction=0.0)),
            "validation_gate": {
                "mean_seed_fold_mse_ratio": 1.0,
                "worst_fold_mean_mse_ratio": 1.0,
                "beats_zero_on_mean_and_every_fold": False,
                "usage": "untrained_reference",
            },
        }
        _export(
            root / "zero_reference",
            zero,
            cfg,
            manifest,
            zero_selection,
            metrics[1],
            context,
            "untrained_zero_return_reference",
        )
        budget.check()
        write_json(
            root / "status.json",
            {
                "status": "completed",
                "completed_at": utcnow(),
                "elapsed_seconds": time.monotonic() - started,
                "conditions_completed": len(candidates),
                "deployment": str(deployment),
                "validation_approved": bool(accepted),
                "recommended_model": selection["recommended_model"],
                "test_is_independent": False,
                "promotion": "paper_only_if_approved_otherwise_forecast_monitoring",
            },
        )
        return selection
    except BaseException as error:
        write_json(
            root / "status.json",
            {"status": "incomplete", "error": str(error), "at": utcnow()},
        )
        (root / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--minimum-epochs", type=int, default=24)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.root, args.epochs, args.minimum_epochs, args.seconds, args.device)
