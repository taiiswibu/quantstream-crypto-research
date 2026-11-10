"""Lightweight feature drift and delayed predictive-quality monitoring."""
import numpy as np
import pandas as pd
from .features import FEATURES


def drift_reference(frame):
    output = {}
    for feature in FEATURES:
        values = frame[feature].dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, edges, np.inf])
        output[feature] = {"edges": edges.tolist(), "proportions": (counts / counts.sum()).tolist()}
    return output


def population_stability(frame, reference):
    report = []
    for feature, ref in reference.items():
        values = frame[feature].dropna().to_numpy()
        if not len(values):
            continue
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, ref["edges"], np.inf])
        observed = np.clip(counts / counts.sum(), 1e-6, 1)
        expected = np.clip(ref["proportions"], 1e-6, 1)
        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        report.append({"feature": feature, "psi": psi, "review": psi >= 0.25})
    return sorted(report, key=lambda x: x["psi"], reverse=True)


def prediction_quality(samples, history, kind):
    if not history:
        return {"matured_predictions": 0, "status": "insufficient_data"}
    pred = pd.DataFrame(history)
    pred["decision_time"] = pd.to_datetime(pred["decision_time"], utc=True)
    known = samples[["symbol", "decision_time", "target_return"]].dropna()
    joined = pred.merge(known, on=["symbol", "decision_time"], how="inner", validate="one_to_one")
    if not len(joined):
        return {"matured_predictions": 0, "status": "waiting_for_future_labels"}
    threshold = 0.5 if kind == "logistic" else 0
    return {"matured_predictions": len(joined), "status": "early_sample" if len(joined) < 30 else "available",
            "directional_accuracy": float(np.mean((joined.prediction > threshold) == (joined.target_return > 0))),
            "mae_return": None if kind == "logistic" else float(np.mean(np.abs(joined.prediction - joined.target_return)))}
