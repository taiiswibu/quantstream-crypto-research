"""Matched daily-origin interval comparison; all empirical inputs are matured."""

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
WORK = Path(os.environ["QUANTSTREAM_MODERN_OUTPUT"])
ROOT = Path(os.environ["QUANTSTREAM_SNAPSHOT"])
OUT = WORK / "intervals"
OUT.mkdir(exist_ok=True)
features = pd.read_parquet(ROOT / "data/processed/features.parquet")
pred = pd.read_parquet(WORK / "foundation/tirex_plain.parquet")
ridge = pd.read_parquet(WORK / "enriched/ridge-original.parquet")
methods = []
for symbol, group in pred.groupby("symbol"):
    raw = features[features.symbol.eq(symbol)].sort_values("decision_time")
    for row in group.itertuples():
        t = row.decision_time
        current = raw[raw.decision_time.eq(t)].iloc[0]
        sigma = float(np.clip(current.vol_48 * 2, 0.001, 0.05))
        history = raw[
            (raw.label_end_time < t) & (raw.decision_time >= t - pd.Timedelta(days=28))
        ].dropna(subset=["target_return"])
        if len(history) < 168:
            raise ValueError("Not enough mature interval history")
        normalized = history.target_return.to_numpy() / np.clip(
            history.vol_48.to_numpy() * 2, 0.001, 0.05
        )
        ages = (t - history.decision_time).dt.total_seconds().to_numpy() / 3600
        weights = 2 ** (-ages / 168)
        order = np.argsort(normalized)
        positions = (np.cumsum(weights[order]) - 0.5 * weights[order]) / weights.sum()
        weighted = np.interp([0.1, 0.9], positions, normalized[order]) * sigma
        empirical = np.quantile(normalized, [0.1, 0.9]) * sigma
        rp = ridge[ridge.symbol.eq(symbol) & ridge.decision_time.eq(t)].prediction.iloc[
            0
        ]
        choices = {
            "tirex2": (row.q10, row.q90, row.prediction),
            "gaussian_volatility": (
                -1.2815515655446004 * sigma,
                1.2815515655446004 * sigma,
                0.0,
            ),
            "rolling_empirical": (*empirical, 0.0),
            "weighted_empirical": (*weighted, 0.0),
            "ridge_point": (np.nan, np.nan, rp),
        }
        for method, (lo, hi, point) in choices.items():
            y = row.target_return
            score = (
                (hi - lo) + 10 * max(lo - y, 0) + 10 * max(y - hi, 0)
                if np.isfinite(lo)
                else np.nan
            )
            methods.append(
                {
                    "symbol": symbol,
                    "decision_time": t,
                    "fold": row.fold,
                    "target_return": y,
                    "method": method,
                    "prediction": point,
                    "q10": lo,
                    "q90": hi,
                    "covered": float(bool(lo <= y <= hi))
                    if np.isfinite(lo)
                    else np.nan,
                    "width": hi - lo,
                    "interval_score": score,
                }
            )
frame = pd.DataFrame(methods)
frame.to_parquet(OUT / "predictions.parquet", index=False)
summary = []
for method, g in frame.groupby("method"):
    ratios = [
        float(
            ((x.prediction - x.target_return) ** 2).mean() / (x.target_return**2).mean()
        )
        for _, x in g.groupby("fold")
    ]
    summary.append(
        {
            "method": method,
            "rows": len(g),
            "coverage": float(g.covered.mean()) if g.covered.notna().any() else None,
            "mean_width": float(g.width.mean()) if g.width.notna().any() else None,
            "interval_score": float(g.interval_score.mean())
            if g.interval_score.notna().any()
            else None,
            "mean_fold_mse_ratio": float(np.mean(ratios)),
            "fold_ratios": ratios,
        }
    )
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
