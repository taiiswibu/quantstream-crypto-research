"""Development-only benchmark of a fixed 2026 model, unchanged four-hour labels."""

from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd
import torch

from quantstream.modern_models import (
    prepare_window,
    load_tirex,
    forecast_windows,
    ASSETS,
)
from quantstream.final_research import weekly_interval

WORK = Path(os.environ["QUANTSTREAM_MODERN_OUTPUT"])

OUT = WORK / "foundation"
OUT.mkdir(exist_ok=True)
ROOT = Path(os.environ["QUANTSTREAM_SNAPSHOT"])
SPLIT = json.loads((ROOT / "split.json").read_text())
FRAME = pd.read_parquet(ROOT / "data/processed/features.parquet")
# Never let an old-test label enter any input window in this study.
FRAME = FRAME[FRAME.decision_time.lt(pd.Timestamp(SPLIT["test_start"]))].copy()
conditions = {"tirex_plain": [], "tirex_covariates": ["ret_1", "vol_48"]}
plan = {
    "model": "NX-AI/TiRex-2",
    "revision": "352636173653592b6dceb3fbeee9224ca97952a9",
    "context": 128,
    "target": "exact existing 4h return",
    "label_delay_hours": 6,
    "conditions": conditions,
    "evaluation_origin": "every day at 00:00 UTC",
    "point": "median, not mean",
    "old_test_used": False,
    "independent_test": False,
    "pretraining_contamination_not_excluded": True,
    "device": "cpu",
}
p = OUT / "plan.json"
if p.exists() and json.loads(p.read_text()) != plan:
    raise ValueError("Changed plan")
p.write_text(json.dumps(plan, indent=2))
torch.set_num_threads(4)
model = load_tirex(WORK / "models/TiRex-2")
all_results = []
started = time.time()
for name, covariates in conditions.items():
    destination = OUT / (name + ".parquet")
    results = []
    for k, fold in enumerate(SPLIT["tuning_folds"]):
        start, end = (
            pd.Timestamp(fold["validation_start"]),
            pd.Timestamp(fold["validation_end"]),
        )
        origins = pd.DatetimeIndex(
            FRAME.loc[
                FRAME.decision_time.ge(start) & FRAME.label_end_time.lt(end),
                "decision_time",
            ].unique()
        )
        origins = origins[origins.hour == 0].sort_values()
        for first in range(0, len(origins), 8):
            chunk = origins[first : first + 8]
            windows = [prepare_window(FRAME, t, covariates) for t in chunk]
            predictions = forecast_windows(model, windows, batch_size=8)
            for origin, values in zip(chunk, predictions):
                for j, asset in enumerate(ASSETS):
                    y = FRAME.loc[
                        FRAME.symbol.eq(asset) & FRAME.decision_time.eq(origin),
                        "target_return",
                    ].iloc[0]
                    results.append(
                        {
                            "symbol": asset,
                            "decision_time": origin,
                            "target_return": y,
                            "prediction": float(values[j, 4]),
                            "q10": float(values[j, 0]),
                            "q90": float(values[j, -1]),
                            "fold": k,
                        }
                    )
            pd.DataFrame(results).to_parquet(
                OUT / (name + "-partial.parquet"), index=False
            )
            print(
                name,
                "fold",
                k,
                "origin",
                min(first + 8, len(origins)),
                "/",
                len(origins),
                "seconds",
                round(time.time() - started),
                flush=True,
            )
    frame = pd.DataFrame(results)
    frame.to_parquet(destination, index=False)
    ratios = [
        float(
            ((g.prediction - g.target_return) ** 2).mean() / (g.target_return**2).mean()
        )
        for _, g in frame.groupby("fold")
    ]
    row = {
        "model": name,
        "rows": len(frame),
        "mean_fold_mse_ratio": float(np.mean(ratios)),
        "fold_ratios": ratios,
        "mae": float(abs(frame.prediction - frame.target_return).mean()),
        "direction": float(
            ((frame.prediction > 0) == (frame.target_return > 0)).mean()
        ),
        "q10_q90_empirical_coverage": float(
            frame.target_return.between(frame.q10, frame.q90).mean()
        ),
        "interval": weekly_interval(frame),
    }
    all_results.append(row)
    print(json.dumps(row), flush=True)
    (OUT / "results.json").write_text(json.dumps(all_results, indent=2))
(OUT / "status.json").write_text(
    json.dumps(
        {
            "status": "completed",
            "seconds": time.time() - started,
            "conditions": len(conditions),
        },
        indent=2,
    )
)
