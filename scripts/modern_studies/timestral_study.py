from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd
import torch

from quantstream.modern_models import (
    prepare_window,
    load_forecaster,
    forecast_timestral,
    ASSETS,
    MODELS,
)
from quantstream.final_research import weekly_interval

WORK = Path(os.environ["QUANTSTREAM_MODERN_OUTPUT"])

ROOT = Path(os.environ["QUANTSTREAM_SNAPSHOT"])
OUT = WORK / "timestral"
OUT.mkdir(exist_ok=True)
split = json.loads((ROOT / "split.json").read_text())
frame = pd.read_parquet(ROOT / "data/processed/features.parquet")
frame = frame[frame.decision_time.lt(pd.Timestamp(split["test_start"]))].copy()
conditions = [
    (name, cov) for name in ["Timestral-5M", "Timestral-28M"] for cov in [False, True]
]
(OUT / "plan.json").write_text(
    json.dumps(
        {
            "conditions": conditions,
            "context": 128,
            "delay": 6,
            "origin_cadence": "24h at UTC00",
            "point": "median",
            "old_test_used": False,
            "pretraining_contamination_not_excluded": True,
            "models": MODELS,
        },
        indent=2,
    )
)
torch.set_num_threads(4)
start = time.time()
reports = []
for name, cov in conditions:
    model = load_forecaster(str(WORK / "models" / name), name)
    results = []
    for k, fold in enumerate(split["tuning_folds"]):
        begin, end = (
            pd.Timestamp(fold["validation_start"]),
            pd.Timestamp(fold["validation_end"]),
        )
        times = pd.DatetimeIndex(
            frame.loc[
                frame.decision_time.ge(begin) & frame.label_end_time.lt(end),
                "decision_time",
            ].unique()
        )
        times = times[times.hour == 0].sort_values()
        for first in range(0, len(times), 16):
            chunk = times[first : first + 16]
            windows = [
                prepare_window(frame, t, ["ret_1", "vol_48"] if cov else [])
                for t in chunk
            ]
            values = forecast_timestral(model, windows)
            for t, qs in zip(chunk, values):
                for i, asset in enumerate(ASSETS):
                    y = frame.loc[
                        frame.symbol.eq(asset) & frame.decision_time.eq(t),
                        "target_return",
                    ].iloc[0]
                    results.append(
                        {
                            "symbol": asset,
                            "decision_time": t,
                            "target_return": y,
                            "prediction": float(qs[i, 4]),
                            "q10": float(qs[i, 0]),
                            "q90": float(qs[i, -1]),
                            "fold": k,
                        }
                    )
            print(
                name,
                cov,
                "fold",
                k,
                "completed",
                min(first + 16, len(times)),
                "/",
                len(times),
                "seconds",
                round(time.time() - start),
                flush=True,
            )
    data = pd.DataFrame(results)
    key = name + ("-covariates" if cov else "-plain")
    data.to_parquet(OUT / (key + ".parquet"), index=False)
    ratios = [
        float(
            ((g.prediction - g.target_return) ** 2).mean() / (g.target_return**2).mean()
        )
        for _, g in data.groupby("fold")
    ]
    score = (
        (data.q90 - data.q10)
        + 10 * np.maximum(data.q10 - data.target_return, 0)
        + 10 * np.maximum(data.target_return - data.q90, 0)
    )
    result = {
        "model": key,
        "rows": len(data),
        "mean_fold_mse_ratio": float(np.mean(ratios)),
        "fold_ratios": ratios,
        "coverage": float(data.target_return.between(data.q10, data.q90).mean()),
        "width": float((data.q90 - data.q10).mean()),
        "interval_score": float(score.mean()),
        "interval": weekly_interval(data),
    }
    reports.append(result)
    (OUT / "results.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(result), flush=True)
(OUT / "status.json").write_text(
    json.dumps(
        {"status": "completed", "seconds": time.time() - start, "conditions": 4},
        indent=2,
    )
)
