"""Portable inference for exported notebook models; no fitting in this module."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .features import FEATURES


def predict_file(path, frame, metadata):
    from .forecast_contract import restore_return

    path = Path(path)
    family = metadata["hyperparameters"]["family"]
    if family == "zero_return":
        if json.loads(path.read_text()).get("constant") != 0:
            raise ValueError("Invalid zero-return baseline")
        return np.zeros(len(frame))
    if family == "ridge":
        data = json.loads(path.read_text())
        x = (frame[FEATURES].to_numpy() - np.asarray(data["mean"])) / np.asarray(
            data["scale"]
        )
        return restore_return(
            x @ np.asarray(data["coef"]) + data["intercept"],
            frame,
            metadata["hyperparameters"],
        )
    if family == "xgboost":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(path)
        return restore_return(
            model.predict(xgb.DMatrix(frame[FEATURES])),
            frame,
            metadata["hyperparameters"],
        )
    import torch
    from .search import SequenceRegressor
    from .deep import predict_tcn

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = SequenceRegressor(metadata["hyperparameters"])
    model.load_state_dict(payload["state_dict"], strict=True)
    values = predict_tcn(
        model, frame, payload["mean"], payload["std"], payload["lookback"]
    )
    return restore_return(values, frame, metadata["hyperparameters"])


def check_parity(path, metadata, parity):
    """Saved raw feature windows test scaler, architecture, output units and weights."""
    windows = np.asarray(parity["windows"], dtype=np.float32)
    if (
        windows.ndim != 3
        or windows.shape[2] != len(FEATURES)
        or not np.isfinite(windows).all()
    ):
        raise ValueError("Invalid model parity inputs")
    rows = []
    for i, window in enumerate(windows):
        part = pd.DataFrame(window, columns=FEATURES)
        part["symbol"], part["segment"] = str(i), 1
        part["decision_time"] = pd.date_range(
            "2020-01-01", periods=len(window), freq="h", tz="UTC"
        )
        part["target_return"] = np.nan
        rows.append(part)
    data = pd.concat(rows, ignore_index=True)
    pred = predict_file(path, data, metadata)
    last = data.groupby("symbol", sort=False).tail(1).index
    actual = np.asarray(pred)[last]
    expected = np.asarray(parity["predictions"])
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid model parity forecasts")
    maximum = float(np.max(abs(actual - expected)))
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Prediction parity failed: max delta {maximum:g}")
    return maximum
