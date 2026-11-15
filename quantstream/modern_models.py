"""Causal delayed-label inputs for TiRex-2. Forecast the existing return target."""

from pathlib import Path
import json
import hashlib
from functools import lru_cache
import urllib.request
import time

import numpy as np
import pandas as pd

TIREX_REVISION = "352636173653592b6dceb3fbeee9224ca97952a9"
TIREX_WEIGHTS_SHA = "17760e4eee4c14e0d8bad8b9d9ce6f5076aaeeb9d956a88b8e95cf1493a1c176"
CONTEXT = 128
LABEL_DELAY = 6  # Five hours to mature the label plus one hour ingestion margin.
ASSETS = ("BTCUSDT", "ETHUSDT")
MODELS = {
    "TiRex-2": {
        "repo": "NX-AI/TiRex-2",
        "revision": TIREX_REVISION,
        "weights": TIREX_WEIGHTS_SHA,
        "config": "e591152ebc5bd7245b5258cd26a52b57c1f6cfe60d6743111856ab0b12f7425e",
        "license": "Apache-2.0",
    },
    "Timestral-5M": {
        "repo": "mistralai/Timestral-5M",
        "revision": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
        "weights": "7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738",
        "config": "b5f27f61513f8fd1a3aa996f3720d3679cb40bce5bcb08eb3dc320ce726e7c2f",
        "license": "Modified MIT; revenue condition in model LICENSE",
    },
    "Timestral-28M": {
        "repo": "mistralai/Timestral-28M",
        "revision": "49b26409ac87b5935539a5d8ffe7f397b720db47",
        "weights": "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d",
        "config": "0cdf44b8ea981191de53078e0a5ecbbd437046bd939a546d5dbe17868f71281f",
        "license": "Modified MIT; revenue condition in model LICENSE",
    },
}


def check_model_files(folder, name):
    folder = Path(folder)
    spec = MODELS[name]
    for filename, key in [("config.json", "config"), ("model.safetensors", "weights")]:
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != spec[key]:
            raise ValueError(
                name + " " + filename + " differs from the reviewed revision"
            )


def prepare_model(folder, name="Timestral-5M", deadline=None):
    """Public pinned files only; no account credentials or remote-code loading."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = MODELS[name]
    filenames = ["config.json", "model.safetensors"]
    if name.startswith("Timestral"):
        filenames.append("LICENSE")
    for filename in filenames:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Model setup exceeded the study budget")
        destination = folder / filename
        if destination.exists():
            continue
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
        temporary = destination.with_suffix(destination.suffix + ".part")
        size = 0
        with (
            urllib.request.urlopen(url, timeout=45) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Model setup exceeded the study budget")
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise ValueError("Unexpected forecast-model size")
                output.write(chunk)
        temporary.replace(destination)
    check_model_files(folder, name)
    (folder / "provenance.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    return folder


def prepare_window(frame, origin, covariates=(), context=CONTEXT):
    """Target stops at t-6; bridge covariates stop at t. No price after t is read."""
    origin = pd.Timestamp(origin)
    times = pd.date_range(
        end=origin,
        periods=context + LABEL_DELAY,
        freq="h",
    )
    target, future_covariates = [], []
    for asset in ASSETS:
        part = frame[frame.symbol.eq(asset)].set_index("decision_time").reindex(times)
        historical = part.iloc[:context]
        if (
            historical.target_return.isna().any()
            or historical.label_end_time.isna().any()
            or not historical.label_end_time.lt(origin).all()
        ):
            raise ValueError("Incomplete or unmatured target context")
        if (
            not (historical.label_end_time - historical.index)
            .eq(pd.Timedelta(hours=5))
            .all()
        ):
            raise ValueError("Expected four-hour returns with one-hour execution lag")
        y = historical.target_return.to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Invalid return context")
        target.append(y)
        for name in covariates:
            values = part[name].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Missing causal covariate: " + name)
            future_covariates.append(values)
    return {
        "target": np.asarray(target, dtype=np.float32),
        "future_covariates": np.asarray(future_covariates, dtype=np.float32)
        if future_covariates
        else None,
        "last_target_origin": times[context - 1].isoformat(),
        "latest_covariate_origin": origin.isoformat(),
        "context": context,
        "prediction_length": LABEL_DELAY,
    }


def load_tirex(folder, device="cpu"):
    from tirex2 import load_model

    folder = Path(folder)
    check_model_files(folder, "TiRex-2")
    return load_model(folder, device=device, use_flex_attention=False)


def forecast_windows(model, windows, batch_size=8):
    import torch
    from tirex2 import TimeseriesType

    series = [
        TimeseriesType(
            target=torch.from_numpy(w["target"]),
            past_covariates=None,
            future_covariates=torch.from_numpy(w["future_covariates"])
            if w["future_covariates"] is not None
            else None,
        )
        for w in windows
    ]
    with torch.inference_mode():
        forecasts = model.forecast(
            series,
            prediction_length=LABEL_DELAY,
            output_type="numpy",
            batch_size=batch_size,
            pad_context=False,
            tta_diff=False,
            tta_sign_flip=False,
        )
    result = np.asarray([f[:, :, -1] for f in forecasts])
    if result.shape != (len(windows), 2, 9) or not np.isfinite(result).all():
        raise ValueError("Invalid TiRex-2 forecasts")
    if (np.diff(result, axis=-1) < -1e-7).any():
        raise ValueError("Crossed TiRex-2 quantiles")
    return result


def model_provenance(folder):
    path = Path(folder) / "provenance.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=3)
def load_forecaster(folder, name):
    import torch

    torch.set_num_threads(4)
    check_model_files(folder, name)
    if name == "TiRex-2":
        return load_tirex(folder)
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(
        folder, device="cpu", local_files_only=True
    )


def forecast_timestral(model, windows):
    context = np.stack([w["target"] for w in windows])
    known = None
    if windows[0]["future_covariates"] is not None:
        known = np.stack([w["future_covariates"] for w in windows])
    forecast = model.forecast(
        context,
        prediction_length=LABEL_DELAY,
        known_covariates=known,
        symmetric_averaging=known is None,
        clip_nonnegative=False,
    )
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    values = np.stack(
        [forecast.quantile(q)[:, :, -1].cpu().numpy() for q in levels], axis=-1
    )
    if values.shape != (len(windows), 2, 9) or not np.isfinite(values).all():
        raise ValueError("Invalid Timestral forecasts")
    if (np.diff(values, axis=-1) < -1e-7).any():
        raise ValueError("Crossed Timestral quantiles")
    return values


def predict_latest_distribution(frame, folder, name):
    origin = min(frame.groupby("symbol").decision_time.max())
    window = prepare_window(frame, origin, covariates=())
    model = load_forecaster(str(Path(folder).resolve()), name)
    values = (
        forecast_windows(model, [window])[0]
        if name == "TiRex-2"
        else forecast_timestral(model, [window])[0]
    )
    return [
        {
            "symbol": asset,
            "decision_time": origin.isoformat(),
            "entry_time": (origin + pd.Timedelta(hours=1)).isoformat(),
            "horizon_end": (origin + pd.Timedelta(hours=5)).isoformat(),
            "q10": float(values[i, 0]),
            "q50": float(values[i, 4]),
            "q90": float(values[i, -1]),
            "source": name,
            "weights_sha256": MODELS[name]["weights"],
            "forecaster_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "input_sha256": hashlib.sha256(
                window["target"].tobytes() + origin.isoformat().encode()
            ).hexdigest(),
            "latest_label_used": (origin - pd.Timedelta(hours=1)).isoformat(),
            "parameters": {
                "context": CONTEXT,
                "delay": LABEL_DELAY,
                "covariates": [],
                "revision": MODELS[name]["revision"],
            },
            "nominal_coverage": 0.8,
            "coverage_is_guaranteed": False,
        }
        for i, asset in enumerate(ASSETS)
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="Timestral-5M")
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_model(args.folder, args.model))
