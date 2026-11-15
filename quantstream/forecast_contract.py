"""Causal target units and explicit final-model acceptance."""

import numpy as np


def target_scale(frame, params):
    transform = params.get("target_transform", "none")
    if transform == "none":
        return np.ones(len(frame))
    if transform != "volatility":
        raise ValueError("Unknown forecast target transform")
    sigma = frame["vol_48"].to_numpy(dtype=float) * np.sqrt(
        params.get("forecast_horizon", 4)
    )
    if not np.isfinite(sigma).all():
        raise ValueError("Nonfinite causal target scale")
    return 100 * np.clip(sigma, 0.001, 0.05)


def restore_return(prediction, frame, params):
    return np.asarray(prediction) * target_scale(frame, params)


def allows_paper_entries(metadata):
    if "signal_approved" in metadata:
        return metadata["signal_approved"] is True
    gate = metadata.get("validation_gate")
    if isinstance(gate, dict):
        return gate.get("beats_zero_on_mean_and_every_fold") is True
    return True  # Preserve legacy research sessions that predate the final contract.
