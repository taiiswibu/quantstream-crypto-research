"""Figures from saved recent-method results; hourly and daily cohorts stay separate."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def figures(root):
    root = Path(root)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"), dpi=150, bbox_inches="tight")
        plt.show()

    frame = pd.read_parquet(root / "enriched/enriched_features.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for i, asset in enumerate(["BTCUSDT", "ETHUSDT"]):
        g = frame[frame.symbol.eq(asset)].dropna(
            subset=["funding_hourly", "premium_close"]
        )
        g = g.iloc[::24]
        axes[i, 0].plot(g.decision_time, g.funding_hourly * 100, color="#0f766e")
        axes[i, 1].plot(g.decision_time, g.premium_close * 100, color="#2563eb")
        axes[i, 0].set(title=asset + " funding / hour", ylabel="Percent")
        axes[i, 1].set(title=asset + " premium index", ylabel="Percent")
        for ax in axes[i]:
            ax.grid(alpha=0.2)
    fig.suptitle("New archive inputs · daily sample · source publication lag applied")
    save("01-funding-premium")

    hourly = json.loads((root / "enriched/results.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(hourly):
        interval = r["weekly_interval"]
        ax.plot(
            [interval["lower"] + 1, interval["upper"] + 1],
            [i, i],
            color="#2563eb",
            lw=2,
        )
        ax.scatter(r["mean_fold_mse_ratio"], i, color="#0f766e", zorder=3)
    ax.axvline(1, color="#b91c1c", ls="--")
    ax.set(
        yticks=range(len(hourly)),
        yticklabels=[r["model"] for r in hourly],
        xlabel="Mean fold MSE / zero-return MSE",
        title="Hourly cohort · approximate weekly intervals on reused development",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    save("02-hourly-methods")

    neural = json.loads((root / "foundation/results.json").read_text()) + json.loads(
        (root / "timestral/results.json").read_text()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(neural):
        ax.plot(
            [r["interval"]["lower"] + 1, r["interval"]["upper"] + 1],
            [i, i],
            color="#64748b",
            lw=2,
        )
        ax.scatter(r["mean_fold_mse_ratio"], i, color="#b45309", zorder=3)
    ax.axvline(1, color="#b91c1c", ls="--")
    ax.set(
        yticks=range(len(neural)),
        yticklabels=[r["model"] for r in neural],
        xlabel="Mean fold MSE / zero-return MSE",
        title="Foundation screening · daily UTC 00:00 origins · median forecast",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    save("03-foundation-screen")

    intervals = json.loads((root / "intervals/summary.json").read_text())
    intervals = [r for r in intervals if r["coverage"] is not None]
    for r in neural:
        if r["model"].endswith("-plain"):
            intervals.append(
                {
                    "method": r["model"],
                    "coverage": r["coverage"],
                    "interval_score": r["interval_score"],
                }
            )
    labels = [r["method"] for r in intervals]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    y = np.arange(len(intervals))
    axes[0].barh(y, [r["coverage"] * 100 for r in intervals], color="#0f766e")
    axes[0].axvline(80, color="#b91c1c", ls="--", label="Nominal 80%")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Observed coverage (%)")
    axes[0].legend()
    axes[1].barh(y, [r["interval_score"] * 100 for r in intervals], color="#2563eb")
    axes[1].set(xlabel="Mean interval score ×100 · lower is better")
    axes[0].invert_yaxis()
    fig.suptitle(
        "Interval quality is separate from point accuracy · matched daily cohort"
    )
    save("04-interval-quality")

    trace = pd.read_parquet(root / "enriched/delayed_rls_bank-original.parquet")
    fig, ax = plt.subplots(figsize=(12, 4))
    for asset, g in trace.groupby("symbol"):
        ax.plot(g.decision_time, g.coefficient_norm, label=asset, lw=1)
    ax.set(
        title="Actual online coefficient trace · each fold restarts from its train state",
        xlabel="Decision time UTC",
        ylabel="Coefficient norm",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    save("05-online-learning")

    daily = pd.read_parquet(root / "intervals/predictions.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, asset in zip(axes, ["BTCUSDT", "ETHUSDT"]):
        g = (
            daily[daily.symbol.eq(asset) & daily.method.eq("rolling_empirical")]
            .sort_values("decision_time")
            .tail(90)
        )
        ax.fill_between(
            g.decision_time,
            g.q10 * 100,
            g.q90 * 100,
            alpha=0.2,
            color="#0f766e",
            label="Causal empirical 10–90% interval",
        )
        ax.plot(
            g.decision_time,
            g.target_return * 100,
            color="#2563eb",
            lw=1,
            label="Observed 4h return",
        )
        ax.set(title=asset, ylabel="Return (%)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Forecast origin UTC · last 90 daily observations")
    save("06-observed-and-intervals")

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot([-134, -6], [2, 2], lw=13, color="#0f766e", label="Mature target history")
    ax.plot(
        [-134, 0],
        [1, 1],
        lw=13,
        color="#2563eb",
        label="Covariates known by decision time",
    )
    ax.plot(
        [1, 5], [0, 0], lw=13, color="#b45309", label="Prices needed for current target"
    )
    ax.axvline(0, color="#b91c1c", ls="--")
    ax.set(
        yticks=[0, 1, 2],
        yticklabels=["Future truth", "Covariates", "Target context"],
        xlabel="Hours relative to decision time t",
        title="Causal alignment · six-step model horizon maps to the current 4h return target",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3)
    save("07-causal-alignment")
