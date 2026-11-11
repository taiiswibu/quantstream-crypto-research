"""Render the verified development comparison; never train or fetch market data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/experiments/run-004-seed-stability"


def main():
    metrics = pd.read_csv(FOLDER / "metrics-verified.csv")
    summary = json.loads((FOLDER / "verification.json").read_text())
    pooled = metrics[metrics.asset == "ALL"]
    zero = pooled[pooled.model == "zero_return"].iloc[0]
    baseline = pooled[pooled.model == "zero_shot"].iloc[0]
    seeds = pooled[pooled.model == "lora_B"].sort_values("seed")
    mean = summary["pooled_seed_mean"]["mae"]
    blue, gold, ink = "#2563A5", "#B56D14", "#28323D"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": ink, "text.color": ink, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#AAB2BA"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.77, wspace=0.60)
    fig.suptitle("Run 004 — development forecast errors", x=0.04, ha="left", y=0.98, fontsize=18, weight="bold")
    fig.text(0.04, 0.90, "BTC/ETH · same 512 previously observed windows · 200 steps × 3 seeds", fontsize=11)

    values = [zero.mae, baseline.mae, *seeds.mae, mean]
    labels = ["Zero return", "Chronos zero-shot", "LoRA seed 42", "LoRA seed 123", "LoRA seed 2026", "Mean seed MAE"]
    ratios = np.array(values) / zero.mae
    for index, value in enumerate(ratios):
        color = ink if index < 2 else gold if index == 5 else blue
        marker = "D" if index == 5 else "o"
        left.scatter(value, index, color=color, marker=marker, s=55, zorder=3)
        left.annotate(f"{value:.4f}", (value, index), xytext=(7, 0), textcoords="offset points", va="center", fontsize=10)
    left.axvline(1, color=ink, linestyle="--", linewidth=1, zorder=1)
    left.set_yticks(range(6), labels)
    left.set_ylim(5.6, -0.6)
    left.set_xlim(0.995, 1.052)
    left.set_xticks([1.00, 1.01, 1.02, 1.03, 1.04, 1.05])
    left.set_title("Error relative to the zero-return baseline", loc="left", fontsize=11, pad=12)
    left.set_xlabel("MAE / zero-return MAE\nLower is better; 1.0 is the baseline", labelpad=10)
    left.grid(axis="x", color="#E5E9ED", linewidth=0.7)
    left.set_axisbelow(True)

    delta = seeds.mae.to_numpy() - baseline.mae
    bootstrap = summary["paired_mae_bootstrap"]
    center = bootstrap["estimate"] * 1e4
    low, high = np.array(bootstrap["percentile_95"]) * 1e4
    right.scatter(delta * 1e4, [0, 1, 2], color=blue, s=55, zorder=3)
    right.errorbar(center, 3, xerr=[[center - low], [high - center]], color=gold,
                   marker="D", markersize=7, linewidth=2, capsize=5, zorder=3)
    right.axvline(0, color=ink, linestyle="--", linewidth=1)
    right.set_yticks([0, 1, 2, 3], ["Seed 42", "Seed 123", "Seed 2026", "Mean error\n95% interval"])
    right.set_ylim(3.7, -0.7)
    right.set_xlim(-0.37, 0.04)
    right.set_xticks([-0.3, -0.2, -0.1, 0])
    right.set_title("MAE change versus zero-shot", loc="left", fontsize=11, pad=12)
    right.set_xlabel("MAE difference (basis points of return)\nNegative = lower error", labelpad=10)
    right.grid(axis="x", color="#E5E9ED", linewidth=0.7)
    right.set_axisbelow(True)
    fig.text(0.04, 0.09, "Mean error averages the three seed errors, not ensemble predictions. Interval: 2,000 paired resamples of 22 UTC weeks.", fontsize=9)
    fig.text(0.04, 0.045, "Descriptive development comparison, conditional on these seeds. No independent test, PnL result or model promotion.", fontsize=9, color="#505B66")
    fig.savefig(FOLDER / "forecast-errors.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
