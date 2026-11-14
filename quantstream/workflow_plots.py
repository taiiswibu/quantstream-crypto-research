"""Notebook figures drawn only from saved dataset and training artifacts."""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

COLORS = ["#0f766e", "#2563eb", "#b45309", "#64748b"]


def show_saved(root, name):
    if root is not None:
        folder = Path(root) / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(folder / (name + ".png"), dpi=150, bbox_inches="tight")
    plt.show()


def dataset_figures(root):
    from .features import FEATURES

    root = Path(root)
    bars = pd.read_parquet(root / "data/processed/bars.parquet")
    features = pd.read_parquet(root / "data/processed/features.parquet")
    split = json.loads((root / "split.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for (symbol, g), color in zip(bars.groupby("symbol"), COLORS):
        axes[0].plot(
            g.timestamp,
            g.close / g.close.iloc[0] * 100,
            label=symbol,
            color=color,
            lw=0.8,
        )
        axes[1].plot(
            g.timestamp,
            g.close.pct_change() * 100,
            label=symbol,
            color=color,
            lw=0.5,
            alpha=0.6,
        )
    axes[0].set(title="Real hourly candles", ylabel="Price index (start = 100)")
    axes[1].set(xlabel="UTC", ylabel="Hourly return (%)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    plt.tight_layout()
    show_saved(root, "02-prices-returns")
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(features[FEATURES].corr(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(
        xticks=range(len(FEATURES)),
        yticks=range(len(FEATURES)),
        xticklabels=FEATURES,
        yticklabels=FEATURES,
        title="Causal feature correlation · full history for description only",
    )
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    show_saved(root, "03-feature-correlation")
    rows = []
    first = features.decision_time.min()
    for i, b in enumerate(split["tuning_folds"]):
        rows.extend(
            [
                (f"Fold {i + 1} train", first, pd.Timestamp(b["validation_start"])),
                (
                    f"Fold {i + 1} validation",
                    pd.Timestamp(b["validation_start"]),
                    pd.Timestamp(b["validation_end"]),
                ),
            ]
        )
    rows.append(
        (
            "Retrospective test",
            pd.Timestamp(split["test_start"]),
            pd.Timestamp(split["end"]),
        )
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (name, start, end) in enumerate(rows):
        ax.barh(name, end - start, left=start, color=COLORS[i % 4], height=0.55)
    ax.xaxis_date()
    ax.set(
        title="Temporal blocks · purge overlapping labels at each boundary",
        xlabel="UTC",
    )
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    show_saved(root, "04-temporal-splits")


def scaler_figure(train, root=None):
    from .features import FEATURES

    mean = train[FEATURES].mean()
    std = train[FEATURES].std().clip(lower=1e-6)
    z = (train[FEATURES] - mean) / std
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(FEATURES)), train[FEATURES].std(), color=COLORS[0])
    axes[1].bar(range(len(FEATURES)), z.std(), color=COLORS[1])
    for ax, title in zip(
        axes,
        [
            "Feature standard deviation · train only",
            "After train-fitted standardization",
        ],
    ):
        ax.set(
            xticks=range(len(FEATURES)),
            xticklabels=FEATURES,
            title=title,
            ylabel="Standard deviation",
        )
        plt.setp(ax.get_xticklabels(), rotation=80, ha="right")
        ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    show_saved(root, "05-scaler")
    return pd.DataFrame(
        {"feature": FEATURES, "train_mean": mean.values, "train_std": std.values}
    )


def training_figures(root):
    root = Path(root)
    path = root / "trials.csv"
    if not path.exists():
        print("No completed trial evidence yet.")
        return
    trials = pd.read_csv(path)
    valid = trials[trials.state.eq("COMPLETE")].dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, g), color in zip(valid.groupby("params_family"), COLORS):
        ax.scatter(g.number, g.value, label=name, color=color, s=45)
    ax.axhline(1, color="#b91c1c", ls="--", label="Zero-return baseline")
    ax.set(
        title="All completed trial objectives",
        xlabel="Trial",
        ylabel="Mean fold MSE / zero-return MSE",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    show_saved(root, "07-trials")
    selection_path = root / "selection.json"
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())
        frame = pd.DataFrame(selected["shortlist"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            frame.trial.astype(str),
            frame.mean_objective,
            yerr=frame.std_objective,
            color=COLORS[: len(frame)],
            capsize=6,
        )
        ax.axhline(1, color="#b91c1c", ls="--")
        ax.set(
            title="Shortlist · mean ± SD over the three fixed seeds",
            xlabel="Trial",
            ylabel="Normalized MSE",
        )
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        show_saved(root, "08-seed-stability")
    history = []
    neural = (
        valid[valid.params_family.isin(["tcn", "gru"])]
        .sort_values("value")
        .groupby("params_family")
        .head(1)
    )
    for trial in neural.itertuples():
        history.extend(
            (root / "trials" / f"{trial.number:03}").glob(
                "fold-*/early-stop/history.csv"
            )
        )
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for file in history:
            frame = pd.read_csv(file)
            name = "/".join(file.relative_to(root / "trials").parts[:2])
            axes[0].plot(frame.epoch, frame.train_huber_scaled, marker="o", label=name)
            if "inner_validation_mse" in frame:
                axes[1].plot(
                    frame.epoch, frame.inner_validation_mse, marker="o", label=name
                )
        for ax, title in zip(
            axes,
            ["Training Huber · target × 100", "Inner validation MSE · return units"],
        ):
            ax.set(title=title, xlabel="Epoch", ylabel="Loss")
            ax.legend()
            ax.grid(alpha=0.2)
        plt.tight_layout()
        show_saved(root, "09-learning-curves")


def evaluation_figures(root):
    root = Path(root)
    rows = pd.read_parquet(root / "retrospective_predictions.parquet")
    metrics = pd.read_json(root / "retrospective_metrics.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(metrics.model, metrics.mae * 100, color=COLORS)
    axes[0].set(title="Retrospective MAE", ylabel="Return percentage points")
    for (symbol, g), color in zip(rows.groupby("symbol"), COLORS):
        axes[1].scatter(
            g.target_return * 100,
            g.prediction * 100,
            s=4,
            alpha=0.2,
            color=color,
            label=symbol,
        )
        axes[2].hist(
            (g.prediction - g.target_return) * 100,
            bins=60,
            alpha=0.5,
            color=color,
            label=symbol,
        )
    axes[1].set(
        title="Prediction and observed return",
        xlabel="Observed (%)",
        ylabel="Predicted (%)",
    )
    axes[2].set(
        title="Error distribution",
        xlabel="Prediction − observed (pp)",
        ylabel="Windows",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend()
    axes[2].legend()
    plt.tight_layout()
    show_saved(root, "10-forecast-errors")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (symbol, g) in zip(axes, rows.groupby("symbol")):
        g = g.tail(300)
        ax.plot(
            g.decision_time,
            g.target_return * 100,
            label="Observed",
            color=COLORS[1],
            alpha=0.7,
        )
        ax.plot(g.decision_time, g.prediction * 100, label="Predicted", color=COLORS[0])
        ax.set(title=symbol, ylabel="4h return (%)")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Decision time · UTC")
    plt.tight_layout()
    show_saved(root, "11-forecast-timeline")
    curve = pd.read_parquet(root / "retrospective_equity.parquet")
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name, color in zip(["model", "buy_hold", "ema"], COLORS):
        axes[0].plot(curve.timestamp, curve[name], label=name, color=color)
    axes[0].set(
        title="Retrospective equity after assumed costs", ylabel="Initial equity = 1"
    )
    axes[0].legend()
    drawdown = curve.model / curve.model.cummax().clip(lower=1) - 1
    axes[1].fill_between(curve.timestamp, drawdown * 100, 0, color="#b91c1c", alpha=0.3)
    axes[1].set(ylabel="Drawdown (%)", xlabel="UTC")
    for ax in axes:
        ax.grid(alpha=0.2)
    plt.tight_layout()
    show_saved(root, "12-equity")
    trades = pd.read_parquet(root / "retrospective_trades.parquet")
    print(
        "Selected strategy closed trades:",
        len(trades),
        "; low exposure can produce a nearly flat equity curve.",
    )
    print(
        "These are historically observed periods; this is not independent investment evidence."
    )
    return metrics


def architecture_figure(root=None):
    from matplotlib.patches import FancyBboxPatch

    paths = {
        "TCN": [
            "48 hourly steps × 18 features",
            "Causal residual conv · dilation 1",
            "Causal residual conv · dilation 2",
            "Causal residual conv · dilation 4",
            "Causal residual conv · dilation 8",
            "Last step → Dropout → Linear",
        ],
        "GRU": [
            "48 hourly steps × 18 features",
            "GRU · sigmoid gates / tanh state",
            "Hidden sequence · 48 × width",
            "Select last hidden state",
            "Dropout",
            "Linear → scaled return",
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, (family, labels) in zip(axes, paths.items()):
        ax.set(xlim=(0, 1), ylim=(0, 1), title=family)
        ax.axis("off")
        for i, text in enumerate(labels):
            y = 0.89 - i * 0.145
            ax.add_patch(
                FancyBboxPatch(
                    (0.08, y - 0.045),
                    0.84,
                    0.09,
                    boxstyle="round,pad=.015",
                    facecolor="#ecfdf5",
                    edgecolor="#0f766e",
                )
            )
            ax.text(0.5, y, text, ha="center", va="center", fontsize=10)
            if i < len(labels) - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.09),
                    xytext=(0.5, y - 0.06),
                    arrowprops={"arrowstyle": "->", "color": "#64748b"},
                )
    fig.suptitle(
        "Architecture before training · TCN activations searched: GELU / ReLU / SiLU"
    )
    plt.tight_layout()
    show_saved(root, "06-architecture")
