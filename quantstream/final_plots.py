"""Final research figures from saved evidence, including failed candidates."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .workflow_plots import show_saved


def figures(root):
    root = Path(root)
    trials = pd.read_csv(root / "trials.csv")
    labels = trials.params_family + " / " + trials.params_target_transform
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(trials.value, np.arange(len(trials)), s=65, color="#0f766e")
    ax.axvline(1, color="#b91c1c", ls="--", label="Zero-return baseline")
    ax.set(
        yticks=np.arange(len(trials)),
        yticklabels=labels,
        xlabel="Mean fold MSE / zero-return MSE",
        title="Eight balanced conditions · same four-hour return target",
    )
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    show_saved(root, "15-balanced-conditions")
    epochs = []
    for trial in trials.itertuples():
        if trial.params_family not in ("tcn", "gru"):
            continue
        folder = root / "trials" / f"{trial.number:03}"
        report = json.loads((folder / "report.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for fold in report["folds"]:
            h = pd.read_csv(folder / f"fold-{fold['fold']}" / "early-stop/history.csv")
            name = f"Fold {fold['fold'] + 1}"
            axes[0].plot(h.epoch, h.train_loss_scaled, label=name, marker=".")
            axes[1].plot(h.epoch, h.inner_validation_mse, label=name, marker=".")
            epochs.append(
                {
                    "fit": f"{trial.params_family}/{trial.params_target_transform}/{name}",
                    "trained": len(h),
                    "selected": fold["epochs"],
                }
            )
        units = (
            "100 × raw return"
            if trial.params_target_transform == "none"
            else "return / past volatility"
        )
        axes[0].set(title="Training MSE · " + units, ylabel="Training objective")
        axes[1].set(title="Early-stop MSE · original return units", ylabel="MSE")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(f"{trial.params_family.upper()} · {trial.params_target_transform}")
        plt.tight_layout()
        show_saved(root, f"16-learning-{trial.number:03}")
    e = pd.DataFrame(epochs)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        e.fit, e.trained, label="Epochs actually trained", color="#0f766e", alpha=0.7
    )
    ax.scatter(
        e.selected, e.fit, label="Checkpoint selected", color="#b45309", zorder=3
    )
    ax.set(
        title="Training duration and selected checkpoint are different", xlabel="Epoch"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    show_saved(root, "17-actual-epochs")
    selection = json.loads((root / "selection.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, item in enumerate(selection["shortlist"]):
        interval = item["validation_gate"]["weekly_interval_seed42"]
        if interval["available"]:
            ax.plot(
                [interval["lower"], interval["upper"]], [i, i], lw=3, color="#2563eb"
            )
            ax.scatter(
                [(interval["lower"] + interval["upper"]) / 2], [i], color="#2563eb"
            )
    ax.axvline(0, color="#b91c1c", ls="--")
    ax.set(
        yticks=range(len(selection["shortlist"])),
        yticklabels=[x["params"]["family"] for x in selection["shortlist"]],
        xlabel="Relative MSE difference to zero · negative is better",
        title="Approximate 95% weekly-cluster intervals · reused validation, not independent proof",
    )
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    show_saved(root, "18-weekly-uncertainty")


def diagnostic_zip(root, output):
    from zipfile import ZipFile, ZIP_DEFLATED

    root, output = Path(root), Path(output)
    with ZipFile(output, "w", ZIP_DEFLATED) as zipped:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            keep = (
                (
                    len(rel.parts) == 1
                    and path.suffix in {".json", ".csv", ".log", ".txt", ".parquet"}
                )
                or (
                    rel.parts[0] in {"trials", "stability", "final-fit", "audit_model"}
                    and path.suffix in {".json", ".csv"}
                )
                or rel.parts[0] in {"deployment", "zero_reference"}
                or (rel.parts[0] == "audit_model" and "deployment" in rel.parts)
            )
            if keep:
                zipped.write(path, rel.as_posix())
    return output
