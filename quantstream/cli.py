"""Explicit local commands; no credentials or real-money execution required."""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path
from .common import Config, stamp


def main():
    parser = argparse.ArgumentParser(description="QuantStream local research and paper trading")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "prepare", "research", "evaluate", "export-kaggle", "demo", "stream", "paper-reset", "doctor"]:
        sub.add_parser(name)
    p = sub.add_parser("paper")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    p = sub.add_parser("tcn-train")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", default="artifacts/tcn_training")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("register-tcn")
    p.add_argument("folder")
    p = sub.add_parser("chronos-finetune")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", default="artifacts/chronos")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-windows", type=int, default=512)
    p.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Training snapshots contain their own config, so Kaggle commands do not need a root config.
    if args.command == "tcn-train":
        from .deep import train_snapshot
        print(json.dumps(train_snapshot(args.snapshot, args.output, args.epochs, args.device), indent=2))
        return
    if args.command == "chronos-finetune":
        from .finetune import chronos_experiment
        print(json.dumps(chronos_experiment(args.snapshot, args.output, args.steps, args.max_windows, args.device), indent=2))
        return
    cfg = Config.load(root / args.config)
    from . import data, features, research
    if args.command in ("download", "demo"):
        data.download_history(root, cfg)
    if args.command in ("prepare", "demo"):
        samples, plan = features.prepare(root, cfg)
        print(f"Features ready: {len(samples):,} rows. Final holdout starts {plan['test_start']}")
    if args.command in ("research", "demo"):
        print("Research artifacts:", research.research(root, cfg))
    if args.command == "evaluate":
        print("Locked final report:", research.final_evaluate(root, cfg))
    if args.command == "demo":
        print("Development-only Kaggle input:", research.export_kaggle(root, cfg))
        print("Research complete. Choose your champion BEFORE running: python -m quantstream evaluate")
    if args.command == "export-kaggle":
        print(research.export_kaggle(root, cfg))
    if args.command == "register-tcn":
        from .deep import register_tcn
        print(json.dumps(register_tcn(root, args.folder), indent=2))
    if args.command == "paper":
        from .paper import paper_step
        if args.poll_seconds < 30:
            parser.error("poll-seconds must be >= 30")
        while True:
            state = paper_step(root, cfg)
            print(f"{state['updated_at']} paper equity={state['equity']:.2f} USDT", flush=True)
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
    if args.command == "paper-reset":
        path = root / "paper/state.json"
        if path.exists():
            archive = path.with_name(f"state_archived_{stamp()}.json")
            path.rename(archive)
            print("Archived:", archive)
    if args.command == "stream":
        from .stream import capture
        asyncio.run(capture(root, cfg))
    if args.command == "doctor":
        import importlib.metadata
        for name in ["numpy", "pandas", "pyarrow", "duckdb", "scikit-learn", "xgboost-cpu", "xgboost", "streamlit", "torch", "chronos-forecasting"]:
            try:
                print(f"{name}: {importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                print(f"{name}: not installed (some packages are optional)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Saved artifacts remain on disk.")
