"""Small configuration, hashing and atomic-write helpers."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    interval: str = "1h"
    start_month: str = "2023-01"
    end_month: str = "2026-10"
    horizon: int = 4
    execution_lag_bars: int = 1
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    position_fraction: float = 0.95
    drawdown_halt: float = 0.20
    test_fraction: float = 0.20
    walk_forward_folds: int = 3
    minimum_trades: int = 20
    seed: int = 42
    threads: int = 4
    lookback: int = 48

    def __post_init__(self):
        if self.interval != "1h":
            raise ValueError("MVP supports 1h only; annualisation and gap rules assume 1h.")
        if len(self.symbols) == 0 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be non-empty and unique")
        if self.horizon < 1 or self.execution_lag_bars < 1:
            raise ValueError("horizon >= 1 and execution_lag_bars >= 1 required")
        if not 0 < self.position_fraction <= 1 or not 0 < self.drawdown_halt < 1:
            raise ValueError("invalid risk configuration")
        if not 0.05 <= self.test_fraction <= 0.4 or self.walk_forward_folds < 2:
            raise ValueError("invalid temporal split")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("costs cannot be negative")

    @classmethod
    def load(cls, path="config.json"):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["symbols"] = tuple(d["symbols"])
        return cls(**d)

    def to_dict(self):
        return dataclasses.asdict(self)


def utcnow():
    return pd.Timestamp.now(tz="UTC").isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                              default=json_default, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_parquet(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def stamp():
    return pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")


def git_provenance(root):
    """Record the real revision and working-tree state without requiring Git on Kaggle."""
    try:
        def run(*args):
            return subprocess.run(["git", "-C", str(Path(root).resolve()), *args],
                                  capture_output=True, text=True, timeout=5, check=False)
        commit = run("rev-parse", "HEAD")
        if commit.returncode:
            return {"commit": None, "branch": None, "dirty": None, "status": "unavailable"}
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        state = run("status", "--porcelain", "--untracked-files=normal")
        return {"commit": commit.stdout.strip(), "branch": branch.stdout.strip(),
                "dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
                "status": "available"}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "branch": None, "dirty": None, "status": "unavailable"}
