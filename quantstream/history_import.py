"""Import checked historical candles without changing models or paper state."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from filelock import FileLock
import pandas as pd

from .common import utcnow, write_json, write_parquet
from .data import validate_bars


def import_history(root, archive):
    from .terminal import config, read_json

    root = Path(root)
    cfg = config(root)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if (
            len(members) != 2
            or {p.filename for p in members}
            != {"bars.parquet", "history_manifest.json"}
            or any(p.file_size > 64 * 1024 * 1024 for p in members)
        ):
            raise ValueError("Expected a history bundle with two bounded files")
        manifest = json.loads(zipped.read("history_manifest.json"))
        payload = zipped.read("bars.parquet")
    digest = hashlib.sha256(payload).hexdigest()
    if (
        manifest.get("format") != "quantstream-history-v1"
        or manifest.get("synthetic") is not False
        or manifest.get("bars_sha256") != digest
        or sorted(manifest.get("symbols", [])) != sorted(cfg.symbols)
        or manifest.get("interval") != cfg.interval
    ):
        raise ValueError("History manifest, symbols, interval or SHA differs")
    incoming = pd.read_parquet(io.BytesIO(payload))
    now = pd.Timestamp.now(tz="UTC")
    incoming, bad, quality = validate_bars(incoming, now=now)
    if (
        not bad.empty
        or quality["duplicate_rows"]
        or len(incoming) != manifest.get("rows")
        or set(incoming.symbol.unique()) != set(cfg.symbols)
    ):
        raise ValueError("Historical candles failed validation")
    path = root / "data/processed/bars.parquet"
    with FileLock(str(root / "terminal.lock"), timeout=10):
        previous = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        candidate = pd.concat([incoming, previous], ignore_index=True)
        merged, invalid, report = validate_bars(candidate, now=now)
        if not invalid.empty:
            raise ValueError("Existing candles failed validation; no files changed")
        added = len(merged) - len(previous)
        write_parquet(path, merged)
        records_path = root / "data/processed/history_imports.json"
        records = read_json(records_path, {})
        records.setdefault(digest, {"imported_at": utcnow(), "manifest": manifest})
        write_json(records_path, records)
        write_json(root / "data/processed/history_quality.json", report)
    return {
        "rows": len(merged),
        "added_rows": added,
        "first_candle": str(merged.timestamp.min()),
        "last_candle": str(merged.timestamp.max()),
        "source_bars_sha256": digest,
        "models_changed": False,
        "paper_state_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(import_history(arguments.root, arguments.archive), indent=2))
