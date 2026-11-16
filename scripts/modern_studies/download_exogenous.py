"""Fetch public fixed monthly archives, verifying their published SHA256 sidecars."""

from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import io
import urllib.request
import zipfile
import pandas as pd

ROOT = Path(os.environ["QUANTSTREAM_MODERN_OUTPUT"])
OUT = ROOT / "external"
OUT.mkdir(exist_ok=True)
months = pd.period_range("2023-01", "2026-10", freq="M").astype(str).tolist()
jobs = []
for asset in ["BTCUSDT", "ETHUSDT"]:
    for month in months:
        for kind in ["fundingRate", "premiumIndexKlines"]:
            suffix = (
                f"{asset}-fundingRate-{month}.zip"
                if kind == "fundingRate"
                else f"{asset}-1h-{month}.zip"
            )
            path = (
                f"data/futures/um/monthly/{kind}/{asset}/"
                + ("1h/" if kind == "premiumIndexKlines" else "")
                + suffix
            )
            jobs.append((asset, month, kind, path))
(OUT / "plan.json").write_text(
    json.dumps(
        {
            "source": "Binance public archives",
            "start": "2023-01",
            "last_complete_month": "2026-10",
            "files_planned": len(jobs),
            "publication_lag": "one extra hour; funding event time +1h, premium bar open +2h",
            "target_unchanged": True,
        },
        indent=2,
    )
)


def fetch(job):
    asset, month, kind, path = job
    local = OUT / kind / (asset + "-" + month + ".zip")
    local.parent.mkdir(exist_ok=True)
    u = "https://data.binance.vision/" + path
    if not local.exists():
        with urllib.request.urlopen(u, timeout=25) as r:
            raw = r.read()
        with urllib.request.urlopen(u + ".CHECKSUM", timeout=25) as r:
            expected = r.read().decode().split()[0]
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Archive checksum mismatch " + path)
        local.write_bytes(raw)
        local.with_suffix(".sha256").write_text(expected)
    else:
        raw = local.read_bytes()
        expected = local.with_suffix(".sha256").read_text().strip()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Cached archive changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Unexpected archive")
        df = pd.read_csv(z.open(names[0]))
        df["symbol"] = asset
    return kind, df, {"url": u, "sha256": expected, "rows": len(df), "bytes": len(raw)}


records = []
frames = {"fundingRate": [], "premiumIndexKlines": []}
failures = []
with ThreadPoolExecutor(max_workers=6) as pool:
    pending = {pool.submit(fetch, job): job for job in jobs}
    for future in as_completed(pending):
        try:
            kind, data, record = future.result()
            frames[kind].append(data)
            records.append(record)
        except Exception as e:
            failures.append({"job": pending[future], "error": str(e)})
        if (len(records) + len(failures)) % 12 == 0:
            print("archives", len(records), "failed", len(failures), flush=True)
(OUT / "downloads.json").write_text(
    json.dumps({"success": records, "failures": failures}, indent=2)
)
if failures:
    raise RuntimeError(
        "Archive failures retained; no partial data disguised as complete"
    )
for kind, parts in frames.items():
    data = pd.concat(parts, ignore_index=True)
    data.to_parquet(OUT / (kind + ".parquet"), index=False)
    print(kind, len(data), flush=True)
