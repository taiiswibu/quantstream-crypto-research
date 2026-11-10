"""Real Binance historical archives / public REST. No synthetic fallback."""
from __future__ import annotations
import concurrent.futures
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .common import Config, sha256, utcnow, write_json, write_parquet

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines"
REST = "https://data-api.binance.vision/api/v3"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
VALUES = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]


def get_bytes(url, attempts=3):
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantStream-Research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 400, 401, 403, 451) or attempt == attempts - 1:
                raise
            # Respect the server's limit instead of retrying in a tight loop.
            delay = min(float(e.headers.get("Retry-After", 2 ** attempt)), 30)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def timestamp_series(values):
    """Archive Spot timestamps changed from ms to us in 2025. Detect per row."""
    numbers = pd.to_numeric(values, errors="coerce")
    millis = numbers.where(numbers.abs() < 1e14, numbers / 1000)
    return pd.to_datetime(millis, unit="ms", utc=True, errors="coerce")


def normalise(rows, symbol):
    df = pd.DataFrame(rows, columns=COLS)
    df["timestamp"] = timestamp_series(df["open_time"])
    df["source_close_time"] = timestamp_series(df["close_time"])
    for c in VALUES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = symbol
    return df[["symbol", "timestamp", "source_close_time", *VALUES]]


def validate_bars(df, now=None):
    """Quarantine bad rows, reject conflicting duplicate payloads, report gaps."""
    df = df.copy()
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    required = ["timestamp", "source_close_time", "symbol", *VALUES]
    valid = df[required].notna().all(axis=1)
    valid &= np.isfinite(df[VALUES]).all(axis=1)
    valid &= (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= (df[["volume", "quote_volume", "trades", "taker_buy_volume"]] >= 0).all(axis=1)
    valid &= df.high >= df[["open", "close", "low"]].max(axis=1)
    valid &= df.low <= df[["open", "close", "high"]].min(axis=1)
    valid &= df.taker_buy_volume <= df.volume + 1e-8
    valid &= df.timestamp.dt.floor("h").eq(df.timestamp)
    valid &= df.source_close_time.ge(df.timestamp + pd.Timedelta(minutes=59))
    valid &= df.source_close_time.lt(df.timestamp + pd.Timedelta(hours=1))
    valid &= df.source_close_time.lt(current)
    bad = df.loc[~valid].copy()
    clean = df.loc[valid].sort_values(["symbol", "timestamp"])
    duplicates = clean.duplicated(["symbol", "timestamp"], keep=False)
    conflicts = clean.loc[duplicates].groupby(["symbol", "timestamp"])[VALUES].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("Conflicting OHLCV for the same business key. Inspect source revisions.")
    duplicate_count = int(clean.duplicated(["symbol", "timestamp"]).sum())
    clean = clean.drop_duplicates(["symbol", "timestamp"]).reset_index(drop=True)
    gaps = []
    for symbol, group in clean.groupby("symbol"):
        diffs = group.timestamp.diff()
        for index in group.index[diffs.gt(pd.Timedelta(hours=1))]:
            previous = clean.loc[index - 1, "timestamp"]
            missing = int((clean.loc[index, "timestamp"] - previous) / pd.Timedelta(hours=1)) - 1
            gaps.append({"symbol": symbol, "after": previous.isoformat(), "missing_bars": missing})
    return clean, bad, {"input_rows": len(df), "valid_rows": len(clean), "quarantine_rows": len(bad),
                         "duplicate_rows": duplicate_count, "gaps": gaps}


def download_month(root, symbol, month):
    if not re.fullmatch(r"[A-Z0-9]{5,20}", symbol):
        raise ValueError("Invalid symbol")
    filename = f"{symbol}-1h-{month}.zip"
    url = f"{ARCHIVE}/{symbol}/1h/{filename}"
    cache = Path(root) / "data" / "raw" / symbol / filename
    cache.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = cache.with_suffix(".zip.CHECKSUM")
    if not cache.exists() or not checksum_path.exists():
        checksum = get_bytes(url + ".CHECKSUM").decode("utf-8")
        expected = checksum.split()[0]
        content = get_bytes(url)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {filename}")
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(cache)
        checksum_path.write_text(checksum, encoding="utf-8")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256(cache)
    if actual != expected:
        raise ValueError(f"Cached ZIP corrupt: {cache}; remove this ZIP and its CHECKSUM to retry.")
    with zipfile.ZipFile(cache) as z:
        members = [x for x in z.infolist() if x.filename.endswith(".csv")]
        if len(members) != 1 or members[0].file_size > 100_000_000:
            raise ValueError("Unexpected archive layout/size")
        csv = pd.read_csv(io.BytesIO(z.read(members[0])), header=None, names=COLS)
    return normalise(csv.values, symbol), {"symbol": symbol, "month": month, "url": url,
                                           "sha256": actual, "rows": len(csv)}


def download_history(root, cfg: Config):
    root = Path(root)
    months = pd.period_range(cfg.start_month, cfg.end_month, freq="M")
    if len(months) == 0:
        raise ValueError("Empty month range")
    current_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    if str(months[-1]) >= current_month:
        raise ValueError("Monthly archives require completed months; use refresh for current month.")
    work = [(symbol, str(month)) for symbol in cfg.symbols for month in months]
    frames, sources = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(cfg.threads, 4)) as pool:
        futures = [pool.submit(download_month, root, symbol, month) for symbol, month in work]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            frame, source = future.result()  # A missing month stops the build: no silent partial dataset.
            frames.append(frame)
            sources.append(source)
            if i % 10 == 0 or i == len(work):
                print(f"Historical archives: {i}/{len(work)}", flush=True)
    clean, bad, quality = validate_bars(pd.concat(frames, ignore_index=True))
    if set(clean.symbol.unique()) != set(cfg.symbols):
        raise ValueError("Missing requested symbols")
    write_parquet(root / "data/processed/bars.parquet", clean)
    if len(bad):
        write_parquet(root / "data/quarantine/bars.parquet", bad)
    write_json(root / "data/processed/data_quality.json", quality)
    write_json(root / "data/processed/source_manifest.json", {
        "created_at": utcnow(), "provider": "Binance Spot public archives",
        "data_is_synthetic": False, "sources": sorted(sources, key=lambda x: (x["symbol"], x["month"])),
        "bars_sha256": sha256(root / "data/processed/bars.parquet"), "quality": quality})
    return clean


def rest_klines(symbol, start=None, limit=1000):
    params = {"symbol": symbol, "interval": "1h", "limit": limit}
    if start is not None:
        params["startTime"] = int(pd.Timestamp(start).timestamp() * 1000)
    payload = json.loads(get_bytes(f"{REST}/klines?{urllib.parse.urlencode(params)}"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected REST response: {str(payload)[:160]}")
    return normalise(payload, symbol)


def latest_price(symbol):
    payload = json.loads(get_bytes(f"{REST}/ticker/price?symbol={urllib.parse.quote(symbol)}"))
    value = float(payload["price"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Invalid market price")
    return value


def refresh(root, cfg, max_pages=100):
    """REST catch-up with one overlapping candle; live REST is separate from archive provenance."""
    root = Path(root)
    path = root / "data/processed/bars.parquet"
    old = pd.read_parquet(path)
    frames = [old]
    for symbol in cfg.symbols:
        cursor = old.loc[old.symbol.eq(symbol), "timestamp"].max()
        if pd.isna(cursor):
            raise ValueError(f"No history for {symbol}")
        for _ in range(max_pages):
            batch = rest_klines(symbol, cursor)
            if batch.empty:
                break
            frames.append(batch)
            last = batch.timestamp.max()
            if len(batch) < 1000:
                break
            if last <= cursor:
                raise ValueError("REST pagination did not advance")
            cursor = last + pd.Timedelta(hours=1)
        else:
            raise ValueError("Catch-up exceeded page limit; increase max_pages explicitly")
    combined = pd.concat(frames, ignore_index=True)
    # Current unfinished candles are excluded, not treated as corrupt observations.
    combined = combined[combined.timestamp + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC")]
    clean, bad, quality = validate_bars(combined)
    write_parquet(path, clean)
    write_json(root / "data/processed/live_quality.json", {"at": utcnow(), **quality})
    if len(bad):
        write_parquet(root / "data/quarantine/live_bars.parquet", bad)
    return clean
