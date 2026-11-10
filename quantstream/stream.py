"""Optional WebSocket capture: retain only closed, validated candles."""
import asyncio
import json
from pathlib import Path
from .common import utcnow


async def capture(root, cfg):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidHandshake
    names = "/".join(f"{symbol.lower()}@kline_1h" for symbol in cfg.symbols)
    url = f"wss://data-stream.binance.vision:443/stream?streams={names}"
    folder = Path(root) / "data/stream"
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            async with connect(url, ping_interval=20, ping_timeout=30) as ws:
                async for payload in ws:
                    event = json.loads(payload)["data"]
                    candle = event.get("k", {})
                    if candle.get("x") is not True:
                        continue
                    from .data import normalise, validate_bars
                    rows = [[candle["t"], candle["o"], candle["h"], candle["l"], candle["c"], candle["v"],
                             candle["T"], candle["q"], candle["n"], candle["V"], candle["Q"], 0]]
                    clean, bad, _ = validate_bars(normalise(rows, candle["s"]))
                    record = {"received_at": utcnow(), "valid": bool(len(clean) == 1), "event": event}
                    destination = folder / f"{utcnow()[:10]}.jsonl"
                    with destination.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"Closed candle: {candle['s']} valid={record['valid']}", flush=True)
        except (OSError, TimeoutError, ConnectionClosed, InvalidHandshake) as e:
            print(f"Stream reconnect: {type(e).__name__}", flush=True)
            await asyncio.sleep(5)
