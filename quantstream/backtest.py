"""Long/cash event simulation. Fixed horizon, one position per symbol, no leverage."""
from __future__ import annotations
import numpy as np
import pandas as pd


def metrics(equity, trades):
    values = np.r_[1.0, equity.to_numpy(dtype=float)]
    returns = values[1:] / values[:-1] - 1
    sd = np.std(returns, ddof=1) if len(returns) > 1 else 0
    sharpe = float(np.mean(returns) / sd * np.sqrt(365 * 24)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(values)
    trade_returns = [x["net_return"] for x in trades]
    return {"total_return": float(values[-1] - 1), "sharpe_hourly_estimate": sharpe,
            "max_drawdown": float(np.min(values / peak - 1)), "trades": len(trades),
            "win_rate": float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0,
            "turnover": float(sum(x["entry_notional"] + x["exit_notional"] for x in trades)),
            "note": "Hourly Sharpe assumes sqrt(8760) scaling; serial correlation is not adjusted."}


def simulate(bars, signals, start, end, cfg, buy_hold=False):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Lag >= 1 bar permits feature calculation before the assumed entry open.
    signals = signals.copy()
    signals["fill_time"] = signals.decision_time + pd.Timedelta(hours=cfg.execution_lag_bars)
    curves, all_trades = [], []
    fee, slip = cfg.fee_bps / 10000, cfg.slippage_bps / 10000
    timeline = pd.date_range(start, end, freq="h", inclusive="left")
    for symbol in cfg.symbols:
        asset = bars[(bars.symbol == symbol) & bars.timestamp.ge(start) & bars.timestamp.lt(end)].sort_values("timestamp")
        lookup = signals[(signals.symbol == symbol) & signals.go_long].set_index("fill_time")
        if lookup.index.has_duplicates:
            raise ValueError("Duplicate signals for an execution time")
        cash, qty, peak, halted = 1.0, 0.0, 1.0, False
        scheduled_exit, entry = None, None
        points = {}
        rows = list(asset.itertuples(index=False))
        for i, row in enumerate(rows):
            t = row.timestamp
            should_enter = (buy_hold and i == 0) or (t in lookup.index)
            if not qty and should_enter and not halted:
                exit_i = len(rows) - 1 if buy_hold else i + cfg.horizon - 1
                contiguous = (exit_i < len(rows) and rows[exit_i].timestamp == t + pd.Timedelta(hours=exit_i - i))
                if contiguous:
                    entry_price = row.open * (1 + slip)
                    budget = cash * cfg.position_fraction
                    qty = budget / (entry_price * (1 + fee))
                    notional = qty * entry_price
                    cash -= notional * (1 + fee)
                    scheduled_exit = rows[exit_i].timestamp
                    entry = {"symbol": symbol, "entry_time": t.isoformat(), "entry_price": float(entry_price),
                             "entry_notional": float(notional), "entry_fee": float(notional * fee),
                             "capital_before": float(cash + budget)}
            mark = cash + qty * row.close
            peak = max(peak, mark)
            stop = not buy_hold and mark / peak - 1 <= -cfg.drawdown_halt
            if qty and (t == scheduled_exit or i == len(rows) - 1 or stop):
                exit_price = row.close * (1 - slip)
                exit_notional = qty * exit_price
                cash += exit_notional * (1 - fee)
                cost_basis = entry["entry_notional"] + entry["entry_fee"]
                net_return = (exit_notional * (1 - fee)) / cost_basis - 1
                all_trades.append({**entry, "exit_time": (t + pd.Timedelta(hours=1)).isoformat(),
                                   "exit_price": float(exit_price), "exit_notional": float(exit_notional),
                                   "exit_fee": float(exit_notional * fee), "net_return": float(net_return),
                                   "exit_reason": "drawdown_halt" if stop else "horizon"})
                qty = 0.0
                mark = cash
            if stop:
                halted = True
            points[t] = mark
        # Cash remains idle before a symbol's first observation; missing bars are carried forward.
        curve = pd.Series(points, dtype=float).reindex(timeline).ffill().fillna(1.0)
        curves.append(curve.rename(symbol))
    equity = pd.concat(curves, axis=1).mean(axis=1)
    equity.index = equity.index + pd.Timedelta(hours=1)
    return equity, pd.DataFrame(all_trades), metrics(equity, all_trades)


def signals_from_prediction(samples, prediction, threshold):
    out = samples[["symbol", "decision_time"]].copy()
    out["prediction"] = np.asarray(prediction, dtype=float)
    out["go_long"] = out.prediction > threshold
    return out


def evaluate_strategy(bars, samples, predictions, threshold, start, end, cfg):
    return simulate(bars, signals_from_prediction(samples, predictions, threshold), start, end, cfg)
