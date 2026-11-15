"""Fixed causal exogenous-data and delayed-RLS study; old test never selects models."""

from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

from quantstream.features import FEATURES, fold_data
from quantstream.search import with_context
from quantstream.final_research import weekly_interval

WORK = Path(os.environ["QUANTSTREAM_MODERN_OUTPUT"])

ROOT = Path(os.environ["QUANTSTREAM_SNAPSHOT"])
OUT = WORK / "enriched"
OUT.mkdir(exist_ok=True)
EXT = WORK / "external"
EXTRA = [
    "funding_hourly",
    "funding_change",
    "premium_close",
    "premium_mean24",
    "premium_change4",
    "premium_std24",
]
split = json.loads((ROOT / "split.json").read_text())
plan = {
    "methods": ["ridge", "xgboost", "delayed_rls_bank"],
    "groups": ["original", "funding_premium"],
    "label": "unchanged exact 4h return",
    "folds": split["tuning_folds"],
    "test_start": split["test_start"],
    "old_test_used": False,
    "publication_lag": "1 additional hour",
    "rls_forgetting": [1.0, 0.9999, 0.999],
    "ridge_alpha": 1000,
    "point": "mean prediction",
    "selection_scope": "reused development; exploratory",
}
(OUT / "plan.json").write_text(json.dumps(plan, indent=2))
features = pd.read_parquet(ROOT / "data/processed/features.parquet")
fund = pd.read_parquet(EXT / "fundingRate.parquet")
prem = pd.read_parquet(EXT / "premiumIndexKlines.parquet")
fund["available_at"] = pd.to_datetime(
    fund.calc_time, unit="ms", utc=True
) + pd.Timedelta(hours=1)
fund["funding_hourly"] = fund.last_funding_rate / fund.funding_interval_hours
fund = fund.sort_values(["symbol", "available_at"])
fund["funding_change"] = fund.groupby("symbol").funding_hourly.diff()
prem["available_at"] = pd.to_datetime(
    prem.open_time, unit="ms", utc=True
) + pd.Timedelta(hours=2)
prem = prem.sort_values(["symbol", "available_at"]).copy()
prem["premium_close"] = prem.close.astype(float)
prem["_segment"] = (
    prem.groupby("symbol").open_time.diff().ne(3600000).groupby(prem.symbol).cumsum()
)
for _, idx in prem.groupby(["symbol", "_segment"]).groups.items():
    p = prem.loc[idx]
    prem.loc[idx, "premium_mean24"] = p.premium_close.rolling(24, min_periods=24).mean()
    prem.loc[idx, "premium_std24"] = p.premium_close.rolling(24, min_periods=24).std()
    prem.loc[idx, "premium_change4"] = p.premium_close.diff(4)
parts = []
for asset, g in features.groupby("symbol"):
    f = fund[fund.symbol.eq(asset)].drop(columns="symbol")
    p = prem[prem.symbol.eq(asset)].drop(columns="symbol")
    merged = pd.merge_asof(
        g.sort_values("decision_time"),
        f[["available_at", "funding_hourly", "funding_change"]],
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=26),
    ).rename(columns={"available_at": "funding_available_at"})
    merged = pd.merge_asof(
        merged,
        p[
            [
                "available_at",
                "premium_close",
                "premium_mean24",
                "premium_std24",
                "premium_change4",
            ]
        ],
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=3),
    ).rename(columns={"available_at": "premium_available_at"})
    parts.append(merged)
frame = (
    pd.concat(parts, ignore_index=True)
    .sort_values(["decision_time", "symbol"])
    .reset_index(drop=True)
)
frame.to_parquet(OUT / "enriched_features.parquet", index=False)
matched = frame.dropna(subset=EXTRA)
assert (matched.funding_available_at <= matched.decision_time).all()
assert (matched.premium_available_at <= matched.decision_time).all()
quality = {
    "total_rows": len(frame),
    "matched_rows": len(matched),
    "missing_exogenous_rows": len(frame) - len(matched),
    "first_complete": str(matched.decision_time.min()),
    "last_complete": str(matched.decision_time.max()),
    "extra_features": EXTRA,
    "all_matched_sources_available_by_origin": True,
}
(OUT / "quality.json").write_text(json.dumps(quality, indent=2))
print(quality, flush=True)
dev = matched[
    matched.label_end_time.lt(pd.Timestamp(split["test_start"]))
    & matched.target_return.notna()
].copy()


def rls_predict(train, stream, columns, scaler):
    X = scaler.transform(train[columns])
    y = train.target_return.to_numpy() / (
        100 * np.clip(train.vol_48.to_numpy() * 2, 0.001, 0.05)
    )
    P = np.linalg.inv(X.T @ X + 1000 * np.eye(len(columns)))
    w = P @ X.T @ y
    initial = (P.copy(), w.copy())
    all_rows = stream.sort_values("decision_time").copy()
    results = []
    for asset, part in all_rows.groupby("symbol"):
        part = part.sort_values("decision_time").reset_index(drop=True)
        current = part.decision_time.to_numpy()
        label = part.label_end_time.to_numpy()
        xx = scaler.transform(part[columns])
        scales = 100 * np.clip(part.vol_48.to_numpy() * 2, 0.001, 0.05)
        yy = part.target_return.to_numpy() / scales
        states = [(initial[0].copy(), initial[1].copy()) for _ in range(3)]
        j = 0
        last_used = None
        for i in range(len(part)):
            t = current[i]
            while j < i and label[j] < t:
                x = xx[j]
                for k, lamb in enumerate([1.0, 0.9999, 0.999]):
                    pp, ww = states[k]
                    px = pp @ x
                    gain = px / (lamb + x @ px)
                    ww = ww + gain * (yy[j] - x @ ww)
                    pp = (pp - np.outer(gain, px)) / lamb
                    states[k] = (pp, ww)
                last_used = label[j]
                j += 1
            pred = float(np.median([xx[i] @ ww for _, ww in states])) * scales[i]
            if not np.isfinite(pred):
                raise ValueError("Nonfinite RLS")
            if last_used is not None:
                assert pd.Timestamp(last_used) < pd.Timestamp(t)
            results.append(
                {
                    "symbol": asset,
                    "decision_time": t,
                    "prediction": pred,
                    "latest_online_label": last_used,
                    "coefficient_norm": float(
                        np.linalg.norm(np.mean([ww for _, ww in states], axis=0))
                    ),
                }
            )
    return pd.DataFrame(results)


summaries = []
started = time.time()
for name, columns in [("original", FEATURES), ("funding_premium", FEATURES + EXTRA)]:
    for family in ["ridge", "xgboost", "delayed_rls_bank"]:
        preds = []
        for k, fold in enumerate(split["tuning_folds"]):
            train, val = fold_data(
                dev, fold["validation_start"], fold["validation_end"]
            )
            context = with_context(dev, val, fold["validation_start"])
            val = context[context.is_validation & context.eligible_window].copy()
            scaler = StandardScaler().fit(train[columns])
            scale_train = 100 * np.clip(train.vol_48.to_numpy() * 2, 0.001, 0.05)
            scale_val = 100 * np.clip(val.vol_48.to_numpy() * 2, 0.001, 0.05)
            if family == "ridge":
                model = Ridge(alpha=1000, fit_intercept=False).fit(
                    scaler.transform(train[columns]), train.target_return / scale_train
                )
                val["prediction"] = (
                    model.predict(scaler.transform(val[columns])) * scale_val
                )
            elif family == "xgboost":
                model = xgb.XGBRegressor(
                    max_depth=2,
                    learning_rate=0.03,
                    n_estimators=200,
                    reg_lambda=100,
                    min_child_weight=500,
                    n_jobs=4,
                    random_state=42,
                )
                model.fit(train[columns], train.target_return / scale_train)
                val["prediction"] = model.predict(val[columns]) * scale_val
            else:
                stream = dev[
                    dev.label_end_time.ge(pd.Timestamp(fold["validation_start"]))
                    & dev.decision_time.lt(pd.Timestamp(fold["validation_end"]))
                ].copy()
                pp = rls_predict(train, stream, columns, scaler)
                val = val.merge(
                    pp, on=["symbol", "decision_time"], validate="one_to_one"
                )
            val["fold"] = k
            keep = [
                "symbol",
                "decision_time",
                "target_return",
                "prediction",
                "fold",
            ] + (
                ["latest_online_label", "coefficient_norm"]
                if family == "delayed_rls_bank"
                else []
            )
            preds.append(val[keep])
            print(
                name,
                family,
                "fold",
                k,
                "seconds",
                round(time.time() - started),
                flush=True,
            )
        prediction = pd.concat(preds, ignore_index=True)
        prediction.to_parquet(OUT / f"{family}-{name}.parquet", index=False)
        ratios = [
            float(
                ((g.prediction - g.target_return) ** 2).mean()
                / (g.target_return**2).mean()
            )
            for _, g in prediction.groupby("fold")
        ]
        summary = {
            "model": family + "-" + name,
            "rows": len(prediction),
            "mean_fold_mse_ratio": float(np.mean(ratios)),
            "fold_ratios": ratios,
            "weekly_interval": weekly_interval(prediction),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        (OUT / "results.json").write_text(json.dumps(summaries, indent=2))
(OUT / "status.json").write_text(
    json.dumps(
        {"status": "completed", "conditions": 6, "seconds": time.time() - started},
        indent=2,
    )
)
