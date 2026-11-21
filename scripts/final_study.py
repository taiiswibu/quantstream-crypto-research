"""Shared, reproducible calculations for the four portfolio notebooks.

Runs use a private research workspace. Outer validation is diagnostic only:
the TCN always retains epoch 6, without selecting a checkpoint on those labels.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from io import BytesIO
import shutil
import subprocess
import time
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from quantstream import terminal
from quantstream.backtest import evaluate_strategy, simulate
from quantstream.common import Config, sha256, utcnow, write_json
from quantstream.data import validate_bars
from quantstream.features import FEATURES, build_features

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('QUANTSTREAM_STUDY_ROOT', ROOT / 'notebook_work/final_study'))
REPORT = ROOT / 'reports/final_study'
COLORS = ['#168879', '#4566ac', '#b87829', '#a44966', '#78879c']
EPOCHS = 6


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.figsize': (11, 4)})


def show(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=115, bbox_inches='tight',
                facecolor='white', pil_kwargs={'optimize': True})
    display(Image(data=buffer.getvalue(), width=1100))
    plt.close(fig)


def diagram(name):
    if not shutil.which('dot'):
        raise RuntimeError('Cần Graphviz (lệnh dot) để dựng sơ đồ.')
    result = subprocess.run(['dot', '-Tpng', '-Gdpi=110',
                             str(ROOT / 'reports/diagrams' / (name + '.dot'))],
                            capture_output=True, check=True)
    display(Image(data=result.stdout, width=1000))


def prepare():
    source = Path(os.environ.get('QUANTSTREAM_HISTORY_DIR', ROOT / 'research_input'))
    path, mp = source / 'bars.parquet', source / 'history_manifest.json'
    if not path.is_file() or not mp.is_file():
        raise FileNotFoundError('Cần research_input/bars.parquet + history_manifest.json; xem README.')
    manifest = json.loads(mp.read_text(encoding='utf-8-sig'))
    if sha256(path) != manifest['bars_sha256']:
        raise ValueError('SHA256 của nến khác manifest. Không tiếp tục train.')
    bars, quarantine, quality = validate_bars(pd.read_parquet(path))
    if len(quarantine) or len(bars) != manifest['rows']:
        raise ValueError('Snapshot chứa dòng lỗi hoặc sai số lượng.')
    if set(bars.symbol) != {'BTCUSDT', 'ETHUSDT'}:
        raise ValueError('Workflow này yêu cầu BTCUSDT và ETHUSDT.')
    if (WORK / 'paper/state.json').exists() or (WORK / 'artifacts/local_registry.json').exists():
        raise ValueError('Thư mục nghiên cứu đã có model/paper. Chọn QUANTSTREAM_STUDY_ROOT mới để chạy lại.')
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = Config(end_month=bars.timestamp.max().strftime('%Y-%m'),
                 position_fraction=.10, drawdown_halt=.10)
    target = WORK / 'data/processed/bars.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    write_json(WORK / 'config.json', cfg.to_dict())
    write_json(WORK / 'data/processed/history_manifest.json', manifest)
    samples = build_features(bars, cfg)
    samples.to_parquet(WORK / 'features.parquet', index=False)
    write_json(REPORT / 'data_quality.json', quality)
    write_json(REPORT / 'snapshot.json', {
        'bars_sha256': sha256(path), 'rows': len(bars),
        'by_symbol': bars.groupby('symbol').size().to_dict(),
        'first_open': str(bars.timestamp.min()), 'last_open': str(bars.timestamp.max()),
        'feature_rows': len(samples), 'labelled_rows': int(samples.target_return.notna().sum()),
        'feature_code_sha256': sha256(ROOT / 'quantstream/features.py'),
        'feature_count': len(FEATURES), 'config': cfg.to_dict(),
        'source_note': 'Verified saved snapshot; no new network collection in this execution.',
        'terms_url': 'https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md'})
    return bars, samples, cfg, quality


def load_data():
    return (pd.read_parquet(WORK / 'data/processed/bars.parquet'),
            pd.read_parquet(WORK / 'features.parquet'), Config.load(WORK / 'config.json'))


def bounds(bars, cfg):
    recent = bars[bars.timestamp.ge(bars.timestamp.max() + pd.Timedelta(hours=1) - pd.Timedelta(days=90))]
    evaluation = build_features(recent, cfg).dropna(subset=['target_return', 'label_end_time'])
    times = np.sort(evaluation.decision_time.unique())
    edges = [pd.Timestamp(times[int(len(times) * f)]) for f in (.55, .70, .85)]
    edges.append(evaluation.label_end_time.max() + pd.Timedelta(hours=1))
    return list(zip(edges[:-1], edges[1:]))


def data_chart(bars, quality):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
    for (name, group), color in zip(bars.groupby('symbol'), COLORS):
        axes[0].plot(group.timestamp, group.close / group.close.iloc[0] * 100, label=name, color=color, lw=1)
    axes[0].set(title='Lịch sử giá trong snapshot 1H', ylabel='Giá đầu kỳ = 100', xlabel='UTC')
    axes[0].legend()
    counts = bars.groupby('symbol').size()
    axes[1].bar(counts.index, counts, color=COLORS[:2])
    for i, value in enumerate(counts):
        axes[1].text(i, value + 500, f'{value:,}', ha='center')
    axes[1].set(title=f"{sum(g['missing_bars'] for g in quality['gaps'])} nến thiếu được ghi nhận",
                ylabel='Nến hợp lệ', ylim=(0, 39000))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def features_chart(samples):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for (name, group), color in zip(samples.groupby('symbol'), COLORS):
        axes[0].hist(group.target_return.dropna() * 100, bins=100, range=(-5, 5),
                     density=True, alpha=.45, color=color, label=name)
    axes[0].set(title='Phân phối target return 4H', xlabel='Return (%)', ylabel='Mật độ vùng hiển thị')
    axes[0].legend()
    cols = ['ret_1', 'ret_4', 'vol_12', 'rsi_14', 'macd', 'ema_gap']
    heat = axes[1].imshow(samples[cols].corr(), vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1].set(xticks=range(len(cols)), xticklabels=cols, yticks=range(len(cols)),
                yticklabels=cols, title='Tương quan feature; không diễn giải nhân quả')
    axes[1].tick_params(axis='x', rotation=35)
    fig.colorbar(heat, ax=axes[1], shrink=.8)
    fig.tight_layout()
    return fig


def gap_chart(bars):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    center = pd.Timestamp('2023-03-24 12:00', tz='UTC')
    times = pd.date_range(center - pd.Timedelta(hours=6), center + pd.Timedelta(hours=8), freq='h')
    for i, (symbol, group) in enumerate(bars.groupby('symbol')):
        present = times.isin(group.timestamp)
        ax.scatter(times[present], np.full(present.sum(), i), marker='s', s=140, color=COLORS[i], label=symbol)
        ax.scatter(times[~present], np.full((~present).sum(), i), marker='x', s=140, color='#b54556')
    ax.set(title='Khoảng trống thực tế trong snapshot · dấu × là nến thiếu',
           yticks=[0, 1], yticklabels=['BTCUSDT', 'ETHUSDT'], ylim=(-.6, 1.6), xlabel='UTC')
    ax.grid(axis='x', alpha=.15)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def split_chart(bars, samples, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows = []
    for fold, (start, end) in enumerate(bounds(bars, cfg)):
        train = samples[samples.label_end_time.lt(start)]
        val = samples[samples.decision_time.ge(start) & samples.label_end_time.lt(end)]
        assert train.label_end_time.max() < val.decision_time.min()
        rows.append({'fold': fold + 1, 'train_rows': len(train), 'validation_rows': len(val),
                     'last_train_label': str(train.label_end_time.max()),
                     'validation_start': str(start), 'validation_end_exclusive': str(end)})
        axes[0].plot([samples.decision_time.min(), train.decision_time.max()], [fold + 1] * 2,
                     color='#aebdce', lw=14)
        axes[0].plot([start, end], [fold + 1] * 2, color=COLORS[fold], lw=14)
    axes[0].set(title='Expanding train từ 2023; ba fold gần nhất', xlabel='UTC', yticks=[1, 2, 3], ylim=(.5, 3.5))
    train = samples[samples.label_end_time.lt(bounds(bars, cfg)[0][0])]
    raw = train[['ret_1', 'rsi_14', 'volume_z']]
    z = (raw - raw.mean()) / raw.std().clip(lower=1e-6)
    axes[1].boxplot([z[c].to_numpy() for c in z], tick_labels=list(z.columns), showfliers=False)
    axes[1].set(title='Z-score fit riêng trên train fold 1', ylabel='Độ lệch chuẩn', ylim=(-4, 4))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    table = pd.DataFrame(rows)
    table.to_csv(REPORT / 'splits.csv', index=False)
    return table, fig


def train_tabular():
    if (WORK / 'artifacts/local_registry.json').exists():
        raise ValueError('Đã có kết quả; dùng thư mục mới nếu muốn huấn luyện lại.')
    registry = terminal.fit_local(WORK)
    # This rerun is diagnostic research, not an approved investment signal.
    for meta in registry.values():
        meta.update(signal_approved=False, deployment_status='research_validation_failed',
                    artifact_role='all_observed_refit_no_independent_final_test')
    write_json(WORK / 'artifacts/local_registry.json', registry)
    write_json(WORK / 'artifacts/champion.json', registry['ridge'])
    run = WORK / registry['ridge']['run']
    write_json(WORK / 'study.json', {'run': registry['ridge']['run'], 'epochs': EPOCHS})
    return registry, run


def run_path():
    return WORK / json.loads((WORK / 'study.json').read_text())['run']


def train_tcn():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from quantstream.deep import TCN, Windows, predict_tcn
    bars, samples, cfg = load_data()
    run = run_path()
    reference = pd.read_parquet(run / 'validation_predictions.parquet')
    reference = reference[reference.model.eq('ridge')]
    labelled = samples[samples.target_return.notna() & samples.label_end_time.notna()]
    histories, predictions, fit_rows = [], [], []
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    def fit(train, fold, context=None, validation_ids=None):
        torch.manual_seed(cfg.seed)
        mean = train[FEATURES].mean().to_numpy(np.float32)
        std = train[FEATURES].std().clip(lower=1e-6).to_numpy(np.float32)
        ds = Windows(train, mean, std, cfg.lookback)
        loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(cfg.seed))
        model = TCN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        criterion = nn.HuberLoss(delta=1.)
        steps = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total, seen = 0., 0
            for x, y, _ in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                steps += 1
                total += loss.item() * len(y)
                seen += len(y)
            row = {'fold': str(fold), 'epoch': epoch, 'train_huber': total / seen,
                   'optimizer_steps_cumulative': steps, 'training_windows': len(ds)}
            if context is not None:
                pred = pd.Series(predict_tcn(model, context, mean, std, cfg.lookback), index=context.index)
                target = samples.loc[validation_ids, 'target_return']
                values = pred.loc[validation_ids].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Thiếu context cho một số mẫu validation TCN.')
                row['validation_mse'] = float(np.mean((values - target.to_numpy()) ** 2))
            histories.append(row)
            print(f"TCN {fold} epoch {epoch}/{EPOCHS}: Huber={row['train_huber']:.6f}", flush=True)
        fit_rows.append({'fold': str(fold), 'rows': len(train), 'windows': len(ds), 'optimizer_steps': steps})
        return model, mean, std

    for fold, keys in reference.groupby('fold', sort=True):
        val = keys[['symbol', 'decision_time']].merge(
            samples.reset_index(names='sample_id'), on=['symbol', 'decision_time'], validate='one_to_one')
        ids = val.sample_id.to_numpy()
        start, end = bounds(bars, cfg)[int(fold)]
        train = labelled[labelled.label_end_time.lt(start)]
        assert train.label_end_time.max() < val.decision_time.min()
        context = samples[samples.decision_time.ge(start - pd.Timedelta(hours=cfg.lookback - 1))
                          & samples.decision_time.le(val.decision_time.max())].copy()
        context.loc[~context.index.isin(ids), 'target_return'] = np.nan
        model, mean, std = fit(train, int(fold) + 1, context, ids)
        pred = pd.Series(predict_tcn(model, context, mean, std, cfg.lookback), index=context.index)
        part = keys.copy()
        part['prediction'], part['model'] = pred.loc[ids].to_numpy(), 'tcn_fixed6'
        np.testing.assert_allclose(part.target_return, val.target_return, rtol=0, atol=1e-14)
        predictions.append(part)
    model, mean, std = fit(labelled, 'final_refit')
    folder = WORK / 'artifacts/study_tcn'
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / 'model.pt'
    torch.save({'state_dict': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(),
                'lookback': cfg.lookback}, weights)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_parquet(run / 'tcn_predictions.parquet', index=False)
    history = pd.DataFrame(histories)
    history.to_csv(REPORT / 'tcn_learning.csv', index=False)
    write_json(folder / 'study_metadata.json', {
        'created_at': utcnow(), 'sha256': sha256(weights), 'type': 'tcn_research_refit',
        'features': FEATURES, 'config': cfg.to_dict(), 'fixed_epochs': EPOCHS,
        'checkpoint_rule': 'always_last_epoch_no_outer_validation_selection',
        'artifact_role': 'all_observed_refit_no_independent_final_test',
        'signal_approved': False, 'training_rows': len(labelled), 'fits': fit_rows,
        'bars_sha256': sha256(WORK / 'data/processed/bars.parquet'),
        'fit_seconds': time.monotonic() - started,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'standalone_only': True,
        'note': 'Not eligible for legacy holdout-preserving import-tcn; no false holdout claim.'})
    return history, pd.DataFrame(fit_rows)


def learning_charts():
    bars, samples, cfg = load_data()
    start, end = bounds(bars, cfg)[0]
    train = samples[samples.target_return.notna() & samples.label_end_time.lt(start)]
    val = samples[samples.decision_time.ge(start) & samples.label_end_time.lt(end)]
    model = terminal.make_estimator('xgboost_d3', cfg)
    model.fit(train[FEATURES], train.target_return,
              eval_set=[(train[FEATURES], train.target_return), (val[FEATURES], val.target_return)], verbose=False)
    result = model.evals_result()
    log = pd.DataFrame({'round': np.arange(1, 121), 'train_rmse': result['validation_0']['rmse'],
                        'validation_rmse': result['validation_1']['rmse']})
    log.to_csv(REPORT / 'xgboost_learning.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    for column, color in zip(['train_rmse', 'validation_rmse'], COLORS):
        ax.plot(log['round'], log[column] * 100, label=column, color=color)
    ax.set(title='XGBoost fold 1 · 120 rounds cố định; không early stopping',
           xlabel='Boosting round', ylabel='RMSE (điểm % return)')
    ax.legend()
    ax.grid(alpha=.15)
    show(fig)


def tcn_learning_chart():
    log = pd.read_csv(REPORT / 'tcn_learning.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (fold, group), color in zip(log.groupby('fold', sort=True), COLORS):
        axes[0].plot(group.epoch, group.train_huber, marker='o', label=fold, color=color)
        if group.validation_mse.notna().any():
            axes[1].plot(group.epoch, group.validation_mse, marker='o', label=fold, color=color)
    axes[0].set(title='TCN mới chạy CPU · Huber trên target × 100', xlabel='Epoch', ylabel='Huber loss')
    axes[1].set(title='Outer validation chỉ theo dõi; giữ epoch 6', xlabel='Epoch', ylabel='MSE (return)')
    for ax in axes:
        ax.legend(title='Fold / final refit')
        ax.grid(alpha=.15)
    fig.tight_layout()
    return fig


def tuning_chart():
    """Small inner temporal search; does not alter the outer-CV fixed baseline."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    bars, samples, cfg = load_data()
    outer_start = bounds(bars, cfg)[0][0]
    past = samples[samples.label_end_time.lt(outer_start)]
    inner_start = past.decision_time.max() - pd.Timedelta(days=28)
    train = past[past.label_end_time.lt(inner_start)]
    val = past[past.decision_time.ge(inner_start)]
    assert train.label_end_time.max() < val.decision_time.min()
    assert val.label_end_time.max() < outer_start
    rows = []
    for alpha in (1., 10., 100.):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[FEATURES], train.target_return)
        score = terminal.forecast_metrics(val.target_return, model.predict(val[FEATURES]))
        rows.append({'model': 'ridge', 'parameter': alpha, **score})
    for depth in (2, 3, 5):
        model = terminal.make_estimator('xgboost_d3', cfg).set_params(max_depth=depth)
        model.fit(train[FEATURES], train.target_return)
        score = terminal.forecast_metrics(val.target_return, model.predict(val[FEATURES]))
        rows.append({'model': 'xgboost_d3', 'parameter': depth, **score})
    table = pd.DataFrame(rows)
    table.to_csv(REPORT / 'inner_tuning.csv', index=False)
    write_json(REPORT / 'inner_tuning_split.json', {
        'train_rows': len(train), 'validation_rows': len(val),
        'last_train_label': str(train.label_end_time.max()),
        'inner_validation_start': str(val.decision_time.min()),
        'inner_last_label': str(val.label_end_time.max()), 'outer_start': str(outer_start),
        'used_to_replace_fixed_baseline': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name, label, color in zip(axes, ['ridge', 'xgboost_d3'], ['Ridge alpha', 'XGBoost max_depth'], COLORS):
        group = table[table.model.eq(name)]
        ax.plot(group.parameter, group.mse_ratio_to_zero, marker='o', color=color)
        ax.axhline(1, ls='--', color='#a44966')
        ax.set(title='Inner validation trước outer CV', xlabel=label, ylabel='MSE / MSE zero')
        ax.grid(alpha=.15)
    fig.tight_layout()
    return table, fig


def evaluate():
    bars, samples, cfg = load_data()
    run = run_path()
    oof = pd.concat([pd.read_parquet(run / 'validation_predictions.parquet'),
                     pd.read_parquet(run / 'tcn_predictions.parquet')], ignore_index=True)
    reference = oof[oof.model.eq('ridge')].reset_index(drop=True)
    for _, group in oof.groupby('model', sort=False):
        pd.testing.assert_frame_equal(group[['symbol', 'decision_time', 'target_return']].reset_index(drop=True),
                                      reference[['symbol', 'decision_time', 'target_return']])
    keyed = reference[['symbol', 'decision_time']].merge(samples, on=['symbol', 'decision_time'], validate='one_to_one')
    for name, values in [('zero_return', np.zeros(len(reference))), ('naive_ret4', np.expm1(keyed.ret_4))]:
        oof = pd.concat([oof, reference.assign(model=name, prediction=np.asarray(values))], ignore_index=True)
    rows, fold_rows, backtests = [], [], []
    for name, group in oof.groupby('model', sort=False):
        metrics = terminal.forecast_metrics(group.target_return, group.prediction)
        ratios = []
        for fold, part in group.groupby('fold'):
            value = terminal.forecast_metrics(part.target_return, part.prediction)['mse_ratio_to_zero']
            ratios.append(value)
            fold_rows.append({'model': name, 'fold': int(fold) + 1, 'mse_ratio_to_zero': value})
        passed = metrics['mse_ratio_to_zero'] < 1 and all(r < 1 for r in ratios)
        rows.append({'model': name, **metrics, 'beats_zero_every_fold': passed,
                     'independent_final_test': False, 'signal_approved': False})
        if name in ('zero_return', 'naive_ret4'):
            continue
        joined = group.drop(columns='target_return').merge(samples, on=['symbol', 'decision_time'],
                    suffixes=('_stored', ''), validate='one_to_one')
        start, end = group.decision_time.min(), group.label_end_time.max()
        for threshold in (.003, .006, .009):
            curve, trades, summary = evaluate_strategy(bars, joined, joined.prediction, threshold, start, end, cfg)
            backtests.append({'model': name, 'threshold': threshold, **summary})
            if threshold == .006:
                curve.rename('equity').to_frame().reset_index().to_csv(REPORT / (name + '_equity.csv'), index=False)
                trades.to_csv(REPORT / (name + '_trades.csv'), index=False)
    empty = pd.DataFrame(columns=['symbol', 'decision_time', 'go_long'])
    start, end = reference.decision_time.min(), reference.label_end_time.max()
    hold, _, hold_metrics = simulate(bars, empty, start, end, cfg, buy_hold=True)
    ema, _, ema_metrics = evaluate_strategy(bars, keyed, keyed.ema_gap, 0., start, end, cfg)
    pd.DataFrame({'timestamp': hold.index, 'buy_hold': hold.values, 'ema': ema.reindex(hold.index).values}).to_csv(
        REPORT / 'strategy_baselines.csv', index=False)
    write_json(REPORT / 'strategy_baselines.json', {'buy_hold': hold_metrics, 'ema': ema_metrics})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT / 'metrics.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(REPORT / 'fold_metrics.csv', index=False)
    pd.DataFrame(backtests).to_csv(REPORT / 'backtests.csv', index=False)
    oof.to_parquet(WORK / 'oof.parquet', index=False)
    return metrics, oof


def baseline_chart(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field, title, ylabel in [(axes[0], 'mse_ratio_to_zero', 'MSE so với zero-return', 'MSE / MSE zero'),
                                     (axes[1], 'mae', 'MAE trên cùng cohort', 'MAE (điểm % return)')]:
        values = metrics[field] * (100 if field == 'mae' else 1)
        ax.bar(metrics.model, values, color=COLORS)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.18))
        for i, value in enumerate(values):
            ax.text(i, value + max(values) * .025, f'{value:.4f}', ha='center', fontsize=9)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].axhline(1, color='#a44966', ls='--')
    fig.tight_layout()
    return fig


def fold_chart():
    data = pd.read_csv(REPORT / 'fold_metrics.csv')
    table = data[~data.model.isin(['zero_return', 'naive_ret4'])].pivot(index='model', columns='fold', values='mse_ratio_to_zero')
    fig, ax = plt.subplots(figsize=(10, 4))
    table.plot.bar(ax=ax, color=COLORS)
    ax.axhline(1, color='#a44966', ls='--', label='Zero-return')
    ax.set(title='Ổn định theo fold quan trọng hơn một giá trị trung bình', ylabel='MSE ratio', xlabel='Model')
    ax.tick_params(axis='x', rotation=0)
    ax.legend()
    fig.tight_layout()
    return fig


def prediction_charts(oof):
    models = oof[~oof.model.isin(['zero_return', 'naive_ret4'])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (name, group), color in zip(models.groupby('model'), COLORS):
        axes[0].scatter(group.target_return * 100, group.prediction * 100, s=8, alpha=.25, color=color, label=name)
        axes[1].hist((group.prediction - group.target_return) * 100, bins=70, alpha=.4, color=color, label=name)
    axes[0].set(title='Dự báo và nhãn OOF', xlabel='Thực tế (%)', ylabel='Dự báo (%)')
    axes[1].set(title='Phân phối sai số', xlabel='Dự báo − thực tế (điểm %)', ylabel='Số mẫu')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.15)
    fig.tight_layout()
    show(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    btc = models[models.symbol.eq('BTCUSDT')]
    truth = btc[btc.model.eq('ridge')].tail(120)
    ax.plot(truth.decision_time, truth.target_return * 100, label='Nhãn thực tế', color='#82909c', lw=1)
    for (name, group), color in zip(btc.groupby('model'), COLORS):
        group = group.tail(120)
        ax.plot(group.decision_time, group.prediction * 100, label=name, color=color, lw=1)
    ax.set(title='BTCUSDT · 120 dự báo OOF cuối kỳ; nhãn chồng lấn 4H', ylabel='Return (%)', xlabel='UTC')
    ax.legend(ncol=4)
    ax.grid(alpha=.15)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    show(fig)


def strategy_charts():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, color in zip(['ridge', 'xgboost_d3', 'tcn_fixed6'], COLORS):
        data = pd.read_csv(REPORT / (name + '_equity.csv'))
        axes[0].plot(pd.to_datetime(data.iloc[:, 0], utc=True), data.equity, label=name, color=color)
    base = pd.read_csv(REPORT / 'strategy_baselines.csv')
    for name, color in zip(['buy_hold', 'ema'], COLORS[3:]):
        axes[0].plot(pd.to_datetime(base.timestamp, utc=True), base[name], label=name, color=color, lw=1)
    axes[0].set(title='Mô phỏng sau phí · ngưỡng cố định 0,6%', ylabel='Equity đầu kỳ = 1', xlabel='UTC')
    axes[0].legend(fontsize=8)
    table = pd.read_csv(REPORT / 'backtests.csv')
    for (name, group), color in zip(table.groupby('model'), COLORS):
        axes[1].plot(group.threshold * 100, group.trades, marker='o', label=name, color=color)
    axes[1].set(title='Độ nhạy ngưỡng; không chọn ngưỡng từ biểu đồ này', xlabel='Ngưỡng return (%)', ylabel='Số lệnh mô phỏng')
    axes[1].legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def feature_chart():
    registry = json.loads((WORK / 'artifacts/local_registry.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ['ridge', 'xgboost_d3']):
        model = joblib.load(WORK / registry[name]['path'])['model']
        values = model[-1].coef_ if name == 'ridge' else model.feature_importances_
        series = pd.Series(values, index=FEATURES).sort_values(key=abs).tail(10)
        ax.barh(series.index, series.values, color=[COLORS[0] if v >= 0 else COLORS[2] for v in series])
        ax.set(title=name + ' · trọng số/importance của final refit')
    axes[0].set_xlabel('Hệ số trên feature đã chuẩn hóa')
    axes[1].set_xlabel('Importance tương đối của cây')
    fig.tight_layout()
    return fig


def verify_inference():
    from quantstream.deep import predict_tcn_file
    bars, samples, cfg = load_data()
    registry = json.loads((WORK / 'artifacts/local_registry.json').read_text())
    rows = []
    latest = samples.sort_values('decision_time').groupby('symbol').tail(1)
    for name in ['ridge', 'xgboost_d3']:
        meta = registry[name]
        path = WORK / meta['path']
        if sha256(path) != meta['sha256']:
            raise ValueError('Model hash mismatch')
        first, second = joblib.load(path), joblib.load(path)
        a, b = first['model'].predict(latest[FEATURES]), second['model'].predict(latest[FEATURES])
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        for (_, row), value in zip(latest.iterrows(), a):
            rows.append({'model': name, 'symbol': row.symbol, 'decision_time': row.decision_time,
                         'prediction': value, 'reload_max_abs_diff': float(np.max(abs(a - b))),
                         'sha256': meta['sha256'], 'signal_approved': False})
    context = samples.sort_values('decision_time').groupby('symbol').tail(cfg.lookback).sort_index()
    weights = WORK / 'artifacts/study_tcn/model.pt'
    metadata = json.loads((weights.parent / 'study_metadata.json').read_text())
    assert sha256(weights) == metadata['sha256']
    a, b = predict_tcn_file(weights, context), predict_tcn_file(weights, context)
    ids = [context.index.get_loc(i) for i in latest.index]
    np.testing.assert_allclose(a[ids], b[ids], rtol=0, atol=0)
    for (_, row), value in zip(latest.iterrows(), a[ids]):
        rows.append({'model': 'tcn_fixed6', 'symbol': row.symbol, 'decision_time': row.decision_time,
                     'prediction': value, 'reload_max_abs_diff': 0., 'sha256': sha256(weights), 'signal_approved': False})
    table = pd.DataFrame(rows)
    now = pd.Timestamp.now(tz='UTC')
    table['fresh_at_execution'] = table.decision_time.apply(lambda t: 0 <= (now - t).total_seconds() <= 3720)
    table.to_csv(REPORT / 'inference.csv', index=False)
    # Populate the same serving file used by the app. Old snapshot timestamps
    # remain old; terminal.predict_latest does not insert stale forecasts as live.
    terminal.predict_latest(WORK)
    shutil.copy2(WORK / 'artifacts/study_tcn/study_metadata.json', REPORT / 'tcn_artifact.json')
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'torch', 'streamlit']}
    import xgboost
    versions['xgboost'] = xgboost.__version__
    write_json(REPORT / 'execution.json', {'executed_at': utcnow(), 'runtime_packages': versions,
        'snapshot_sha256': sha256(WORK / 'data/processed/bars.parquet'), 'tcn_fixed_epochs': EPOCHS,
        'all_remaining_notebooks': ['01_Data', '02_Training', '03_Evaluation', '04_Inference'],
        'source_files_sha256': {p: sha256(ROOT / p) for p in ['scripts/final_study.py', 'quantstream/features.py', 'quantstream/deep.py', 'quantstream/terminal.py']},
        'source_control': {'status': 'verified_source_copy_with_local_changes',
                           'base_commit': '317895640d259da580eeeafb4ce6dd273246741a'},
        'independent_final_test': False, 'real_orders': False})
    return table


def inference_chart(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    table.pivot(index='model', columns='symbol', values='prediction').mul(100).plot.bar(ax=axes[0], color=COLORS[:2])
    axes[0].set(title='Inference từ artifact đã reload · snapshot cũ', ylabel='Dự báo return (%)', xlabel='Model')
    axes[0].tick_params(axis='x', rotation=15)
    axes[1].axis('off')
    text = ('KIỂM TRA ARTIFACT\n\n3 model × 2 tài sản\nSHA256 trùng metadata\nReload prediction parity: 0\n\nDữ liệu snapshot đã cũ tại lúc chạy\nKhông coi đây là tín hiệu thị trường live\nKhông có test độc lập mới\nKhông phê duyệt lệnh đầu tư')
    axes[1].text(.05, .95, text, va='top', fontsize=12, linespacing=1.6,
                 bbox={'boxstyle': 'round,pad=1', 'fc': '#edf5f3', 'ec': '#9dbab4'})
    fig.tight_layout()
    return fig


def package():
    """Bundle an isolated demo workspace; never overwrite the user's paper session."""
    path = WORK / 'QuantStream_Research_Workspace.zip'
    roots = [WORK / 'config.json', WORK / 'latest_forecasts.json', WORK / 'data/processed/bars.parquet',
             WORK / 'data/processed/history_manifest.json']
    roots += [p for folder in ['artifacts', 'runs'] for p in (WORK / folder).rglob('*') if p.is_file()]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(roots):
            archive.write(file, file.relative_to(WORK).as_posix())
    result = {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path),
              'usage': 'Extract to a NEW directory and set QUANTSTREAM_ROOT; see README.'}
    write_json(REPORT / 'workspace_bundle.json', result)
    return result
