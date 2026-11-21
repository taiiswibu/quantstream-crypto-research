"""Calculations used by the project evidence notebook."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from quantstream import terminal
from quantstream.common import Config, write_json
from quantstream.data import validate_bars
from quantstream.features import FEATURES, build_features

COLORS = ['#147d73', '#4365ad', '#8b98a9', '#b67b27']


def snapshot(folder):
    folder = Path(folder)
    bars_path = folder / 'bars.parquet'
    manifest_path = folder / 'history_manifest.json'
    if not bars_path.exists() or not manifest_path.exists():
        raise FileNotFoundError('Cần bars.parquet và history_manifest.json trong thư mục snapshot.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    digest = hashlib.sha256(bars_path.read_bytes()).hexdigest()
    if digest != manifest['bars_sha256']:
        raise ValueError('Snapshot không khớp SHA256 trong history_manifest.json')
    bars = pd.read_parquet(bars_path)
    clean, quarantine, quality = validate_bars(bars)
    assert len(quarantine) == 0
    assert len(clean) == len(bars) == manifest['rows']
    return clean, manifest, quality


def train_workspace(bars, work):
    work = Path(work)
    if (work / 'paper/state.json').exists():
        raise ValueError('Dùng thư mục nghiên cứu riêng, không dùng phiên paper đang hoạt động.')
    path = work / 'data/processed/bars.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path, index=False)
    cfg = Config(end_month='2026-11', position_fraction=.10, drawdown_halt=.10)
    write_json(work / 'config.json', cfg.to_dict())
    registry = terminal.fit_local(work)
    run = work / registry['ridge']['run']
    samples = build_features(bars, cfg)
    metrics = pd.read_json(run / 'validation_metrics.json')
    oof = pd.read_parquet(run / 'validation_predictions.parquet')
    return cfg, registry, run, samples, metrics, oof


def show(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight',
                facecolor='white', pil_kwargs={'optimize': True})
    display(Image(data=buffer.getvalue(), width=1100))
    plt.close(fig)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.figsize': (11, 4),
                         'figure.facecolor': 'white'})


def data_figure(bars, quality):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
    for (symbol, group), color in zip(bars.groupby('symbol'), COLORS):
        group = group.sort_values('timestamp')
        axes[0].plot(group.timestamp, group.close / group.close.iloc[0] * 100,
                     label=symbol, color=color, linewidth=1)
    axes[0].set(title='Snapshot nến 1H đã kiểm tra', ylabel='Giá chuẩn hóa đầu kỳ = 100', xlabel='UTC')
    axes[0].set_xlim(bars.timestamp.min(), bars.timestamp.max())
    axes[0].legend()
    counts = bars.groupby('symbol').size()
    axes[1].bar(counts.index, counts.values, color=COLORS[:2])
    for i, value in enumerate(counts):
        axes[1].text(i, value + 500, f'{value:,}', ha='center')
    missing = sum(g['missing_bars'] for g in quality['gaps'])
    axes[1].set(title=f'{missing} nến thiếu được ghi nhận', ylabel='Số nến hợp lệ', ylim=(0, 39000))
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def feature_figure(samples, oof):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (symbol, group), color in zip(samples.groupby('symbol'), COLORS):
        axes[0].hist(group.target_return.dropna() * 100, bins=100, density=True,
                     alpha=.45, label=symbol, color=color, range=(-5, 5))
    axes[0].set(title='Phân phối nhãn return 4H', xlabel='Return (%)', ylabel='Mật độ trong vùng hiển thị')
    axes[0].legend()
    reference = oof[oof.model.eq('ridge')]
    recent_start = samples.decision_time.max() - pd.Timedelta(days=90)
    end = reference.label_end_time.max()
    for fold, group in reference.groupby('fold'):
        start = group.decision_time.min()
        finish = group.label_end_time.max()
        axes[1].plot([recent_start, start], [fold + 1] * 2, color='#a8b7c8', lw=15)
        axes[1].plot([start, finish], [fold + 1] * 2, color=COLORS[int(fold)], lw=15)
    axes[1].set(title='Walk-forward trên đoạn 90 ngày gần nhất', xlabel='UTC',
                yticks=[1, 2, 3], yticklabels=['Fold 1', 'Fold 2', 'Fold 3'], ylim=(.5, 3.5))
    axes[1].text(.02, -.30, 'Train còn có lịch sử từ 2023; nhãn chồng biên được purge.',
                 transform=axes[1].transAxes, fontsize=9)
    axes[1].set_xlim(recent_start, end)
    axes[1].tick_params(axis='x', rotation=20)
    fig.tight_layout()
    return fig


def baseline_figure(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(metrics.model, metrics.mae * 100, color=COLORS)
    axes[0].set(title='MAE trên cùng các mẫu OOF', ylabel='MAE (điểm % return)')
    axes[1].bar(metrics.model, metrics.mse_ratio_to_zero, color=COLORS)
    axes[1].axhline(1, color='#a63c3c', ls='--', label='Zero-return')
    axes[1].set(title='MSE so baseline zero-return', ylabel='MSE / zero-return MSE')
    axes[1].legend()
    for i, row in metrics.reset_index(drop=True).iterrows():
        axes[0].text(i, row.mae * 100 + .015, f'{row.mae * 100:.4f}%', ha='center', fontsize=9)
        axes[1].text(i, row.mse_ratio_to_zero + .025, f'{row.mse_ratio_to_zero:.4f}', ha='center', fontsize=9)
    axes[0].set_ylim(0, metrics.mae.max() * 100 * 1.15)
    axes[1].set_ylim(0, metrics.mse_ratio_to_zero.max() * 1.15)
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def forecast_figure(oof):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (model, group), color in zip(oof.groupby('model'), COLORS):
        axes[0].scatter(group.target_return * 100, group.prediction * 100,
                        s=8, alpha=.25, label=model, color=color)
        axes[1].hist((group.prediction - group.target_return) * 100,
                     bins=60, alpha=.45, label=model, color=color)
    axes[0].set(title='Dự báo OOF và nhãn thực tế', xlabel='Return thực tế (%)', ylabel='Return dự báo (%)')
    axes[1].set(title='Phân phối sai số', xlabel='Dự báo trừ thực tế (điểm %)', ylabel='Số mẫu')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.15)
    fig.tight_layout()
    return fig


def learning_curve(bars, samples, oof, cfg):
    start = oof[oof.fold.eq(0)].decision_time.min()
    train = samples[samples.target_return.notna() & samples.label_end_time.lt(start)]
    keys = oof[oof.model.eq('ridge') & oof.fold.eq(0)][['symbol', 'decision_time']]
    validation = keys.merge(samples, on=['symbol', 'decision_time'], validate='one_to_one')
    assert train.label_end_time.max() < validation.decision_time.min()
    model = terminal.make_estimator('xgboost_d3', cfg)
    model.fit(train[FEATURES], train.target_return,
              eval_set=[(train[FEATURES], train.target_return),
                        (validation[FEATURES], validation.target_return)], verbose=False)
    result = model.evals_result()
    table = pd.DataFrame({'round': np.arange(1, 121),
                          'train_rmse': result['validation_0']['rmse'],
                          'validation_rmse': result['validation_1']['rmse']})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(table['round'], table.train_rmse * 100, label='Train', color=COLORS[0])
    ax.plot(table['round'], table.validation_rmse * 100, label='Validation fold 1', color=COLORS[1])
    ax.set(title='XGBoost 120 vòng cố định trên fold 1', xlabel='Boosting round', ylabel='RMSE (điểm % return)')
    ax.legend()
    ax.grid(alpha=.15)
    fig.tight_layout()
    return table, fig


def equity_figure(run):
    curve = pd.read_parquet(Path(run) / 'ridge_equity.parquet')
    report = json.loads((Path(run) / 'ridge_backtest.json').read_text())
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    for key, color in zip(['model', 'buy_hold', 'ema'], COLORS):
        axes[0].plot(curve.timestamp, curve[key], label=key, color=color)
    axes[0].set(title=f"Validation sau phí · Ridge có {report['strategy']['trades']} lệnh",
                ylabel='Equity đầu kỳ = 1')
    axes[0].legend()
    drawdown = curve.model / curve.model.cummax().clip(lower=1) - 1
    axes[1].fill_between(curve.timestamp, drawdown * 100, 0, color='#a63c3c', alpha=.3)
    axes[1].set(xlabel='UTC', ylabel='Drawdown (%)')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    return fig, report


def tcn_history_figure(root):
    meta = json.loads((Path(root) / 'reports/reference/tcn_metadata.json').read_text())
    table = pd.DataFrame([{'fold': fold['fold'] + 1, **row}
                          for fold in meta['history'] for row in fold['epochs']])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fold, group in table.groupby('fold'):
        axes[0].plot(group.epoch, group.train_huber, marker='o', label=f'Fold {fold}')
        axes[1].plot(group.epoch, group.validation_mse, marker='o', label=f'Fold {fold}')
    axes[0].set(title='TCN tham chiếu · train log đã lưu', xlabel='Epoch', ylabel='Huber loss trên target × 100')
    axes[1].set(title='TCN tham chiếu · validation đã lưu', xlabel='Epoch', ylabel='MSE (return)')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.15)
    fig.tight_layout()
    return table, fig


def architecture(root):
    root = Path(root)
    source = root / 'docs/architecture.dot'
    if not shutil.which('dot'):
        raise RuntimeError('Cần Graphviz dot để dựng lại sơ đồ hệ thống trong notebook.')
    return subprocess.run(['dot', '-Tpng', '-Gdpi=120', str(source)],
                          check=True, capture_output=True).stdout
