"""Runtime contracts: no hindsight logging, no stale fills, no silent model switch."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn

from quantstream import terminal
from quantstream.common import Config, sha256, write_json
from test_integrity import fixture_bars
from test_paper_monitor import setup_paper
from quantstream import paper


def test_pause_cancels_pending_and_still_exits_open_positions(tmp_path, monkeypatch):
    cfg, now = setup_paper(tmp_path, monkeypatch)
    one = paper.paper_step(tmp_path, cfg, now=now, refresh_data=False, price_provider=lambda s: 100)
    due = pd.Timestamp(one['assets']['BTCUSDT']['pending']['due'])
    paused = paper.paper_step(tmp_path, cfg, now=due, refresh_data=False, price_provider=lambda s: 100, allow_new_entries=False)
    assert paused['assets']['BTCUSDT']['pending'] is None
    assert paused['assets']['BTCUSDT']['qty'] == 0
    assert paused['events'][-1]['side'] == 'CANCEL'
    paused['assets']['BTCUSDT'].update(qty=1, cash=4900, exit_due=str(due))
    write_json(tmp_path / 'paper/state.json', paused)
    closed = paper.paper_step(tmp_path, cfg, now=due, refresh_data=False, price_provider=lambda s: 100, allow_new_entries=False)
    assert closed['assets']['BTCUSDT']['qty'] == 0
    assert closed['events'][-1]['side'] == 'SELL'


def test_failed_download_keeps_existing_candles(tmp_path, monkeypatch):
    cfg = replace(Config(), symbols=('BTCUSDT',))
    write_json(tmp_path/'config.json', cfg.to_dict())
    path = tmp_path/'data/processed/bars.parquet'
    path.parent.mkdir(parents=True)
    fixture_bars(300).to_parquet(path, index=False)
    original = path.read_bytes()
    def offline(*args):
        raise OSError('offline')
    monkeypatch.setattr(terminal, 'fetch_symbol', offline)
    with pytest.raises(OSError, match='offline'):
        terminal.update_market(tmp_path)
    assert path.read_bytes() == original


def test_forecast_log_is_idempotent_and_excludes_stale_history(tmp_path, monkeypatch):
    cfg = replace(Config(), symbols=('BTCUSDT',))
    write_json(tmp_path/'config.json', cfg.to_dict())
    path = tmp_path/'data/processed/bars.parquet'
    path.parent.mkdir(parents=True)
    bars = fixture_bars(300)
    bars.to_parquet(path, index=False)
    meta = {'model':'ridge', 'sha256':'model-a', 'sklearn_version':sklearn.__version__,
            'feature_code_sha256':sha256(Path(terminal.__file__).parent/'features.py')}
    monkeypatch.setattr(terminal, 'load_metadata', lambda root:(meta, None))
    monkeypatch.setattr(terminal, 'predict_champion', lambda root, rows:np.zeros(len(rows)))
    assert not terminal.predict_latest(tmp_path)[0]['fresh']
    assert terminal.forecast_history(tmp_path).empty
    delta = pd.Timestamp.now(tz='UTC').floor('h') - pd.Timedelta(hours=1) - bars.timestamp.iloc[-1]
    for col in ('timestamp', 'source_close_time'):
        bars[col] += delta
    bars.to_parquet(path,index=False)
    terminal.predict_latest(tmp_path)
    terminal.predict_latest(tmp_path)
    history = terminal.forecast_history(tmp_path)
    assert len(history) == 1
    assert history.iloc[0].model_sha256 == 'model-a'
    meta['feature_code_sha256'] = 'modified-code'
    with pytest.raises(ValueError, match='Feature code changed'):
        terminal.predict_latest(tmp_path)
    assert len(terminal.forecast_history(tmp_path)) == 1


def test_model_switch_and_archive_refuse_open_paper(tmp_path):
    write_json(tmp_path/'artifacts/local_registry.json', {'ridge':{'sha256':'new'}})
    write_json(tmp_path/'paper/state.json', {'model_sha256':'old','assets':{'BTCUSDT':{'qty':1}}})
    with pytest.raises(ValueError, match='Archive'):
        terminal.activate(tmp_path,'ridge')
    with pytest.raises(ValueError, match='still open'):
        terminal.archive_paper(tmp_path)
    assert json.loads((tmp_path/'paper/state.json').read_text())['model_sha256'] == 'old'


def test_local_training_purges_future_labels_from_each_fold(tmp_path, monkeypatch):
    cfg = replace(Config(), symbols=('BTCUSDT',))
    write_json(tmp_path/'config.json', cfg.to_dict())
    path = tmp_path/'data/processed/bars.parquet'
    path.parent.mkdir(parents=True)
    bars = fixture_bars(1800)
    bars.to_parquet(path,index=False)
    samples = terminal.build_features(bars,cfg)
    checked = []
    class Estimator:
        def fit(self, features, target):
            self.train_end = samples.loc[features.index,'label_end_time'].max()
            return self
        def predict(self, features):
            start = samples.loc[features.index,'decision_time'].min()
            assert self.train_end < start
            checked.append((self.train_end,start))
            return np.zeros(len(features))
    monkeypatch.setattr(terminal,'make_estimator',lambda name,cfg:Estimator())
    monkeypatch.setattr(terminal.joblib,'dump',lambda model,path:Path(path).write_bytes(b'test-model'))
    registry = terminal.fit_local(tmp_path)
    assert len(checked) == 6
    assert set(registry) == {'ridge','xgboost_d3'}
    assert all(not item['validation_is_independent_final_test'] for item in registry.values())
    assert (tmp_path/registry['ridge']['run']/'training_bars.parquet').exists()
