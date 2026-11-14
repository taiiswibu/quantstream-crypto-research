"""Generate the code-only, end-to-end Kaggle workflow notebook."""

import argparse
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(source, write=True):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Pin the actual published source commit")
    cells = [
        """# Run 006. Save Version → Save & Run All trên Kaggle GPU, Internet On.
from pathlib import Path
import sys, subprocess, json, importlib, importlib.util, os, time, shutil
import pandas as pd
import numpy as np
from IPython.display import display, FileLink, clear_output

SOURCE_COMMIT = "SOURCE_PIN"
BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()/'notebook_work'
CODE = BASE/('quantstream-source-'+SOURCE_COMMIT[:12])
WORK = BASE/'QuantStream_Run006'
WORK.mkdir(parents=True,exist_ok=True)
N_TRIALS = 24
MAX_EPOCHS = 12
JOB_SECONDS = 7200
NOTEBOOK_URL = ''
assert 4 <= N_TRIALS <= 24 and 1 <= MAX_EPOCHS <= 12 and JOB_SECONDS <= 21600
if not CODE.exists():
    subprocess.check_call(['git','clone','-q','--no-checkout','https://github.com/taiiswibu/quantstream-crypto-research.git',str(CODE)])
subprocess.check_call(['git','-C',str(CODE),'checkout','-q','--detach',SOURCE_COMMIT])
assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==SOURCE_COMMIT
assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain'],text=True).strip()
if 'quantstream' in sys.modules and Path(sys.modules['quantstream'].__file__).resolve().parent != CODE/'quantstream':
    raise RuntimeError('Restart Session: kernel đã import quantstream từ source khác')
sys.path.insert(0,str(CODE))
modules={'optuna':'optuna>=4.5,<6','pyarrow':'pyarrow>=17,<27','xgboost':'xgboost>=2.1,<4',
         'sklearn':'scikit-learn>=1.5,<2','matplotlib':'matplotlib>=3.9,<4','filelock':'filelock>=3.16,<5'}
missing=[package for name,package in modules.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable,'-m','pip','install','--only-binary=:all:',*missing])
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Chọn GPU trong Kaggle Settings trước khi chạy lượt train đầy đủ')
from quantstream.common import write_json, sha256
from quantstream.workflow import prepare_dataset
from quantstream.features import FEATURES, fold_data
from quantstream.workflow_plots import dataset_figures, scaler_figure, training_figures, evaluation_figures, architecture_figure
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(11,4),'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
write_json(WORK/'notebook-run.json',{'notebook_url':NOTEBOOK_URL or None,'source_commit':SOURCE_COMMIT,
           'torch_loaded':torch.__version__,'numpy_loaded':np.__version__,'gpu':torch.cuda.get_device_name(0),
           'trials':N_TRIALS,'epochs':MAX_EPOCHS,'whole_job_seconds':JOB_SECONDS})
print('Source:',SOURCE_COMMIT,'GPU:',torch.cuda.get_device_name(0))""",
        """# Thu thập 2023-01 → 2026-11-13 UTC; checksum archive và nến đã đóng.
manifest = prepare_dataset(WORK)
bars = pd.read_parquet(WORK/'data/processed/bars.parquet')
quality = json.loads((WORK/'data/processed/data_quality.json').read_text())
display(bars.groupby('symbol').agg(rows=('timestamp','size'),first=('timestamp','min'),last=('timestamp','max')))
display(pd.DataFrame([{k:v for k,v in quality.items() if k!='gaps'}]))
display(pd.DataFrame(quality['gaps']))
print('Source:',manifest['provider'],'Synthetic:',manifest['synthetic'])""",
        """# Làm sạch, missingness và tính hợp lệ OHLCV; không điền giá qua khoảng trống.
from quantstream.data import VALUES
missing = bars[VALUES].isna().sum()
fig,axes = plt.subplots(1,2,figsize=(12,4))
axes[0].bar(missing.index,missing.values,color='#0f766e')
axes[0].set(title='Missing cells after validation',ylabel='Cells');axes[0].tick_params(axis='x',rotation=75)
gap_counts={symbol:sum(r['missing_bars'] for r in quality['gaps'] if r['symbol']==symbol) for symbol in bars.symbol.unique()}
axes[1].bar(gap_counts.keys(),gap_counts.values(),color='#b45309')
axes[1].set(title='Missing hourly candles',ylabel='Candles')
for ax in axes: ax.grid(axis='y',alpha=.2)
plt.tight_layout()
(WORK/'figures').mkdir(exist_ok=True)
fig.savefig(WORK/'figures/01-data-quality.png',dpi=150,bbox_inches='tight');plt.show()
assert np.isfinite(bars[VALUES]).all().all()
assert not bars.duplicated(['symbol','timestamp']).any()
print('Quarantine:',quality['quarantine_rows'],'Duplicate rows:',quality['duplicate_rows'])""",
        """# Feature chỉ dùng quá khứ; nhãn return giữ 4 giờ sau trễ vào 1 giờ.
features = pd.read_parquet(WORK/'data/processed/features.parquet')
split = manifest['split']
display(features[['symbol','decision_time','entry_time','label_end_time','target_return',*FEATURES[:4]]].head(10))
display(pd.DataFrame(split['tuning_folds']))
print('Retrospective test:',split['test_start'],'→',split['end'])
print('Các giai đoạn lịch sử đã được quan sát. Không gọi đây là test đầu tư độc lập.')""",
        """# EDA và ranh giới thời gian; đồ thị mô tả không được dùng fit scaler.
dataset_figures(WORK)""",
        """# Fit scaler trên train của fold đầu; không fit trên validation/test.
train, validation = fold_data(features,split['tuning_folds'][0]['validation_start'],split['tuning_folds'][0]['validation_end'])
assert train.label_end_time.max() < validation.decision_time.min()
scaler_stats = scaler_figure(train, WORK)
display(scaler_stats)
print('Đồ thị demo scaler neural ddof=1. Ridge dùng StandardScaler ddof=0; scaler thực của từng fit nằm trong ZIP. XGBoost dùng feature gốc.')""",
        """# Kiến trúc trước train; không dùng output random init làm kết quả nghiên cứu.
from quantstream.search import SequenceRegressor
architecture=[]
for family in ['tcn','gru']:
    params={'family':family,'width':32,'dropout':.15}
    model=SequenceRegressor(params)
    architecture.append({'family':family,'parameters':sum(p.numel() for p in model.parameters()),
                         'context':48,'inputs':18,'width':32,
                         'activation':'GELU + residual causal conv' if family=='tcn' else 'GRU sigmoid/tanh gates',
                         'output':'Linear; return × 100 during training'})
display(pd.DataFrame(architecture))
print(SequenceRegressor({'family':'tcn','width':32,'dropout':.15}))
architecture_figure(WORK)
del model""",
        """# Tìm siêu tham số và 2 finalist × 3 seed. Watchdog bao trùm cả job.
plan_path=WORK/'training_plan.json'
status=json.loads((WORK/'status.json').read_text()) if (WORK/'status.json').exists() else {}
if plan_path.exists():
    old=json.loads(plan_path.read_text())
    assert old['source']['commit']==SOURCE_COMMIT and old['dataset_manifest_sha256']==sha256(WORK/'dataset_manifest.json'), 'Source hoặc dataset khác lượt đã lưu'
    assert (old['trials'],old['epochs'],old['seconds'])==(N_TRIALS,MAX_EPOCHS,JOB_SECONDS), 'Không đổi ngân sách của lượt đã bắt đầu'
    if status.get('status')!='completed':
        raise RuntimeError('Lượt cũ chưa hoàn tất. Giữ ZIP/evidence để kiểm tra; không xóa để train lại lấy điểm đẹp')
    print('Đọc lại lượt đã hoàn tất, không train lại.')
else:
    log=WORK/'training-console.log'
    environment={**os.environ,'PYTHONPATH':str(CODE)+os.pathsep+os.environ.get('PYTHONPATH','')}
    started=time.monotonic()
    try:
        with log.open('w') as stream:
            worker=subprocess.Popen([sys.executable,'-m','quantstream.workflow','train','--root',str(WORK),
                '--trials',str(N_TRIALS),'--epochs',str(MAX_EPOCHS),'--seconds',str(JOB_SECONDS),'--device','cuda'],
                cwd=str(CODE),env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while worker.poll() is None:
                if time.monotonic()-started > JOB_SECONDS+15:
                    import signal
                    os.killpg(worker.pid,signal.SIGTERM) if os.name=='posix' else worker.terminate()
                    try: worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid,signal.SIGKILL) if os.name=='posix' else worker.kill()
                        worker.wait()
                    write_json(WORK/'watchdog.json',{'stopped':True,'elapsed_seconds':time.monotonic()-started,'returncode':worker.returncode})
                    raise TimeoutError('Watchdog đã dừng job; giữ partial evidence, không chốt model')
                clear_output(wait=True)
                print('Elapsed:',round(time.monotonic()-started),'/',JOB_SECONDS,'seconds')
                print(''.join(log.read_text(errors='replace').splitlines(keepends=True)[-8:]))
                time.sleep(5)
            if worker.returncode:
                raise RuntimeError('Job chưa hoàn tất. Xem training-console.log và error.txt trong ZIP')
    finally:
        if 'worker' in locals() and worker.poll() is None:
            worker.terminate()
            try: worker.wait(timeout=10)
            except subprocess.TimeoutExpired: worker.kill();worker.wait()
        evidence=shutil.make_archive(str(BASE/'QuantStream_Run006_Output'),'zip',WORK)
        display(FileLink(evidence))
status=json.loads((WORK/'status.json').read_text())
display(pd.DataFrame([status]))""",
        """# Toàn bộ trial và loss đã ghi; không chỉ giữ trial có điểm tốt.
trials = pd.read_csv(WORK/'trials.csv')
display(trials.drop(columns=['datetime_start','datetime_complete'],errors='ignore').head(24))
training_figures(WORK)""",
        """# Lựa chọn đã khóa trước khi mở block kiểm tra hồi cứu.
selection=json.loads((WORK/'selection.json').read_text())
display(pd.DataFrame(selection['shortlist']))
print('Selected:',selection['params'],'Fixed deployment seed:',selection['seed'])
print('MSE ratio dưới 1 mới vượt zero-return. Không sửa model dựa trên test sau đây.')
print('Ridge là deterministic; seed SD = 0 không chứng minh ổn định ở thị trường tương lai.')""",
        """# Sai số dự báo và equity là hai câu hỏi khác nhau.
assert status['status']=='completed'
metrics=evaluation_figures(WORK)
display(metrics)
display(pd.read_json(WORK/'retrospective_asset_metrics.json'))
strategies=json.loads((WORK/'retrospective_strategies.json').read_text())
display(pd.DataFrame([{'strategy':k,**v} for k,v in strategies.items()]).drop(columns='note',errors='ignore'))""",
        """# Model, scaler, feature contract và forecast parity để mang về local.
DEPLOYMENT=WORK/'deployment'
bundle=json.loads((DEPLOYMENT/'bundle_manifest.json').read_text())
for name,expected in bundle['files'].items():
    assert sha256(DEPLOYMENT/name)==expected,name
exported=json.loads((DEPLOYMENT/'metadata.json').read_text())
display(pd.DataFrame([{k:exported[k] for k in ['model','training_rows','selected_epochs','created_at','sha256','deployment_status']}]))
print('Model và cấu hình:',exported['hyperparameters'])
print('Import không train lại. Parity kiểm tra cùng đầu vào, scaler và đơn vị return.')""",
        """# Tải cả evidence và gói triển khai. Save Version giữ output notebook thật.
evidence=shutil.make_archive(str(BASE/'QuantStream_Run006_Output'),'zip',WORK)
model_zip=shutil.make_archive(str(BASE/'QuantStream_Run006_Model'),'zip',DEPLOYMENT)
display(FileLink(evidence));display(FileLink(model_zip))
print('Local: giải nén Model.zip, mở trang Dữ liệu → Nạp model đã train → chọn thư mục đã giải nén.')
print('Giữ Save Version của Kaggle cùng ZIP để có bằng chứng notebook đã thực thi.')""",
    ]
    nb = nbf.v4.new_notebook(
        cells=[nbf.v4.new_code_cell(c.replace("SOURCE_PIN", source)) for c in cells]
    )
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python"}
    nbf.validate(nb)
    destination = ROOT / "notebooks/09_Kaggle_EndToEnd_Training.ipynb"
    if write:
        nbf.write(nb, destination)
        print(destination)
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    build(parser.parse_args().source)
