"""Generate a code-only recent-method screening notebook with pinned source."""

import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    cells = []
    cells.append(f"""# Run 009. CPU screening; không phải xác nhận hiệu quả đầu tư.
from pathlib import Path
import os, sys, json, time, subprocess, shutil, importlib.util
from IPython.display import display, FileLink, clear_output
SOURCE_COMMIT = {source!r}
BASE = Path(os.environ.get('QUANTSTREAM_RUN_BASE', '/kaggle/working' if Path('/kaggle/working').exists() else str(Path.cwd()/'recent_work')))
BASE.mkdir(parents=True,exist_ok=True)
CODE = Path(os.environ.get('QUANTSTREAM_SOURCE_DIR',str(BASE/('source-'+SOURCE_COMMIT[:12]))))
if not CODE.exists():
    CODE.mkdir(parents=True)
    subprocess.check_call(['git','init','-q',str(CODE)])
    subprocess.check_call(['git','-C',str(CODE),'fetch','--depth','1','https://github.com/taiiswibu/quantstream-crypto-research.git',SOURCE_COMMIT])
    subprocess.check_call(['git','-C',str(CODE),'checkout','--detach','FETCH_HEAD'])
actual=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
assert actual==SOURCE_COMMIT,'Sai source revision'
loaded=sys.modules.get('quantstream')
if loaded and Path(loaded.__file__).resolve().parent != CODE/'quantstream':
    raise RuntimeError('Restart kernel để nạp đúng source mới')
required=['torch','pandas','numpy','pyarrow','sklearn','xgboost','matplotlib','tirex2','timestral','optuna']
if any(importlib.util.find_spec(name) is None for name in required):
    subprocess.check_call([sys.executable,'-m','pip','install',str(CODE)+'[forecast,research,notebook]'])
sys.path.insert(0,str(CODE))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantstream.common import sha256, git_provenance
from quantstream.modern_models import MODELS
WORK=BASE/'QuantStream_Recent_Methods'
STUDY=WORK/'study'
WORK.mkdir(exist_ok=True)
print('Source:',git_provenance(CODE))
print('Runtime: CPU; foundation point forecast = median; không có optimizer step cho zero-shot.')""")
    cells.append("""# Tái sử dụng snapshot đã khóa; nếu không có Input thì thu thập lại cùng khoảng ngày.
from quantstream.extended_training import prepare_dataset, find_parent_snapshot
parent = Path(os.environ['QUANTSTREAM_PARENT_SNAPSHOT']) if os.environ.get('QUANTSTREAM_PARENT_SNAPSHOT') else find_parent_snapshot(Path('/kaggle/input'),BASE/'snapshot-cache')
SNAPSHOT=WORK/'snapshot'
manifest=prepare_dataset(SNAPSHOT,parent)
print('Input:',parent,'Features SHA:',manifest['files']['data/processed/features.parquet'])
display(pd.DataFrame([{'rows':manifest['rows'],'synthetic':manifest.get('synthetic'),'frozen_end':manifest['frozen_end']}]))""")
    cells.append("""# Model mới được kiểm tra; không dùng điểm benchmark ngoài để chọn thắng.
model_table=[]
for name,spec in MODELS.items():
    model_table.append({'model':name,'revision':spec['revision'],'license':spec['license']})
display(pd.DataFrame(model_table))
print('Mục tiêu giữ nguyên: return 4h, chờ vào 1h. Target context dừng ở t−6; covariate bridge dừng ở t.')
print('Validation đã dùng nghiên cứu; chưa loại trừ pretraining contamination. Source: docs/METHODS_2026_VI.md.')""")
    cells.append("""# Toàn job có trần 30 phút. Lượt completed được đọc lại, không fit lặp để chọn điểm.
environment={**os.environ,'PYTHONPATH':str(CODE)+os.pathsep+os.environ.get('PYTHONPATH','')}
command=[sys.executable,'-m','quantstream.recent_study','--snapshot',str(SNAPSHOT),'--output',str(STUDY),'--seconds','1800']
if os.environ.get('QUANTSTREAM_MODEL_CACHE_ROOT'):
    command+=['--model-cache',os.environ['QUANTSTREAM_MODEL_CACHE_ROOT']]
started=time.monotonic()
log=WORK/'worker.log'
with log.open('w') as stream:
    worker=subprocess.Popen(command,cwd=str(CODE),env=environment,stdout=stream,stderr=subprocess.STDOUT)
    try:
        while worker.poll() is None:
            if time.monotonic()-started>1815:
                worker.kill();worker.wait()
                raise TimeoutError('Giữ partial evidence; worker vượt watchdog')
            clear_output(wait=True)
            print('Elapsed:',round(time.monotonic()-started),'seconds')
            if (STUDY/'progress.json').exists():
                progress=json.loads((STUDY/'progress.json').read_text());print(progress)
                stage_log=STUDY/(progress['stage']+'.log')
                if stage_log.exists():print(*stage_log.read_text(errors='replace').splitlines()[-3:],sep=chr(10))
            time.sleep(2)
        if worker.returncode:
            raise RuntimeError(log.read_text(errors='replace')[-4000:])
    finally:
        if worker.poll() is None:
            worker.kill();worker.wait()
status=json.loads((STUDY/'status.json').read_text())
assert status['status']=='completed'
display(pd.DataFrame([status]))""")
    cells.append("""# Archive checksum, thiếu dữ liệu và thời gian nguồn thực sự khả dụng.
downloads=json.loads((STUDY/'external/downloads.json').read_text())
quality=json.loads((STUDY/'enriched/quality.json').read_text())
display(pd.DataFrame([quality]))
print('Archives verified:',len(downloads['success']),'Failures:',len(downloads['failures']))
assert not downloads['failures']
print('Kho exogenous chỉ có các tháng hoàn tất đến 10/2026; không điền phần thiếu bằng 0.')""")
    cells.append("""# Cohort hourly: cùng dòng và cùng thời gian nhãn cho các điều kiện.
hourly=json.loads((STUDY/'enriched/results.json').read_text())
display(pd.DataFrame(hourly)[['model','rows','mean_fold_mse_ratio','fold_ratios']])
print('RLS chỉ cập nhật bằng nhãn đã xuất hiện. Mốc zero-return có ratio 1.')""")
    cells.append("""# Cohort foundation: dự báo tại 00:00 UTC mỗi ngày; không trộn với bảng hourly.
foundation=json.loads((STUDY/'foundation/results.json').read_text())
timestral=json.loads((STUDY/'timestral/results.json').read_text())
display(pd.DataFrame(foundation+timestral)[['model','rows','mean_fold_mse_ratio','fold_ratios']])
display(pd.DataFrame(json.loads((STUDY/'intervals/summary.json').read_text())))
print('Coverage, interval score và MSE trả lời các câu hỏi khác nhau; không đổi mục tiêu để gọi là đạt.')""")
    cells.append("""# Biểu đồ đọc từ prediction/log đã lưu, không phải đường minh họa.
from quantstream.modern_plots import figures
figures(STUDY)""")
    cells.append("""# Kiểm tra inference thực với input snapshot; dữ liệu cũ không được ghi thành forecast mới.
from quantstream.modern_models import predict_latest_distribution
from quantstream.risk_reference import empirical_interval
frame=pd.read_parquet(SNAPSHOT/'data/processed/features.parquet')
reference=[empirical_interval(frame,s) for s in ['BTCUSDT','ETHUSDT']]
display(pd.DataFrame(reference)[['symbol','decision_time','q10','q50','q90','history_rows']])
latest=[]
for name in MODELS:
    latest.extend(predict_latest_distribution(frame,STUDY/'models'/name,name))
display(pd.DataFrame(latest)[['source','symbol','decision_time','q10','q50','q90']])
(STUDY/'snapshot_inference.json').write_text(json.dumps(latest,indent=2))
print('Các giá trị trên thuộc snapshot đã lưu; không phải evidence dự báo trực tiếp hôm nay.')""")
    cells.append("""# Gói nhẹ; weights và raw archives được giữ riêng.
from zipfile import ZipFile, ZIP_DEFLATED
diagnostics=BASE/'QuantStream_Recent_Methods_Diagnostics.zip'
with ZipFile(diagnostics,'w',ZIP_DEFLATED) as z:
    for path in STUDY.rglob('*'):
        rel=path.relative_to(STUDY)
        if not path.is_file() or rel.parts[0] in ['models','figures','external'] or path.name=='enriched_features.parquet':continue
        if path.suffix in ['.json','.csv','.parquet','.log','.txt']:z.write(path,rel.as_posix())
    for name in ['plan.json','downloads.json']:
        z.write(STUDY/'external'/name,'external/'+name)
display(FileLink(str(diagnostics)))
print('App core dùng model đã nhập. Xem docs/VAN_HANH_VI.md để chuẩn bị model thử nghiệm riêng.')""")
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(c) for c in cells])
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    for cell in notebook.cells:
        compile(cell.source, "<notebook-cell>", "exec")
    nbf.validate(notebook)
    destination = ROOT / "notebooks/12_Recent_Methods_and_Risk.ipynb"
    nbf.write(notebook, destination)
    print(destination)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    build(args.source)
