"""Generate the reproducible results notebook and two Kaggle training notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks"
FOLDER.mkdir(exist_ok=True)


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    nbf.write(nb, FOLDER / name)


save("01_Research_Results.ipynb", [
md("""# QuantStream — kết quả nghiên cứu đã chạy

Notebook đọc artifacts của các lượt train thật, không tự fit lại model. Dataset là BTC/USDT và ETH/USDT, nến 1 giờ. TCN được chọn bằng ba fold validation trước khi mở final holdout. Kết quả test âm sau phí được giữ nguyên.

**Mục tiêu học:** hiểu feature/label, temporal split, training, backtesting và vì sao accuracy hơi trên 50% chưa chứng minh chiến lược có lợi nhuận."""),
code("""from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Notebook có thể mở từ thư mục gốc hoặc thư mục notebooks.
ROOT = Path.cwd()
if not (ROOT / 'config.json').exists():
    ROOT = ROOT.parent
cfg = json.loads((ROOT / 'config.json').read_text())
run = ROOT / json.loads((ROOT / 'artifacts/latest_run.json').read_text())['run']
quality = json.loads((ROOT / 'data/processed/data_quality.json').read_text())
split = json.loads((ROOT / 'data/processed/split_plan.json').read_text())
report = json.loads((run / 'final_report.json').read_text())
bars = pd.read_parquet(ROOT / 'data/processed/bars.parquet')
# Paper runtime có thể thêm dữ liệu mới; chỉ đọc khoảng lịch sử của thí nghiệm này.
historical_end = pd.Timestamp(split['end'])
bars = bars[bars.timestamp < historical_end]
features = pd.read_parquet(ROOT / 'data/processed/features.parquet')
display(bars.groupby('symbol').agg(candles=('close','size'), first=('timestamp','min'), last=('timestamp','max')))
display(pd.DataFrame([{k:v for k,v in quality.items() if k != 'gaps'}]))
print('Khoảng trống dữ liệu:', quality['gaps'])"""),
md("""## Feature và nhãn

Tất cả feature lấy từ nến đã đóng. Chờ thêm một giờ để xử lý rồi mới vào ở giá mở nến kế tiếp. Giữ vị thế 4 giờ. Nhãn là `exit_close / entry_open - 1`; phí và slippage được tính trong simulator.

Không dùng scaler fit trên toàn dữ liệu. Nến thiếu làm indicators/sequence khởi động lại."""),
code("""display(features[['symbol','decision_time','entry_time','label_end_time','ret_1','rsi_14','target_return']].head(8))
display(pd.DataFrame(split['folds']))
print('Final holdout bắt đầu:', split['test_start'])
print('Rule:', split['rule'])"""),
md("""## So sánh model trên development

Ridge, Logistic và hai cấu hình XGBoost được đánh giá cùng 3 fold. TCN tự viết bằng PyTorch, cũng dùng 3 fold. Epoch được chọn bằng validation MSE; sau đó chọn threshold chiến lược trên development. Số giao dịch tối thiểu là 20 gộp qua các fold. Đây là kết quả dùng để chọn model, không phải kết quả final test."""),
code("""comparison = pd.DataFrame(json.loads((run / 'comparison.json').read_text())).drop(columns=['last_fold','selection_score'])
tcn = json.loads((ROOT / 'artifacts/tcn_training/tcn_metadata.json').read_text())
tcn_row = {'model':'tcn', 'threshold':tcn['threshold']}
tcn_row.update({k:v for k,v in tcn['selection'].items() if k in comparison.columns})
comparison = pd.concat([comparison,pd.DataFrame([tcn_row])],ignore_index=True)
display(comparison.sort_values('mean_validation_sharpe',ascending=False))
history = []
for fold in tcn['history']:
    for epoch in fold['epochs']:
        history.append({'fold':fold['fold'] + 1, **epoch})
history = pd.DataFrame(history)
fig, ax = plt.subplots(figsize=(9,3))
for fold, group in history.groupby('fold'):
    ax.plot(group.epoch,group.validation_mse,marker='o',label=f'Fold {fold}')
ax.set(xlabel='Epoch',ylabel='Validation MSE',title='TCN: validation loss by fold')
ax.legend(); ax.grid(alpha=.2); plt.tight_layout(); plt.show()"""),
md("""## Final holdout — chỉ mở sau khi khóa lựa chọn

Các con số dưới đây đã tính phí 10 bps và slippage 5 bps **mỗi chiều**, chờ khớp 1 giờ, giữ 4 giờ. Halt theo drawdown 20% của mỗi phần vốn tài sản, kiểm tra ở giá đóng nến. Buy-and-hold là benchmark giữ liên tục; EMA dùng cùng cơ chế giữ 4 giờ và risk rules với model."""),
code("""rows=[]
for name,key in [('TCN','strategy'),('Buy and hold','buy_hold'),('EMA','ema')]:
    rows.append({'strategy':name,**report[key]})
display(pd.DataFrame(rows).drop(columns='note'))
display(pd.DataFrame([report['prediction']]))
curve=pd.read_parquet(run/'test_equity.parquet')
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
for column in ['model','buy_hold','ema']:
    axes[0].plot(curve.timestamp,curve[column],label=column)
axes[0].set(ylabel='Equity (initial = 1)',title='Final holdout: equity after transaction costs')
axes[0].legend(); axes[0].grid(alpha=.2)
dd=curve.model/curve.model.cummax().clip(lower=1)-1
axes[1].fill_between(curve.timestamp,dd,0,color='tomato',alpha=.4)
axes[1].set(ylabel='Drawdown',xlabel='UTC'); axes[1].grid(alpha=.2)
plt.tight_layout(); plt.show()
display(pd.read_parquet(run/'test_trades.parquet').head(10))"""),
md("""## Fine-tuning pretrained bằng LoRA

Lượt kiểm chứng đã fine-tune Chronos-2 trên log-price quá khứ trong train, rồi so sánh trước/sau trên cùng 512 cửa sổ validation lấy mẫu cố định. Không đưa future OHLCV vào model. Phần này exploratory vì chưa xác minh trùng lặp dữ liệu pretraining; không dùng để chọn champion hoặc mở final test."""),
code("""chronos=json.loads((ROOT/'artifacts/chronos_200/chronos_report.json').read_text())
display(pd.DataFrame([
    {'mode':'zero_shot','mae':chronos['zero_shot_mae'],'directional_accuracy':chronos['zero_shot_directional_accuracy']},
    {'mode':'LoRA_200_steps','mae':chronos['finetuned_mae'],'directional_accuracy':chronos['finetuned_directional_accuracy']}
]))
print('Base revision:',chronos['resolved_revision'])
print('Device used:',chronos['device'])
print('Pretraining cutoff verified:',chronos['pretraining_data_cutoff_verified'])"""),
md("""## Kết luận

Pipeline train và fine-tuning đã chạy được, nhưng chiến lược TCN hiện vẫn lỗ trên final holdout; LoRA cũng chưa cải thiện MAE ở lượt này. Điều này không được đổi thành tuyên bố model thành công về giao dịch.

Các hướng nghiên cứu tiếp: kiểm tra hypothesis theo regime, calibration ở nhóm tín hiệu lớn, mức độ nhạy với chi phí và uncertainty qua các block thời gian. Hypothesis/model mới phải được đánh giá trên giai đoạn tương lai mới; final test đang có đã được xem.

Để trình diễn: giải thích data quality, nhãn đúng với thời điểm khớp, purge tại ranh giới, kết quả validation khác test và cách monitoring phát hiện thay đổi. Notebook 02 và 03 cho phép bạn chạy lại phần training trên tài khoản Kaggle của mình.""")
])

BOOTSTRAP = """from pathlib import Path
import sys, zipfile, shutil, json

WORK = Path('/kaggle/working/quantstream') if Path('/kaggle/working').exists() else Path.cwd()/'notebook_work'
WORK.mkdir(parents=True,exist_ok=True)
INPUT = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd().parent
archives = list(INPUT.rglob('QuantStream_Kaggle_Input.zip'))
if archives:
    with zipfile.ZipFile(archives[0]) as z:
        for name in z.namelist():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Invalid archive path')
        z.extractall(WORK)
else:
    # Kaggle may automatically extract a ZIP uploaded as a Dataset.
    manifests=[p for p in INPUT.rglob('manifest.json') if (p.parent/'development.parquet').exists()]
    if len(manifests)!=1:
        raise FileNotFoundError('Attach the Dataset containing QuantStream_Kaggle_Input.zip (or its extracted contents).')
    source=manifests[0].parent
    shutil.copytree(source,WORK/'snapshot',dirs_exist_ok=True)
    shutil.copytree(source.parent/'code',WORK/'code',dirs_exist_ok=True)
SNAPSHOT=WORK/'snapshot'
sys.path.insert(0,str(WORK/'code'))
manifest=json.loads((SNAPSHOT/'manifest.json').read_text())
assert manifest['contains_final_holdout'] is False
print('Snapshot:',SNAPSHOT)
print('Final holdout included:',manifest['contains_final_holdout'])
"""

save("02_Kaggle_TCN.ipynb", [
md("""# QuantStream — tự train TCN bằng PyTorch trên Kaggle

Upload `QuantStream_Kaggle_Input.zip` thành Dataset riêng tư, Add Input vào notebook. Bật Internet và chọn GPU nếu tài khoản còn hạn mức. Mã và snapshot được lấy từ ZIP; final holdout không được xuất sang Kaggle.

Notebook train ba fold walk-forward, chọn epoch/threshold bằng development, refit model cuối trên development rồi xuất weights và metadata. Đây là train từ đầu, không phải fine-tune pretrained."""),
code("""import subprocess,sys
# Chỉ cài dependencies cần cho training; không cài dashboard trong notebook.
subprocess.check_call([sys.executable,'-m','pip','install','-q','pyarrow>=17,<27','scikit-learn>=1.5,<2'])"""),
code(BOOTSTRAP),
code("""import torch
print('PyTorch:',torch.__version__)
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))"""),
md("""## Train

TCN dùng convolution causal có dilation; không nhìn các bước phía sau. Scaler của mỗi fold chỉ fit train. Huber loss trên lợi suất được nhân 100 để ổn định thang số. Chọn epoch bằng validation MSE. Strategy threshold được so sánh trên ba fold sau phí.

`epochs=8` là cấu hình dự kiến cho lần chạy Kaggle. Lượt kiểm chứng kèm gói đã dùng 6 epoch/fold và lưu kết quả riêng."""),
code("""from quantstream.deep import train_snapshot
OUTPUT=WORK/'tcn_output'
metadata=train_snapshot(SNAPSHOT,OUTPUT,epochs=8,device='auto')
print(json.dumps(metadata['selection'],indent=2))"""),
md("""## Download output

Download ZIP trong Output, giải nén về local rồi xem báo cáo validation trước khi chọn model. `python -m quantstream register-tcn <folder>` kiểm tra hash và đăng ký model. Holdout đã mở từ trước vẫn bị khóa với model mới; dùng giai đoạn tương lai mới hoặc forward paper trading."""),
code("""archive=shutil.make_archive(str(WORK.parent/'TCN_Kaggle_Output'),'zip',OUTPUT)
print('Download:',archive)
from IPython.display import FileLink,display
display(FileLink(archive))""")
])

save("03_Kaggle_Chronos_LoRA.ipynb", [
md("""# QuantStream — fine-tune Chronos-2 bằng LoRA trên Kaggle

Add Input chứa `QuantStream_Kaggle_Input.zip`, bật Internet và GPU khả dụng. Notebook tải checkpoint công khai `amazon/chronos-2`, pin revision thực tế, fine-tune LoRA bằng log-price quá khứ và so sánh trước/sau trên cùng validation windows.

**Giới hạn:** chưa xác minh dữ liệu pretraining có trùng giai đoạn tài chính đang đánh giá. Đây là thí nghiệm học fine-tuning, không phải bằng chứng alpha độc lập hoặc kết quả competition."""),
code("""import subprocess,sys,os
os.environ['HF_HUB_DISABLE_XET']='1'
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'chronos-forecasting==2.3.2','peft>=0.18.1,<1','accelerate>=1.1,<2',
    'datasets>=3,<5','pyarrow>=17,<27','scikit-learn>=1.5,<2'])"""),
code(BOOTSTRAP),
code("""import torch
print('CUDA:',torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
# Nếu GPU chưa có, hãy chọn accelerator trong Settings trước khi train.
# Hàm hỗ trợ CPU cho smoke run; GPU thuận tiện hơn cho lần train đầy đủ."""),
md("""## Fine-tuning thật

LoRA cập nhật các adapter trọng số; khác với thử learning rate hoặc threshold. PEFT phải cài thành công; mã không chấp nhận âm thầm chuyển sang full fine-tuning khi thiếu PEFT. Dữ liệu train dừng trước validation. Future covariates không chứa giá tương lai.

200 steps, context 128 và batch size 8 là cấu hình nhỏ để chạy thử. Không đổi cấu hình vì xem kết quả final holdout."""),
code("""from quantstream.finetune import chronos_experiment
OUTPUT=WORK/'chronos_output'
report=chronos_experiment(SNAPSHOT,OUTPUT,steps=200,max_windows=512,device='auto')
print(json.dumps(report,indent=2))"""),
md("""## Đọc kết quả và lưu adapter

MAE thấp hơn mới là cải thiện sai số dự báo; directional accuracy tăng không tự động có nghĩa chiến lược có lãi sau phí. Adapter không tự ghi đè champion local. Output kèm base revision, snapshot hash, số cửa sổ và các hạn chế để bạn viết báo cáo đúng."""),
code("""import pandas as pd
from IPython.display import display,FileLink
display(pd.DataFrame([
 {'mode':'zero_shot','mae':report['zero_shot_mae'],'directional_accuracy':report['zero_shot_directional_accuracy']},
 {'mode':'LoRA','mae':report['finetuned_mae'],'directional_accuracy':report['finetuned_directional_accuracy']}
]))
# Chỉ xuất adapter và evidence, không cần ZIP optimizer checkpoints.
EXPORT=WORK/'chronos_export'
EXPORT.mkdir(exist_ok=True)
shutil.copytree(OUTPUT/'adapter',EXPORT/'adapter',dirs_exist_ok=True)
for name in ['chronos_report.json','chronos_validation.parquet','training_snapshot_manifest.json']:
 shutil.copy2(OUTPUT/name,EXPORT/name)
archive=shutil.make_archive(str(WORK.parent/'Chronos_Kaggle_Output'),'zip',EXPORT)
display(FileLink(archive))""")
])

print('Built three notebooks in', FOLDER)
