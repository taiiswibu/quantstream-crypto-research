from pathlib import Path
import nbformat as nbf
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parents[1]
cells = [
'''# Kiểm định baseline local; chỉ đọc artifacts, không fit lại.
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from quantstream import terminal

ROOT = Path.cwd()
if not (ROOT/'workspace').exists():
    ROOT = ROOT.parent
WORK = ROOT/'workspace'
meta = terminal.read_json(WORK/'artifacts/champion.json')
run = WORK/meta['run']
metrics = pd.read_json(run/'validation_metrics.json')
bars = pd.read_parquet(run/'training_bars.parquet')
plt.rcParams.update({'figure.figsize': (11, 4), 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'font.size': 10})
COLORS = ['#0f766e','#2563eb','#64748b','#b45309']
display(bars.groupby('symbol').agg(candles=('timestamp','size'), first=('timestamp','min'), last=('timestamp','max')))
print('Model:', meta['model'], '· fit:', meta['created_at'])
print('Nguồn: Binance Spot; validation đã dùng so model, chưa phải final test độc lập.')''',
'''# Chuỗi giá train thực tế; mỗi tài sản chuẩn hóa về 100.
fig, ax = plt.subplots()
for (symbol, group), color in zip(bars.groupby('symbol'), COLORS):
    group = group.sort_values('timestamp')
    ax.plot(group.timestamp, group.close/group.close.iloc[0]*100, label=symbol, color=color)
ax.set(title='Dữ liệu model local: nến 1 giờ', xlabel='UTC', ylabel='Giá chuẩn hóa (đầu kỳ = 100)')
ax.legend(); ax.grid(alpha=.2); plt.tight_layout(); plt.show()''',
'''# MSE dưới 1 mới vượt zero-return; giữ cả các cấu hình kém baseline.
display(metrics.round(6))
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].bar(metrics.model, metrics.mae*100, color=COLORS)
axes[0].set(title='Sai số validation', ylabel='MAE (điểm % return)')
axes[1].bar(metrics.model, metrics.mse_ratio_to_zero, color=COLORS)
axes[1].axhline(1,color='#b91c1c',linestyle='--',label='Zero-return')
axes[1].set(title='So baseline',ylabel='MSE / zero-return MSE'); axes[1].legend()
for ax in axes:
    ax.tick_params(axis='x',rotation=15); ax.grid(axis='y',alpha=.2)
plt.tight_layout(); plt.show()''',
'''# Dự báo ngoài train của từng fold; các nhãn chồng lấn đã purge ở biên.
oof = pd.read_parquet(run/'validation_predictions.parquet')
fig, axes = plt.subplots(1,2,figsize=(12,4))
for (name, group), color in zip(oof.groupby('model'), COLORS):
    axes[0].scatter(group.target_return*100, group.prediction*100, s=7, alpha=.2, color=color, label=name)
    axes[1].hist((group.prediction-group.target_return)*100,bins=45,alpha=.5,color=color,label=name)
axes[0].set(title='Thực tế và dự báo',xlabel='Return thực tế (%)',ylabel='Return dự báo (%)')
axes[1].set(title='Phân phối sai số',xlabel='Dự báo − thực tế (điểm %)',ylabel='Số cửa sổ')
for ax in axes:
    ax.legend(); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
display(oof.groupby(['model','fold']).agg(rows=('prediction','size'),start=('decision_time','min'),end=('label_end_time','max')))''',
'''# Cùng phí/slippage; equity này thuộc validation, không phải tiền thật.
curve = pd.read_parquet(run/(meta['model']+'_equity.parquet'))
fig, axes = plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
for name,color in zip(['model','buy_hold','ema'],COLORS):
    axes[0].plot(curve.timestamp,curve[name],label=name,color=color)
axes[0].set(title='Mô phỏng validation sau chi phí',ylabel='Equity (đầu kỳ = 1)'); axes[0].legend()
drawdown=curve.model/curve.model.cummax().clip(lower=1)-1
axes[1].fill_between(curve.timestamp,drawdown*100,0,color='#b91c1c',alpha=.3)
axes[1].set(xlabel='UTC',ylabel='Drawdown (%)')
for ax in axes: ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
report=terminal.read_json(run/(meta['model']+'_backtest.json'))
display(pd.DataFrame([{'strategy':name,**v} for name,v in report.items()]).drop(columns='note',errors='ignore'))''',
'''# Ridge có hệ số tuyến tính; XGBoost dùng cây, không có hàm kích hoạt neural.
import joblib
registry=terminal.read_json(run/'registry.json')
for name,item in registry.items():
    model=joblib.load(WORK/item['path'])['model']
    if name=='ridge':
        display(pd.DataFrame([{'model':name,'alpha':model.named_steps['ridge'].alpha,'scaler':'StandardScaler fit trên train','threshold':item['threshold']}]))
    else:
        keys=['n_estimators','max_depth','learning_rate','min_child_weight','reg_lambda','random_state']
        display(pd.DataFrame([{'model':name,**{k:model.get_params()[k] for k in keys}}]))
print('TCN/LoRA và loss theo step nằm ở notebook nghiên cứu tương ứng; không tạo loss curve giả cho Ridge.')
print('Đọc bảng MSE / zero-return của lần fit này; validation chưa phải test đầu tư độc lập.')'''
]
nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(c) for c in cells])
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata.language_info={'name':'python','version':'3.12'}
nbf.validate(nb)
# The runtime blocks kernel sockets; execute the same cells in one IPython shell.
shell = TerminalInteractiveShell.instance()
shell.run_line_magic('matplotlib', 'inline')
shell.display_formatter.active_types = ['text/plain', 'image/png', 'text/html']
for i, cell in enumerate(nb.cells, 1):
    with capture_output() as captured:
        result = shell.run_cell(cell.source, store_history=False)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Cell {i} failed') from (result.error_before_exec or result.error_in_exec)
    cell.execution_count = i
    cell.outputs = []
    if captured.stdout:
        cell.outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbf.v4.new_output('display_data', data=output.data, metadata=output.metadata))
nb.metadata['quantstream_validation'] = {'execution': 'top_to_bottom_in_process_IPython', 'kernel_socket_launch': 'blocked_by_runtime', 'training_rerun': False}
nbf.write(nb,root/'notebooks/08_Local_Model_Review.ipynb')
print('Executed',len(nb.cells),'cells;',sum(o.output_type=='display_data' and 'image/png' in o.data for c in nb.cells for o in c.get('outputs',[])),'figures')
