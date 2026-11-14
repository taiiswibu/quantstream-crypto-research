"""Keep exported notebooks compact without changing experiment cells."""
from pathlib import Path
import nbformat

NOTES = {
    '01': ['Kết quả đã chạy; chỉ đọc artifacts.', 'Feature từ nến đã đóng; nhãn khớp sau một giờ.', 'So sánh trên development, chưa phải final test.', 'Final holdout đã mở; giữ nguyên kết quả sau phí.', 'LoRA exploratory; chưa xác minh pretraining overlap.'],
    '02': ['Kaggle: Internet On, thêm dataset QuantStream và chọn GPU.', 'Nạp snapshot development, không gồm final holdout.', 'Kiểm tra CUDA trước khi train.', 'Giữ nguyên 6 epoch của thí nghiệm.', 'Lưu output và metadata.'],
    '03': ['Kaggle: Internet On, thêm dataset QuantStream và chọn GPU.', 'Nạp snapshot development.', 'Kiểm tra GPU; không đổi checkpoint giữa các cấu hình.', 'LoRA 200 steps, 512 cửa sổ validation.', 'Xuất adapter; MAE không phải lợi nhuận sau phí.'],
    '04': ['Thí nghiệm Run 002; không chạy lại để chọn kết quả đẹp hơn.', 'Kiểm tra snapshot và checksum.', 'Chỉ dùng development.', 'Giữ ngân sách trial và epoch đã khóa.', 'Objective dưới 1 mới vượt baseline zero-return.'],
    '05': ['Thí nghiệm Run 003; Internet On, GPU, cùng dataset development.', 'Đối chiếu source và snapshot.', 'Kiểm tra CUDA và môi trường.', 'Mỗi cấu hình bắt đầu từ cùng base revision.', 'Giữ cả cấu hình kém hơn baseline.'],
    '06': ['Run 004 đã hoàn thành; không train lại để thay kết quả.', 'Mặc định preflight; chỉ chuyển train sau khi kiểm tra evidence.', 'Giữ source, snapshot và candidate lock đúng SHA.', 'Preflight smoke không tính là research fit; lưu cả evidence khi lỗi.'],
    '07': ['Run 005: chỉ inference, không cập nhật trọng số; đọc controls và reference riêng.', 'Giữ runtime đã đối chiếu probe; lỗi thì dừng, không bỏ kiểm tra.', 'Biểu đồ Run 004 đọc log cũ đã xác minh, không phải lần train mới.', 'Chạy đủ đối chứng; không chọn riêng lượt có điểm tốt.', 'Completed khác controls_passed và reference_reproduced.'],
    '07b': ['Probe runtime chỉ đọc; dùng CPU, không cần dataset.', 'Lưu ZIP chẩn đoán, không sửa package trong cell này.'],
}


CHARTS = {
'02': '''# Loss theo fold; đường final train không có validation mới.
import pandas as pd
import matplotlib.pyplot as plt
loss_rows = [{'fold': f['fold'] + 1, **e} for f in metadata['history'] for e in f['epochs']]
loss = pd.DataFrame(loss_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for fold, group in loss.groupby('fold'):
    axes[0].plot(group.epoch, group.train_huber, label=f'Fold {fold}')
    axes[1].plot(group.epoch, group.validation_mse, label=f'Fold {fold}')
for ax, title in zip(axes, ['Train Huber (scaled target)', 'Validation MSE (return)']):
    ax.set(title=title, xlabel='Epoch', ylabel='Loss'); ax.legend(); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()''',
'03': '''# So lỗi trước/sau LoRA trên cùng các cửa sổ development.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(['Zero-shot', 'LoRA 200 steps'], [report['zero_shot_mae'], report['finetuned_mae']], color=['#64748b', '#0f766e'])
ax.set(title='Chronos: development validation', ylabel='MAE (return)', xlabel='Model')
ax.grid(axis='y', alpha=.2); plt.tight_layout(); plt.show()''',
'04': '''# Giữ tất cả trial, kể cả trial không vượt baseline.
import matplotlib.pyplot as plt
valid_trials = trials.dropna(subset=['value']).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(valid_trials)), valid_trials.value, color='#0f766e')
ax.axhline(1, color='#b45309', linestyle='--', label='Zero-return baseline')
ax.set(title='Model search: all completed objectives', xlabel='Trial (CSV order)', ylabel='MSE / zero-return MSE')
ax.legend(); ax.grid(axis='y', alpha=.2); plt.tight_layout(); plt.show()''',
'05': '''# So từng cấu hình với zero-shot tương ứng, không chỉ chọn dòng tốt nhất.
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(len(table))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x-.18, table.zero_shot_mae, .36, label='Zero-shot', color='#64748b')
ax.bar(x+.18, table.finetuned_mae, .36, label='LoRA', color='#0f766e')
ax.set(xticks=x, xticklabels=table.experiment, ylabel='MAE (return)', title='LoRA: development validation')
ax.legend(); ax.grid(axis='y', alpha=.2); plt.tight_layout(); plt.show()''',
'06': '''# Preflight chưa có kết quả research; chỉ vẽ khi train hoàn tất.
if MODE == 'train' and 'metrics' in result:
    import pandas as pd
    import matplotlib.pyplot as plt
    seed_metrics = pd.DataFrame(result['metrics'])
    display(seed_metrics)
    metric_columns = [c for c in seed_metrics.columns if 'mae' in c.lower() and pd.api.types.is_numeric_dtype(seed_metrics[c])]
    if metric_columns:
        ax = seed_metrics[metric_columns].plot.bar(figsize=(10, 4), color=['#64748b', '#0f766e', '#2563eb'])
        ax.set(title='Run 004: all reported seed MAE', xlabel='Metric row', ylabel='MAE (return)')
        ax.grid(axis='y', alpha=.2); plt.tight_layout(); plt.show()
else:
    print('Preflight: chưa có metric research để vẽ.')'''
}

def compact(nb, name):
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    if any(cell.cell_type == 'markdown' for cell in nb.cells):
        key = '07b' if name.startswith('07b') else name[:2]
        for cell, note in zip(code, NOTES.get(key, [])):
            cell.source = '# ' + note + '\n' + cell.source
    nb.cells = code
    if name[:2] in CHARTS and not any(c.metadata.get('quantstream_plot') for c in code):
        cell = nbformat.v4.new_code_cell(CHARTS[name[:2]])
        cell.metadata['quantstream_plot'] = True
        nb.cells.append(cell)
    return nb


def write_notebook(nb, path):
    path = Path(path)
    compact(nb, path.name)
    nbformat.validate(nb)
    nbformat.write(nb, path)
