"""Build separate research notebooks without replacing executed results or run 001."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


BOOTSTRAP = """from pathlib import Path
import sys, zipfile, shutil, json, hashlib
WORK=Path('/kaggle/working/quantstream_research') if Path('/kaggle/working').exists() else Path.cwd()/'notebook_work'
WORK.mkdir(parents=True,exist_ok=True)
INPUT=Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd().parent
archives=list(INPUT.rglob('QuantStream_Kaggle_Input.zip'))
if len(archives)>1:
    raise ValueError('Chỉ Add Input một phiên bản QuantStream dataset; bỏ các phiên bản cũ.')
if archives:
    with zipfile.ZipFile(archives[0]) as z:
        for name in z.namelist():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts: raise ValueError('Invalid ZIP path')
        z.extractall(WORK)
else:
    manifests=[p for p in INPUT.rglob('manifest.json') if (p.parent/'development.parquet').exists()]
    if len(manifests)!=1: raise FileNotFoundError('Add Input: QuantStream_Kaggle_Input.zip bản mới nhất.')
    source=manifests[0].parent
    shutil.copytree(source,WORK/'snapshot',dirs_exist_ok=True)
    shutil.copytree(source.parent/'code',WORK/'code',dirs_exist_ok=True)
SNAPSHOT=WORK/'snapshot'
manifest=json.loads((SNAPSHOT/'manifest.json').read_text())
if manifest.get('contains_final_holdout') is not False: raise ValueError('Development-only input required')
if not manifest.get('code_files') or not (WORK/'code/quantstream/search.py').exists():
    raise ValueError('Input cũ: tải lại ZIP mới dành cho notebook 04/05.')
for name,expected in manifest['code_files'].items():
    if hashlib.sha256((WORK/'code'/name).read_bytes()).hexdigest()!=expected:
        raise ValueError('Source checksum mismatch: '+name)
sys.path.insert(0,str(WORK/'code'))
print('Source:',manifest['source_control'])
print('Development only. Final holdout included:',manifest['contains_final_holdout'])
"""


def save(name, cells):
    path = FOLDER / name
    if path.exists():
        previous = nbf.read(path, as_version=4)
        for old, new in zip(previous.cells, cells):
            if old.cell_type == new.cell_type:
                new.id = old.id
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    nbf.write(nb, path)


save("04_Kaggle_Model_Search.ipynb", [
md("""# QuantStream Research Lab — model comparison + hyperparameter tuning

**Chạy notebook này trước.** Tạo Dataset version mới từ `QuantStream_Kaggle_Input.zip` mới nhất, chỉ Add Input version đó. Internet On; GPU khả dụng cho TCN/GRU. Không cài lại PyTorch của Kaggle.

Mục tiêu: học cách đặt giả thuyết, so baseline, dùng Optuna và kiểm tra độ ổn định. Chưa có kết quả chạy thật cho notebook này. Không tự chọn champion giao dịch.

- Ridge / XGBoost: baseline tabular; TCN / GRU: mạng chuỗi thời gian.
- 12 trial **tổng cộng**, tối đa 8 epoch cho mỗi lượt train neural. Bốn trial đầu bao phủ đủ bốn họ model; không phải 12 model khác nhau.
- Hai fold sớm dùng xếp hạng. Mỗi neural fold có inner train/early-stop riêng, sau đó refit outer train với epoch cố định.
- Top 2 cấu hình kiểm tra seed 42, 123, 2026 trên hai fold sớm. Ridge là deterministic nên đổi seed không tạo thêm bằng chứng độc lập.
- Khóa lựa chọn trước khi xem fold 3 của candidate. Fold 3 đã từng được xem trong dự án; chỉ là diagnostic development, **không phải holdout độc lập mới**.
- Final holdout cũ đã mở, không xuất trong input và không được gọi ở đây."""),
code("""import subprocess,sys,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'pyarrow>=17,<27','scikit-learn>=1.5,<2','xgboost-cpu>=2.1,<4','optuna>=4.5,<6'])"""),
code(BOOTSTRAP),
code("""import torch,pandas as pd
from IPython.display import display,FileLink
from quantstream.search import run_search,load_snapshot
m,dev,bars,cfg=load_snapshot(SNAPSHOT)
print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
print('Rows:',len(dev),'Test excluded from:',m['split']['test_start'])
display(pd.DataFrame(m['split']['folds']))"""),
md("""## Đóng băng ngân sách và chạy

Objective là trung bình **MSE / MSE của dự báo lợi suất bằng 0** trên hai fold. Nhỏ hơn 1 nghĩa là tốt hơn baseline 0 theo sai số, chưa có nghĩa có lãi. Threshold 0.006 cố định; phí 10 bps và slippage 5 bps mỗi chiều. Không chọn tham số bằng PnL fold 3.

`SEARCH_TIMEOUT=7200` chỉ giới hạn thời gian **bắt đầu trial tìm kiếm**. Trial đang chạy, kiểm tra nhiều seed và diagnostic có thể tiếp tục sau đó; đây không phải cam kết xong trong hai giờ. Nếu thiếu quota, chạy 4 trial / 4 epoch ngay từ đầu trong thí nghiệm riêng và ghi rõ thay đổi.

Giữ nguyên thư mục output để resume. SQLite và CSV giữ cả trial tốt, xấu và bị dừng. Sau khi diagnostic bắt đầu, chương trình từ chối âm thầm chạy lại khi lỗi. Save Version / tải output để tránh mất lịch sử khi Kaggle kết thúc phiên."""),
code("""N_TRIALS=12
MAX_EPOCHS=8
SEARCH_TIMEOUT=7200
OUTPUT=WORK/'run-002-model-search'
try:
    result=run_search(SNAPSHOT,OUTPUT,n_trials=N_TRIALS,max_epochs=MAX_EPOCHS,
                      timeout_seconds=SEARCH_TIMEOUT,device='auto')
finally:
    # Preserve partial evidence even if a fit fails or the budget expires.
    if OUTPUT.exists():
        archive=shutil.make_archive(str(WORK.parent/'QuantStream_Research_Output'),'zip',OUTPUT)
        print('Evidence ZIP:',archive)
        display(FileLink(archive))"""),
md("""## Đọc kết quả như researcher

Đọc toàn bộ trial, không chỉ dòng đẹp nhất. So objective giữa seed, số giao dịch, drawdown và ảnh hưởng khi tăng chi phí 1.5x / 2x. Chi phí nhạy cảm hoặc ít giao dịch là kết quả cần ghi nhận, không xóa trial để chạy lại.

Nếu best model thua dự báo 0 hoặc chiến lược âm, kết luận đúng là chưa tìm được tín hiệu hữu ích. Chọn thêm model sau khi xem diagnostic tạo một vòng nghiên cứu mới, không biến fold cũ thành test mới. Không dùng checkpoint TCN đã train toàn development để warm-start các fold lịch sử."""),
code("""trials=pd.read_csv(OUTPUT/'trials.csv')
display(trials.sort_values('value').head(20))
display(pd.DataFrame(result['cost_sensitivity']))
print('Selected params:',result['selection']['params'])
print('Across-seed mean/std normalized MSE:',result['selection']['mean_objective'],result['selection']['std_objective'])
print('Status:',result['deployment_status'])
print('Download QuantStream_Research_Output.zip rồi gửi lại để phân tích và commit kết quả thật.')"""),
])

save("05_Kaggle_LoRA_Lab.ipynb", [
md("""# QuantStream — phòng thực hành fine-tuning Chronos-2 + LoRA

**Chạy riêng sau notebook 04**, dùng cùng Dataset input mới, Internet On và GPU còn quota. Notebook này tải pretrained model, học adapter và so sánh với zero-shot trên cùng 512 cửa sổ development. Không chạy đồng thời với notebook 04.

Chạy từ **session mới** (Restart Session rồi Run All). Cell cài đặt gỡ `torchao` vì lab không dùng lượng tử hóa; bản cũ có thể làm PEFT lỗi ngay khi gắn LoRA. Cell cũng thử gắn adapter nhỏ trên CPU trước khi tải checkpoint. Không cài lại PyTorch/CUDA của Kaggle. Đây là bản sửa môi trường cho lần chạy sau, không thay thế evidence Run 003 đã hoàn thành.

Đây là fine-tuning thật: cập nhật adapter trọng số từ mô hình đã pretrained. TCN/GRU ở notebook 04 được train từ đầu. Pretraining overlap chưa xác minh nên kết quả Chronos chỉ phục vụ nghiên cứu/học tập, không chứng minh alpha độc lập.

Ba cấu hình được đặt trước: (100 steps, LR 1e-5), (200 steps, LR 1e-5), (200 steps, LR 3e-6). So A/B để học tác động số bước, B/C để học tác động learning rate. Giữ seed, context, batch size, base revision và cửa sổ chấm điểm cố định."""),
code("""import subprocess,sys,os,importlib.util,importlib.metadata,platform
# Package discovery is cached after imports; changing pip packages cannot clear it safely.
loaded=[name for name in ('peft','torchao','chronos','transformers')
        if any(module==name or module.startswith(name+'.') for module in sys.modules)]
if loaded:
    raise RuntimeError('Restart Session rồi Run All trước khi cài thư viện: '+', '.join(loaded))
os.environ['HF_HUB_DISABLE_XET']='1'
# This lab uses ordinary float32 weights, not torchao quantization.
subprocess.check_call([sys.executable,'-m','pip','uninstall','-y','torchao'])
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'chronos-forecasting==2.3.2','peft==0.21.2','accelerate>=1.1,<2',
    'datasets>=3,<5','pyarrow>=17,<27','scikit-learn>=1.5,<2'])
importlib.invalidate_caches()
if importlib.util.find_spec('torchao') is not None:
    raise RuntimeError('torchao vẫn còn sau khi gỡ; kiểm tra pip output và Restart Session.')
import torch
from peft import LoraConfig,get_peft_model
from peft.import_utils import is_torchao_available
if is_torchao_available():
    raise RuntimeError('Lab yêu cầu PEFT chạy không có torchao.')
# Exercise the same adapter-injection path that previously failed, without a download.
smoke=get_peft_model(torch.nn.Sequential(torch.nn.Linear(4,4)),
                    LoraConfig(r=2,lora_alpha=4,target_modules=['0']))
smoke(torch.ones(1,4)).sum().backward()
if not any(p.requires_grad and p.grad is not None for p in smoke.parameters()):
    raise RuntimeError('LoRA preflight không tạo được gradient cho adapter.')
del smoke
LORA_ENVIRONMENT={'python':platform.python_version(),'torch':torch.__version__,
    'packages':{name:importlib.metadata.version(name) for name in
        ('chronos-forecasting','peft','transformers','accelerate','datasets')},
    'torchao_installed':False,'lora_injection_backward_ok':True}
LORA_PIP_FREEZE=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
print('LoRA preflight OK:',LORA_ENVIRONMENT)"""),
code(BOOTSTRAP),
code("""import torch,gc
from quantstream.search import load_snapshot
from quantstream.finetune import chronos_experiment
from IPython.display import display,FileLink
load_snapshot(SNAPSHOT)  # Verify data hashes and reject final-holdout contamination.
if not torch.cuda.is_available():
    raise RuntimeError('Lab này cần GPU để giữ ngân sách thời gian. Chọn GPU khả dụng trước khi chạy.')
print(torch.__version__,torch.cuda.get_device_name(0))
BASE_REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'
CONFIGS=[{'id':'A_steps100_lr1e5','steps':100,'learning_rate':1e-5},
         {'id':'B_steps200_lr1e5','steps':200,'learning_rate':1e-5},
         {'id':'C_steps200_lr3e6','steps':200,'learning_rate':3e-6}]
OUTPUT=WORK/'run-003-lora-lab-rerun'
OUTPUT.mkdir(exist_ok=True)
plan={'base_revision':BASE_REVISION,'configurations':CONFIGS,'seed':42,
      'context_length':128,'batch_size':8,'max_windows':512,
      'snapshot_manifest_sha256':hashlib.sha256((SNAPSHOT/'manifest.json').read_bytes()).hexdigest()}
from quantstream.common import write_json
if (OUTPUT/'plan.json').exists() and json.loads((OUTPUT/'plan.json').read_text())!=plan:
    raise ValueError('Không thay đổi protocol của thí nghiệm đã bắt đầu.')
write_json(OUTPUT/'plan.json',plan)
environment_path=OUTPUT/'environment.json'
if environment_path.exists() and json.loads(environment_path.read_text())!=LORA_ENVIRONMENT:
    raise ValueError('Môi trường đã đổi; giữ output cũ để review trước khi chạy tiếp.')
write_json(environment_path,LORA_ENVIRONMENT)
freeze_path=OUTPUT/'pip-freeze.txt'
if freeze_path.exists() and freeze_path.read_text(encoding='utf-8')!=LORA_PIP_FREEZE:
    raise ValueError('Danh sách package đã đổi; giữ output cũ để review trước khi chạy tiếp.')
freeze_path.write_text(LORA_PIP_FREEZE,encoding='utf-8')"""),
md("""## Fine-tune và xuất evidence

Mỗi cấu hình bắt đầu từ **cùng pretrained checkpoint**, không nối tiếp adapter A → B → C. Nếu B kém A, chưa thể kết luận thêm bước luôn xấu; cần xem loss/seed ở vòng nghiên cứu sau. Không chỉnh cấu hình ngay sau khi thấy kết quả để rồi gọi validation là test.

Output mới dùng thư mục `run-003-lora-lab-rerun` và ZIP `QuantStream_LoRA_Rerun_Output.zip`, giữ riêng evidence Run 003 cũ. Không cần chạy lại A/B/C chỉ để xác nhận bản sửa cài đặt. Nếu chủ động chạy lại, đây vẫn là development diagnostic, không phải thí nghiệm ngoài mẫu mới.

ZIP giữ adapter, báo cáo, dự báo, phiên bản thư viện và trainer log dạng JSON/JSONL/log nếu có. Các optimizer checkpoint lớn được bỏ khỏi ZIP chia sẻ. Nếu phiên bị ngắt, Save Version/Download output; không xóa bằng chứng trial lỗi. ZIP được tạo cả khi lỗi, nên chỉ xem là hoàn tất khi có đủ ba báo cáo và `comparison.json`."""),
code("""reports=[]
try:
    for spec in CONFIGS:
        folder=OUTPUT/spec['id']
        report_path=folder/'chronos_report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
        else:
            report=chronos_experiment(SNAPSHOT,folder,steps=spec['steps'],
                learning_rate=spec['learning_rate'],max_windows=512,device='cuda',
                revision=BASE_REVISION,batch_size=8,context_length=128,seed=42)
        reports.append({'experiment':spec['id'],**report})
        gc.collect();torch.cuda.empty_cache()
    write_json(OUTPUT/'comparison.json',reports)
finally:
    # Include lightweight trainer evidence, never model/optimizer checkpoint binaries.
    with zipfile.ZipFile(WORK.parent/'QuantStream_LoRA_Rerun_Output.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in OUTPUT.rglob('*'):
            if path.is_file():
                relative=path.relative_to(OUTPUT)
                if 'trainer' not in relative.parts or path.suffix in {'.json','.jsonl','.log'}:
                    z.write(path,relative)
    display(FileLink(str(WORK.parent/'QuantStream_LoRA_Rerun_Output.zip')))"""),
code("""import pandas as pd
table=pd.DataFrame(reports)
table['mae_change']=table.finetuned_mae-table.zero_shot_mae
display(table[['experiment','steps','learning_rate','zero_shot_mae','finetuned_mae',
               'mae_change','zero_shot_directional_accuracy','finetuned_directional_accuracy']])
print('mae_change < 0: sai số giảm trên các cửa sổ development này. Chưa phải lợi nhuận giao dịch.')
print('Gửi QuantStream_LoRA_Rerun_Output.zip để review adapter và ghi kết quả lên Git.')"""),
])

print("Built notebooks 04 and 05")
