"""Build notebook 07 with one Run All path and no training/preflight mode switch."""
import argparse
from pathlib import Path
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(source_commit, destination):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Pin an immutable source commit")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# QuantStream — Run 005: đối chứng inference với trọng số cố định

**Chỉ cần Run All một lần. Không có biến MODE và không fine-tune lại.** Runner dùng adapter seed 42 đã chạy ở Run 004, nạp dạng PEFT không merge và so sánh `eval()` với dropout bật dưới `torch.inference_mode()`.

Trước khi chạy:

1. Import notebook này vào Kaggle, bật Internet và chọn **Tesla T4 như Run 004**. Bắt đầu session mới.
2. Add Input **Dataset snapshot cũ** đang dùng cho Run 004.
3. Add Input thêm **`QuantStream_Run004_Output.zip`** đã gửi để review. Nếu Kaggle giải nén ZIP thành các thư mục `seed-42`, `seed-123`, `seed-2026`, runner vẫn kiểm tra đúng file/hashes. Không dùng ZIP preflight.
4. Chọn **Run All**. Tải `QuantStream_Run005_Output.zip` gửi lại, kể cả khi có lỗi. Nếu lỗi trước khi ZIP được tạo, gửi traceback.

Ngân sách cố định: một lượt zero-shot để đối chiếu và tám lượt cùng adapter, mỗi lượt 512 cửa sổ development đã xem; **0 optimizer steps**. Chưa có kết quả Run 005 trên Kaggle khi phát hành notebook này. Mục tiêu là đo ảnh hưởng của mode/RNG, không tìm model hoặc seed mới."""),
        md("## 1. Nạp source đã pin\n\nMỗi source SHA dùng checkout riêng. Dữ liệu và output nằm ngoài thư mục source để giữ Git tree sạch."),
        code(f"""from pathlib import Path
import sys, subprocess, json, zipfile, traceback, importlib.metadata, platform
from IPython.display import display, FileLink

loaded=[name for name in ('peft','torchao','chronos','transformers','quantstream')
        if any(module==name or module.startswith(name+'.') for module in sys.modules)]
if loaded:
    raise RuntimeError('Restart Session rồi Run All: '+', '.join(loaded))
SOURCE_COMMIT='{source_commit}'
WORK=Path('/kaggle/working/quantstream_run005')
WORK.mkdir(parents=True,exist_ok=True)
CODE=WORK/('source-'+SOURCE_COMMIT[:12])
OUTPUT=WORK/'run-005-inference'
ARCHIVE=WORK.parent/'QuantStream_Run005_Output.zip'
if OUTPUT.exists() or ARCHIVE.exists():
    raise FileExistsError('Đã có evidence Run 005; tải lại để review, không xóa nhằm chạy lại.')
if not CODE.exists():
    subprocess.check_call(['git','init','-q',str(CODE)])
    subprocess.check_call(['git','-C',str(CODE),'remote','add','origin',
                           'https://github.com/taiiswibu/quantstream-crypto-research.git'])
    subprocess.check_call(['git','-C',str(CODE),'fetch','-q','--depth','1','origin',SOURCE_COMMIT])
    subprocess.check_call(['git','-C',str(CODE),'checkout','-q','--detach','FETCH_HEAD'])
actual=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
if actual!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(CODE),'status','--porcelain'],text=True).strip():
    raise ValueError('Cần source SHA đã pin và checkout sạch')
sys.path.insert(0,str(CODE))
print('Pinned source:',SOURCE_COMMIT)"""),
        md("""## 2. Giữ môi trường Run 004

Cell giữ PyTorch/CUDA/NumPy do Kaggle cung cấp, gỡ optional torchao và cài đúng phiên bản thư viện đã review. Constraint inventory lấy từ lock Run 004. Runner sẽ so toàn bộ inventory trước khi tải checkpoint; nếu môi trường khác, dừng và xuất ZIP để kiểm tra.

Lock là version inventory, chưa phải container image hoặc bảo đảm bitwise tái lập. Python/PyTorch của base image khác sẽ được báo ra, không tự thay PyTorch GPU."""),
        code("""import os, importlib.util
os.environ['HF_HUB_DISABLE_XET']='1'
old_lock=json.loads((CODE/'reports/experiments/run-004-seed-stability/kaggle-preflight-lock-rerun-001.json').read_text())
expected=old_lock['environment']
constraints=WORK/'run004-constraints.txt'
constraints.write_text(''.join(f'{name}=={version}\\n' for name,version in expected['packages'].items()),encoding='utf-8')
try:
    if platform.python_version()!=expected['python']:
        raise RuntimeError('Python base image khác Run 004: '+platform.python_version())
    for name in ('torch','numpy'):
        actual=importlib.metadata.version(name)
        if actual!=expected['packages'][name]:
            raise RuntimeError(name+' base image khác Run 004: '+actual)
    subprocess.check_call([sys.executable,'-m','pip','uninstall','-y','torchao'])
    packages=('chronos-forecasting','peft','transformers','accelerate','datasets',
              'huggingface-hub','safetensors','pyarrow','pandas','scikit-learn','tokenizers')
    subprocess.check_call([sys.executable,'-m','pip','install','-q','-c',str(constraints),
                          *[name+'=='+expected['packages'][name] for name in packages]])
    importlib.invalidate_caches()
    if importlib.util.find_spec('torchao') is not None:
        raise RuntimeError('torchao vẫn còn; giữ traceback và Restart Session sau review')
except BaseException:
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/'setup-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    (OUTPUT/'status.json').write_text(json.dumps({'status':'setup_failed','optimizer_steps':0}),encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.iterdir():
            archive.write(path,path.name)
    display(FileLink(str(ARCHIVE)))
    raise
print('Dependency setup completed. No fitting or backward pass was performed.')"""),
        md("""## 3. Chạy đối chứng và tải ZIP

Runner dùng cùng checkpoint, adapter, batch 16, context 128, horizon và cách chuyển forecast log-price thành return của Run 004. Adapter được nạp không merge; tất cả parameters bị freeze. Mỗi lượt ghi mode, RNG fingerprints, full model-state hash trước/sau, dự báo từng hàng và metric theo tài sản.

Các case đã đặt trước: eval cùng RNG hai lần; eval khác RNG; dropout bật cùng RNG hai lần; dropout bật với RNG 123/2026; cuối cùng eval lại sau các lượt dropout. Bản đối chiếu cùng seed 42 của Run 004 cũng được giữ trong ZIP. Không chọn lượt có MAE đẹp nhất.

Nếu các control hoặc reference không khớp tolerance rtol 1e-6 / atol 1e-9, báo cáo ghi rõ; không tự sửa precision hoặc chạy lại. Nếu trọng số/buffer thay đổi hoặc có NaN, runner dừng và giữ partial evidence."""),
        code("""from quantstream.inference_lab import prepare_inputs,run_inference_lab
from quantstream.seed_lab import archive_evidence
try:
    snapshot,reference=prepare_inputs(Path('/kaggle/input'),WORK)
    report=run_inference_lab(snapshot,reference,OUTPUT,device='cuda')
    print('Optimizer steps:',report['optimizer_steps'])
    print('Controls passed:',report['controls_passed'])
    print('Run 004 reference reproduced:',report['reference_reproduced'])
    print('Đã xong inference diagnostic. Tải QuantStream_Run005_Output.zip và gửi lại.')
except BaseException:
    if not OUTPUT.exists():
        OUTPUT.mkdir(parents=True)
        (OUTPUT/'input-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (OUTPUT/'status.json').write_text(json.dumps({'status':'input_failed','optimizer_steps':0}),encoding='utf-8')
    raise
finally:
    saved=archive_evidence(OUTPUT,ARCHIVE)
    if saved:
        display(FileLink(saved))"""),
        md("""## Cách đọc kết quả

`completed` xác nhận đủ lượt đã chạy, không tự xác nhận control/reference đều đạt. Đọc riêng `controls_passed`, `reference_reproduced`, model-state hashes và mọi dự báo. Các RNG seed trong lab là seed của **inference**, không phải training seed mới; adapter luôn là bản seed 42.

Một mode effect trong môi trường này chưa tái dựng được chính xác runtime/RNG lịch sử không được lưu ở Run 003. Dữ liệu development cũ không trở thành test độc lập. Giữ `research_only_no_promotion`; không có PnL, forward evaluation hoặc model được triển khai."""),
    ]
    destination = Path(destination)
    if destination.exists():
        old = nbf.read(destination, as_version=4)
        for previous, current in zip(old.cells, cells):
            if previous.cell_type == current.cell_type:
                current.id = previous.id
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    nbf.validate(notebook)
    nbf.write(notebook, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/07_Kaggle_Inference_Check.ipynb")
    args = parser.parse_args()
    build(args.source_commit, args.output)
