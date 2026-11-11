"""Build notebook 06 pinned to an existing clean source commit, without touching 04/05."""
import argparse
from pathlib import Path
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(source_commit, destination):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("An immutable 40-character source commit is required")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    setup_notebook = nbf.read(ROOT / "notebooks/05_Kaggle_LoRA_Lab.ipynb", as_version=4)
    setup = next(cell.source for cell in setup_notebook.cells if cell.cell_type == "code")
    cells = [
        md("""# QuantStream — Run 004: kiểm tra độ ổn định LoRA qua ba seed

## Mục tiêu

**Lần đầu giữ `MODE='preflight'` rồi Run All.** Dùng đúng Dataset đã chạy Run 003, bật Internet và GPU, chạy từ session mới. Không tạo lại dataset. Chưa có kết quả Run 004; lần đầu chỉ thử một optimizer step trên chuỗi sin/cos, không train trên dữ liệu tài chính.

Notebook tải code mới theo Git SHA cố định; chỉ lấy snapshot từ Dataset cũ. Không dùng code cũ nằm trong input ZIP.

Sau khi preflight xong, tải `QuantStream_Run004_Preflight.zip` gửi lại để kiểm tra môi trường và khóa cấu hình trước ba lần fit. Nếu lỗi, vẫn gửi ZIP cùng traceback. ZIP xuất hiện chưa có nghĩa đã thành công.

Kế hoạch research: giữ LoRA B (200 steps, LR 1e-5), seeds 42/123/2026, context 128, batch 8 và cùng 512 cửa sổ development đã xem. Không chọn seed đẹp nhất và không promote model."""),
        md("## 1. Cài thư viện và thử gắn adapter\n\nRestart Session trước Run All. Giữ PyTorch/CUDA của Kaggle; gỡ optional torchao như Notebook 05."),
        code(setup),
        md("## 2. Chọn chế độ\n\nGiữ nguyên các tham số. `NOTEBOOK_URL` chỉ cần điền trước khi chuyển sang train sau khi preflight được kiểm tra. Thư mục output cũ được giữ để tránh âm thầm chạy lại trial lỗi."),
        code(f"""from pathlib import Path
import sys, subprocess, json, zipfile, shutil, hashlib
from IPython.display import display, FileLink

MODE='preflight'
NOTEBOOK_URL=''  # Khi train: URL thật https://www.kaggle.com/code/<user>/<notebook>
SOURCE_COMMIT='{source_commit}'
WORK=Path('/kaggle/working/quantstream_run004')
WORK.mkdir(parents=True,exist_ok=True)
CODE=WORK/'source'
SNAPSHOT=WORK/'snapshot'
PREFLIGHT=WORK/'preflight'
OUTPUT=WORK/'run-004-seed-stability'
LOCK=PREFLIGHT/'candidate-lock.json'
if MODE not in {{'preflight','train'}}:
    raise ValueError('MODE phải là preflight hoặc train')
print('Mode:',MODE,'Source:',SOURCE_COMMIT)"""),
        md("## 3. Nạp đúng code và snapshot\n\nSnapshot manifest phải có SHA-256 `c8e01129…747fd0`. Nếu khác, dừng và giữ lỗi để kiểm tra; không đổi hash cho qua. Không đọc final holdout."),
        code("""if not CODE.exists():
    subprocess.check_call(['git','init','-q',str(CODE)])
    subprocess.check_call(['git','-C',str(CODE),'remote','add','origin',
                           'https://github.com/taiiswibu/quantstream-crypto-research.git'])
    subprocess.check_call(['git','-C',str(CODE),'fetch','-q','--depth','1','origin',SOURCE_COMMIT])
    subprocess.check_call(['git','-C',str(CODE),'checkout','-q','--detach','FETCH_HEAD'])
actual=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
if actual!=SOURCE_COMMIT:
    raise ValueError('Source khác SHA đã khóa; giữ output để kiểm tra')
if subprocess.check_output(['git','-C',str(CODE),'status','--porcelain'],text=True).strip():
    raise ValueError('Source working tree phải sạch')
if any(name=='quantstream' or name.startswith('quantstream.') for name in sys.modules):
    raise RuntimeError('Restart Session để không dùng nhầm quantstream đã import')
sys.path.insert(0,str(CODE))

INPUT=Path('/kaggle/input')
archives=list(INPUT.rglob('QuantStream_Kaggle_Input.zip'))
if len(archives)>1:
    raise ValueError('Chỉ Add Input một phiên bản dataset Run 003')
if not SNAPSHOT.exists():
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            for info in archive.infolist():
                name=Path(info.filename)
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('Unsafe ZIP path')
                if name.parts and name.parts[0]=='snapshot':
                    archive.extract(info,WORK)
    else:
        manifests=[p for p in INPUT.rglob('manifest.json')
                   if (p.parent/'development.parquet').exists()]
        if len(manifests)!=1:
            raise FileNotFoundError('Add Input đúng dataset cũ đã chạy Run 003')
        shutil.copytree(manifests[0].parent,SNAPSHOT)
from quantstream.seed_lab import preflight_run004,run_seed_lab,archive_evidence,SNAPSHOT_SHA
if hashlib.sha256((SNAPSHOT/'manifest.json').read_bytes()).hexdigest()!=SNAPSHOT_SHA:
    raise ValueError('Snapshot không phải bản Run 003 đã khóa. Gửi lỗi để kiểm tra.')
print('Source và manifest đúng SHA. Chưa chạy research fit.')"""),
        md("""## 4. Chạy và tải evidence

`preflight` tải checkpoint cố định và thử fit/predict với dữ liệu tổng hợp trên GPU. Lưu phiên bản Python/PyTorch/CUDA, package versions, cấu hình LoRA đầy đủ, source/data hashes và số optimizer step thật. Đây là bước kiểm tra môi trường, tách khỏi ngân sách 600 steps nghiên cứu.

Chỉ dùng `train` sau khi kiểm tra ZIP preflight và giữ nguyên lock/môi trường. Khi đó mỗi seed nạp lại pretrained model; ghi train loss mỗi step, trạng thái lỗi, dự báo và adapter riêng. Nếu phiên mất dữ liệu, gửi ZIP đã tải để được hướng dẫn khôi phục; không xóa trial rồi chạy lại nhằm lấy kết quả đẹp.

Nếu Kaggle bị kill cả process, `finally` có thể không chạy: Save Version/Download thư mục output còn lại. Notebook chưa chạy hết trên Kaggle GPU khi được phát hành; preflight này là bước cần bạn thực hiện trước."""),
        code("""target=PREFLIGHT if MODE=='preflight' else OUTPUT
filename='QuantStream_Run004_Preflight.zip' if MODE=='preflight' else 'QuantStream_Run004_Output.zip'
archive=WORK.parent/filename
# Refuse before entering finally, so a repeated click cannot overwrite a previous archive.
if target.exists() or archive.exists():
    raise FileExistsError('Đã có evidence. Tải ZIP cũ và gửi kiểm tra, không xóa để chạy lại.')
try:
    if MODE=='preflight':
        result=preflight_run004(SNAPSHOT,PREFLIGHT)
        print('Preflight hoàn tất. Research fits đã chạy: 0')
        print('Tải QuantStream_Run004_Preflight.zip và gửi lại trước khi train.')
    else:
        result=run_seed_lab(SNAPSHOT,OUTPUT,LOCK,NOTEBOOK_URL)
        import pandas as pd
        display(pd.DataFrame(result['metrics']))
        print('Deployment:',result['deployment_status'])
finally:
    saved=archive_evidence(target,archive)
    if saved:
        display(FileLink(saved))"""),
        md("""## 5. Đọc kết quả

Preflight thành công khi `status.json` ghi `completed`, có `candidate-lock.json`, smoke `global_step=1`, adapter và forecast hợp lệ. Chưa có kết quả tài chính nào ở giai đoạn này.

Research hoàn tất khi đủ ba seed, mỗi seed `global_step=200`, cùng 512 hàng/baseline và có `comparison.json`. MAE delta âm là giảm sai số trên development đã xem; không suy ra lợi nhuận. Báo mọi seed và từng tài sản, mean/std, paired bootstrap theo tuần UTC. Giữ `research_only_no_promotion` dù có seed tốt hơn zero-shot."""),
    ]
    destination = Path(destination)
    if destination.exists():
        previous = nbf.read(destination, as_version=4)
        for old, new in zip(previous.cells, cells):
            if old.cell_type == new.cell_type:
                new.id = old.id
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    nbf.validate(notebook)
    nbf.write(notebook, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/06_Kaggle_LoRA_Seed_Stability.ipynb")
    args = parser.parse_args()
    build(args.source_commit, args.output)
