"""Build a standalone, read-only Kaggle probe for blocked Run 005 environments."""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    source = (ROOT / "scripts/diagnose_run005_runtime.py").read_text()
    assert source.count('\nif __name__ == "__main__":') == 1
    definitions = source.split('\nif __name__ == "__main__":', 1)[0]
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("""# QuantStream — kiểm tra runtime Run 005

**Run All một lần**, Internet không bắt buộc; không cần Add Input. Có thể dùng CPU để không tốn quota GPU; chỉ bật GPU nếu cần kiểm tra luôn driver/CUDA. Notebook không cài/gỡ package, không tải checkpoint, không train và không đọc/xóa output cũ.

Lock Run 004 ghi Python 3.13.16 và NumPy 1.26.4. NumPy 1.26.4 chỉ hỗ trợ chính thức Python 3.9–3.12. Ngoài ra, lock cũ lấy version cuối cùng của metadata trùng tên, còn cell setup gọi `metadata.version()` lấy match đầu; hai cách có thể trả kết quả khác nhau ngay trong cùng môi trường. Chưa biết trường hợp nào đang xảy ra, nên không ép downgrade hoặc đổi lock.

Probe giữ mọi metadata candidate, module đã nạp trong kernel và phiên bản/file thật từ subprocess sạch, cùng kiểm tra NumPy↔PyTorch và `pip check`. Mỗi lần tạo tên output riêng. Tải **QuantStream_Run005_Runtime_Probe_…zip** bên dưới và đính kèm trực tiếp vào chat. Link `/kaggle/working/...` chỉ là link tải trong phiên Kaggle, không thay thế file đính kèm.

[NumPy 1.26.4 release notes](https://numpy.org/doc/stable/release/1.26.4-notes.html). Đây là kiểm tra môi trường, không phải kết quả inference Run 005."""),
        nbf.v4.new_code_cell(definitions),
        nbf.v4.new_code_cell("""from IPython.display import display, FileLink
archive,report=run_probe(Path('/kaggle/working'))
print('Kernel Python:',report['kernel']['python'])
print('NumPy metadata first match:',report['inventory']['metadata_version_first_match'].get('numpy'))
print('NumPy metadata candidates:',report['inventory']['all_candidates'].get('numpy'))
print('Fresh-process NumPy:',report.get('fresh_process_modules',{}).get('modules',{}).get('numpy'))
display(FileLink(str(archive)))"""),
    ])
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    notebook.metadata.probe_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    destination = Path(destination)
    if destination.exists():
        old = nbf.read(destination, as_version=4)
        for before, after in zip(old.cells, notebook.cells):
            if before.cell_type == after.cell_type:
                after.id = before.id
    nbf.validate(notebook)
    nbf.write(notebook, destination)


if __name__ == "__main__":
    build(ROOT / "notebooks/07b_Kaggle_Runtime_Probe.ipynb")
