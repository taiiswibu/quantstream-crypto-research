"""Build the final code-only notebook with an explicit CPU/GPU execution record."""

import argparse
from pathlib import Path
import nbformat as nbf
from build_workflow_notebook import build

ROOT = Path(__file__).resolve().parents[1]


def final_notebook(source):
    n = build(source, write=False)
    for cell in n.cells:
        cell.source = cell.source.replace("Run006", "Run008").replace(
            "Run 006", "Run 008"
        )
    c = n.cells[0]
    c.source = c.source.replace(
        "BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()/'notebook_work'",
        "BASE = Path(os.environ.get('QUANTSTREAM_RUN_BASE', '/kaggle/working' if Path('/kaggle/working').exists() else str(Path.cwd()/'notebook_work')))\nBASE.mkdir(parents=True,exist_ok=True)",
    )
    c.source = c.source.replace(
        "CODE = BASE/('quantstream-source-'+SOURCE_COMMIT[:12])",
        "CODE = Path(os.environ.get('QUANTSTREAM_SOURCE_DIR',str(BASE/('quantstream-source-'+SOURCE_COMMIT[:12]))))",
    )
    c.source = c.source.replace("N_TRIALS = 24", "N_TRIALS = 8")
    c.source = c.source.replace("MAX_EPOCHS = 12", "MAX_EPOCHS = 96\nMIN_EPOCHS = 24")
    c.source = c.source.replace("JOB_SECONDS = 7200", "JOB_SECONDS = 10800")
    c.source = c.source.replace(
        "assert 4 <= N_TRIALS <= 24 and 1 <= MAX_EPOCHS <= 12 and JOB_SECONDS <= 21600",
        "assert N_TRIALS == 8 and 1 <= MIN_EPOCHS <= MAX_EPOCHS <= 120 and 60 <= JOB_SECONDS <= 14400",
    )
    c.source = c.source.replace(
        "if not torch.cuda.is_available():",
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\nif Path('/kaggle/working').exists() and DEVICE != 'cuda':",
    )
    c.source = c.source.replace(
        "torch.cuda.get_device_name(0)",
        "(torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU')",
    )
    c.source = c.source.replace(
        "from quantstream.workflow import prepare_dataset",
        "from quantstream.extended_training import prepare_dataset, find_parent_snapshot",
    )
    n.cells[1].source = n.cells[1].source.replace(
        "manifest = prepare_dataset(WORK)",
        "parent = Path(os.environ['QUANTSTREAM_PARENT_SNAPSHOT']) if os.environ.get('QUANTSTREAM_PARENT_SNAPSHOT') else find_parent_snapshot(Path('/kaggle/input'),BASE/'snapshot-cache')\n"
        "print('Snapshot:',str(parent) if parent else 'Collect the same frozen period')\nmanifest = prepare_dataset(WORK,parent)",
    )
    n.cells[
        7
    ].source = """# Tám điều kiện cân bằng; sau đó kiểm tra một finalist mỗi họ trên ba seed.
plan_path=WORK/'training_plan.json'
status=json.loads((WORK/'status.json').read_text()) if (WORK/'status.json').exists() else {}
if plan_path.exists():
    old=json.loads(plan_path.read_text())
    assert old['protocol']=='run008-final-v1' and old['source']['commit']==SOURCE_COMMIT
    assert old['dataset_manifest_sha256']==sha256(WORK/'dataset_manifest.json')
    assert (old['epochs'],old['minimum_epochs'],old['seconds'])==(MAX_EPOCHS,MIN_EPOCHS,JOB_SECONDS)
    if status.get('status')!='completed': raise RuntimeError('Giữ evidence của lượt chưa hoàn tất; không xóa để chạy lại chọn điểm')
    print('Đọc lượt đã hoàn tất; không train lại.')
else:
    log=WORK/'training-console.log'
    environment={**os.environ,'PYTHONPATH':str(CODE)+os.pathsep+os.environ.get('PYTHONPATH','')}
    started=time.monotonic()
    try:
        with log.open('w') as stream:
            worker=subprocess.Popen([sys.executable,'-m','quantstream.final_research','--root',str(WORK),
                '--epochs',str(MAX_EPOCHS),'--minimum-epochs',str(MIN_EPOCHS),'--seconds',str(JOB_SECONDS),'--device',DEVICE],
                cwd=str(CODE),env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while worker.poll() is None:
                if time.monotonic()-started > JOB_SECONDS+15:
                    import signal
                    os.killpg(worker.pid,signal.SIGTERM) if os.name=='posix' else worker.terminate()
                    try: worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid,signal.SIGKILL) if os.name=='posix' else worker.kill()
                        worker.wait()
                    raise TimeoutError('Dừng bởi watchdog; giữ partial evidence')
                clear_output(wait=True)
                print('Runtime:',DEVICE,'Elapsed:',round(time.monotonic()-started),'/',JOB_SECONDS,'seconds')
                if (WORK/'progress.json').exists(): print(json.loads((WORK/'progress.json').read_text()))
                print(''.join(log.read_text(errors='replace').splitlines(keepends=True)[-8:]))
                time.sleep(5)
            if worker.returncode: raise RuntimeError('Xem error.txt và training-console.log trong ZIP')
    finally:
        if 'worker' in locals() and worker.poll() is None:
            worker.terminate()
            try: worker.wait(timeout=10)
            except subprocess.TimeoutExpired: worker.kill();worker.wait()
        evidence=shutil.make_archive(str(BASE/'QuantStream_Run008_Output'),'zip',WORK)
        display(FileLink(evidence))
status=json.loads((WORK/'status.json').read_text())
display(pd.DataFrame([status]))"""
    n.cells[8].source = """# Giữ mọi điều kiện, cả model chưa vượt baseline.
display(pd.read_csv(WORK/'trials.csv'))
from quantstream.final_plots import figures
figures(WORK)"""
    n.cells[
        9
    ].source = """# Quyết định chỉ dùng development; khoảng tuần là chẩn đoán xấp xỉ.
selection=json.loads((WORK/'selection.json').read_text())
display(pd.DataFrame(selection['shortlist']))
print('Best research configuration:',selection['params'])
print('Recommended:',selection['recommended_model'],'Signal approved:',selection['signal_approved'])
print('Validation đã dùng nghiên cứu nhiều lần; không phải bằng chứng đầu tư độc lập.')"""
    n.cells[
        11
    ].source += "\nprint('Artifact role:',exported.get('artifact_role'))\nprint('Training last label:',exported.get('training_last_label'))\nprint('Independent test of these refit weights:',exported.get('test'))\nprint('Model chưa đạt gate vẫn ghi dự báo; app không nhận lệnh paper mới.')"
    n.cells[
        12
    ].source = """# Gói nhỏ để nhập local; giữ riêng checkpoint đánh giá và weights refit mới.
from quantstream.final_plots import diagnostic_zip
evidence=shutil.make_archive(str(BASE/'QuantStream_Run008_Output'),'zip',WORK)
model_zip=shutil.make_archive(str(BASE/'QuantStream_Run008_Model'),'zip',DEPLOYMENT)
audit_zip=shutil.make_archive(str(BASE/'QuantStream_Run008_Audit_Model'),'zip',WORK/'audit_model/deployment')
diagnostics=diagnostic_zip(WORK,BASE/'QuantStream_Run008_Diagnostics.zip')
for file in [model_zip,audit_zip,str(diagnostics),evidence]: display(FileLink(file))
print('Local: nạp Model.zip. Điểm hồi cứu chỉ thuộc Audit_Model.zip, không gán cho weights refit mới.')"""
    nbf.validate(n)
    destination = ROOT / "notebooks/11_Final_Research_and_Deployment.ipynb"
    nbf.write(n, destination)
    print(destination)
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    final_notebook(parser.parse_args().source)
