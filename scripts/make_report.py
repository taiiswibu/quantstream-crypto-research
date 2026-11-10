"""Build an offline HTML report directly from executed experiment artifacts."""
import html
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / json.loads((ROOT / "artifacts/latest_run.json").read_text())["run"]
report = json.loads((run / "final_report.json").read_text())
quality = json.loads((ROOT / "data/processed/data_quality.json").read_text())
comparison = json.loads((run / "comparison.json").read_text())
tcn = json.loads((ROOT / "artifacts/tcn_training/tcn_metadata.json").read_text())
chronos_path = ROOT / "artifacts/chronos_200/chronos_report.json"
chronos = json.loads(chronos_path.read_text()) if chronos_path.exists() else {}
curve = pd.read_parquet(run / "test_equity.parquet")
fig = go.Figure()
for column, label, colour in [("model", "TCN", "#39d98a"), ("buy_hold", "Buy and hold", "#7997ff"), ("ema", "EMA", "#eeb56d")]:
    fig.add_trace(go.Scatter(x=curve.timestamp, y=curve[column], name=label, line_color=colour))
fig.update_layout(template="plotly_dark", paper_bgcolor="#111f2b", plot_bgcolor="#111f2b", height=480,
                  title="Final holdout · vốn ban đầu = 1 · đã tính chi phí", margin=dict(l=30, r=30, t=60, b=30))
chart = fig.to_html(full_html=False, include_plotlyjs=True)
rows = []
for label, key in [("TCN", "strategy"), ("Buy and hold", "buy_hold"), ("EMA", "ema")]:
    m = report[key]
    rows.append(f"<tr><td>{label}</td><td>{m['total_return']:+.2%}</td><td>{m['max_drawdown']:.2%}</td><td>{m['sharpe_hourly_estimate']:.2f}</td><td>{m['trades']}</td></tr>")
selection = pd.DataFrame(comparison).drop(columns=["last_fold", "selection_score"])
extra = {"model": "tcn", "threshold": tcn["threshold"], **{k: tcn["selection"][k] for k in selection.columns if k in tcn["selection"]}}
selection = pd.concat([selection, pd.DataFrame([extra])], ignore_index=True)
selection = selection.sort_values("mean_validation_sharpe", ascending=False)
selection_html = selection.to_html(index=False, float_format=lambda v: f"{v:.4f}", classes="compact")
chronos_html = ""
if chronos:
    chronos_html = f"""<h2>Fine-tuning Chronos-2</h2><p>Đã chạy LoRA {chronos['steps']} steps trên CPU, đánh giá {chronos['evaluation_windows']} cửa sổ validation được lấy mẫu cố định. Checkpoint gốc: <code>{html.escape(chronos['resolved_revision'])}</code>.</p>
    <table><tr><th>Chỉ tiêu</th><th>Zero-shot</th><th>Sau LoRA</th></tr>
    <tr><td>MAE lợi suất</td><td>{chronos['zero_shot_mae']:.6f}</td><td>{chronos['finetuned_mae']:.6f}</td></tr>
    <tr><td>Đúng hướng</td><td>{chronos['zero_shot_directional_accuracy']:.2%}</td><td>{chronos['finetuned_directional_accuracy']:.2%}</td></tr></table>
    <p>Đây là thí nghiệm exploratory. Chưa xác minh trùng lặp dữ liệu pretraining; kết quả này không dùng để chọn model cho final holdout.</p>"""
document = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuantStream · Báo cáo chạy thật</title>
<style>body{{font:16px/1.65 system-ui;background:#0b1118;color:#d9e5ef;max-width:1150px;margin:40px auto;padding:0 24px}}h1{{font-size:46px;color:#39d98a;margin-bottom:0}}h2{{margin-top:38px}}.muted{{color:#9db0bf}}.card{{background:#111f2b;border:1px solid #253d47;padding:22px;border-radius:14px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #2a3d4c;text-align:left}}th{{color:#39d98a}}code{{overflow-wrap:anywhere}}.negative{{color:#ff8598}}a{{color:#39d98a}}.scroll{{overflow-x:auto}}</style>
<h1>QuantStream</h1><p class="muted">Local Crypto Research · dữ liệu thật · báo cáo từ các lượt train đã chạy</p>
<div class="card"><b>Trạng thái: MVP nghiên cứu.</b> TCN được chọn trên validation. Trên final holdout, chiến lược có lợi nhuận <b class="negative">{report['strategy']['total_return']:+.2%}</b> sau chi phí. Chưa có bằng chứng để dùng chiến lược giao dịch bằng tiền thật.</div>
<h2>Dữ liệu và phương pháp</h2><p>{quality['valid_rows']:,} nến BTC/USDT và ETH/USDT, khung 1 giờ, từ 01/2023 đến hết 10/2026. Kiểm tra SHA-256 cho 92 ZIP nguồn; {quality['quarantine_rows']} dòng được quarantine. Không điền giả các nến thiếu; indicators khởi động lại sau khoảng trống.</p>
<p>18 features, ba fold walk-forward dùng chung mốc UTC cho hai tài sản. Loại mẫu có nhãn vượt ranh giới split. Model dự báo lợi suất của vị thế giữ 4 giờ sau khi chờ 1 giờ để khớp. Mỗi tài sản có phần vốn riêng; 95% phần vốn được dùng cho mỗi vị thế.</p>
<p>Phí mỗi chiều {report['assumptions']['fee_bps_each_side']} bps, slippage mỗi chiều {report['assumptions']['slippage_bps_each_side']} bps. Halt khi drawdown theo từng phần vốn chạm 20%, kiểm tra ở giá đóng nến. Đây không phải stop-loss nội nến.</p>
<h2>Final holdout</h2><p class="muted">{html.escape(report['period']['start'])} → {html.escape(report['period']['end'])}</p>
<table><tr><th>Chiến lược</th><th>Lợi nhuận</th><th>Max drawdown</th><th>Sharpe ước tính</th><th>Giao dịch</th></tr>{''.join(rows)}</table>
{chart}<p>Directional accuracy: {report['prediction']['directional_accuracy']:.2%}; MAE lợi suất: {report['prediction']['mae_return']:.6f}. Sharpe quy đổi theo giờ chưa hiệu chỉnh tự tương quan.</p>
<h2>Chọn model trên validation</h2><p>Chọn bằng Sharpe trung bình ba fold, yêu cầu tối thiểu 20 giao dịch gộp. TCN được chọn trước khi mở test; mức chênh lệch validation với Ridge nhỏ, chưa đủ để kết luận TCN vượt trội ổn định.</p><div class="scroll">{selection_html}</div>
{chronos_html}
<h2>Giới hạn và bước nghiên cứu tiếp</h2><p>Hai tài sản được chọn có selection bias. Giá khớp từ OHLCV là xấp xỉ; không có order book hoặc kiểm tra capacity. Mức phí là giả định, không phải biểu phí tài khoản cụ thể. Backtest sau phí chưa đạt yêu cầu lợi nhuận. Muốn cải tiến cần hypothesis mới, kiểm tra ổn định theo regime và một giai đoạn đánh giá tương lai mới; không tiếp tục chọn model bằng final holdout này.</p>
<p class="muted">Source, notebook, model weights, manifests và báo cáo JSON nằm cùng gói dự án. Mở README.md để chạy local hoặc train lại trên Kaggle.</p></html>"""
(ROOT / "BAO_CAO_KET_QUA.html").write_text(document, encoding="utf-8")
print(ROOT / "BAO_CAO_KET_QUA.html")
