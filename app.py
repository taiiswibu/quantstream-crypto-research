"""Read-only research dashboard plus explicit local paper-trading refresh."""
from pathlib import Path
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from quantstream.common import Config

ROOT = Path(os.environ.get("QUANTSTREAM_ROOT", Path(__file__).resolve().parent)).resolve()
st.set_page_config(page_title="QuantStream · Local Research", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp{background:#0b1118}.block-container{padding-top:2rem}
[data-testid="stMetric"]{background:#111f2b;border:1px solid #213a43;border-radius:12px;padding:14px}
h1,h2,h3{letter-spacing:-.025em} .stCaption{color:#93a7b5}
</style>""", unsafe_allow_html=True)


def read_json(path, default=None):
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


@st.cache_data(show_spinner=False)
def parquet(path, modified):
    return pd.read_parquet(path)


def get_frame(path):
    p = ROOT / path
    return parquet(str(p), p.stat().st_mtime_ns) if p.exists() else pd.DataFrame()


def style_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1118", plot_bgcolor="#0b1118",
                      font=dict(color="#cbd5e1"), margin=dict(l=10, r=10, t=30, b=15),
                      legend=dict(orientation="h", y=1.12), height=430)
    return fig


cfg = Config.load(ROOT / "config.json")
metadata = read_json("artifacts/champion.json", {})
latest = read_json("artifacts/latest_run.json", {})
run = latest.get("run", "runs/not_yet_created")
report = read_json(f"{run}/final_report.json", {})
quality = read_json("data/processed/data_quality.json", {})
paper = read_json("paper/state.json", {})
bars = get_frame("data/processed/bars.parquet")

st.title("QuantStream")
st.caption("CRYPTO RESEARCH LAB · dữ liệu thật · chạy local · paper trading")
with st.sidebar:
    st.subheader("Research configuration")
    symbol = st.selectbox("Tài sản", cfg.symbols)
    st.write("Nến **1 giờ** · giữ vị thế **4 giờ**")
    st.write(f"Chờ khớp: **{cfg.execution_lag_bars} giờ** sau tín hiệu")
    st.write(f"Phí mỗi chiều: **{cfg.fee_bps:g} bps**")
    st.write(f"Slippage mỗi chiều: **{cfg.slippage_bps:g} bps**")
    st.divider()
    st.caption("Không kết nối tài khoản giao dịch. Số dư và lệnh trong app là mô phỏng.")
    if st.button("Đọc lại kết quả", width="stretch"):
        st.cache_data.clear()
        st.rerun()

overview, backtest, models, paper_tab, data_tab = st.tabs(["Thị trường", "Backtest", "Models", "Paper trading", "Dữ liệu"])
with overview:
    if bars.empty:
        st.info("Chạy: python -m quantstream demo")
    else:
        asset = bars[bars.symbol.eq(symbol)].sort_values("timestamp")
        last = asset.iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric("Giá đóng nến gần nhất", f"{last.close:,.2f} USDT")
        change = (last.close / asset.iloc[-25].close - 1) if len(asset) >= 25 else 0
        b.metric("Thay đổi 24 giờ", f"{change:+.2%}")
        c.metric("Nến đã lưu", f"{len(asset):,}")
        d.metric("Model đang chọn", metadata.get("model", "Chưa train"))
        st.caption(f"Nến lịch sử cuối: {last.timestamp} UTC. Giá ở đây là giá từ dataset đã lưu, không mặc định là giá hiện tại.")
        window = st.select_slider("Số nến hiển thị", options=[48, 168, 336, 720], value=168)
        view = asset.tail(window)
        fig = go.Figure(go.Candlestick(x=view.timestamp, open=view.open, high=view.high, low=view.low,
                                       close=view.close, increasing_line_color="#39d98a", decreasing_line_color="#ff6b81"))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(style_chart(fig), width="stretch")
        st.caption("Dự báo và hiệu quả chiến lược được đánh giá riêng; mô hình có thể thua benchmark sau phí.")
with backtest:
    if not report:
        st.info("Chưa mở final holdout. So sánh validation trong tab Models, chọn champion rồi chạy: python -m quantstream evaluate")
    else:
        st.write(f"Kết quả final holdout của **{report['model']}**")
        st.caption(f"{report['period']['start']} → {report['period']['end']}")
        if report["model_sha256"] != metadata.get("sha256"):
            st.warning("Báo cáo này thuộc model phiên bản trước. Model đang chọn chưa được đánh giá trên một holdout mới.")
        m = report["strategy"]
        a, b, c, d = st.columns(4)
        a.metric("Lợi nhuận sau chi phí", f"{m['total_return']:+.2%}")
        b.metric("Maximum drawdown", f"{m['max_drawdown']:.2%}")
        c.metric("Sharpe ước tính", f"{m['sharpe_hourly_estimate']:.2f}")
        d.metric("Số giao dịch", f"{m['trades']:,}")
        curve = get_frame(f"{run}/test_equity.parquet")
        if not curve.empty:
            fig = px.line(curve, x="timestamp", y=["model", "buy_hold", "ema"],
                          color_discrete_sequence=["#39d98a", "#7997ff", "#eeb56d"])
            fig.update_yaxes(title="Vốn ban đầu = 1")
            st.plotly_chart(style_chart(fig), width="stretch")
            drawdown = curve[["timestamp"]].copy()
            drawdown["drawdown"] = curve.model / curve.model.cummax().clip(lower=1) - 1
            st.plotly_chart(style_chart(px.area(drawdown, x="timestamp", y="drawdown", color_discrete_sequence=["#ff6b81"])), width="stretch")
        table = [{"Chiến lược": name, **report[key]} for name, key in [("Model", "strategy"), ("Buy and hold", "buy_hold"), ("EMA", "ema")]]
        st.dataframe(pd.DataFrame(table).drop(columns="note"), width="stretch", hide_index=True)
        trades = get_frame(f"{run}/test_trades.parquet")
        st.subheader("Giao dịch mô phỏng trong backtest")
        st.dataframe(trades.tail(100), width="stretch", hide_index=True)
        st.caption("Sharpe dùng quy đổi theo giờ, chưa hiệu chỉnh tự tương quan. Khớp lệnh từ OHLCV là xấp xỉ; phí là giả định cấu hình.")
with models:
    comparison = read_json(f"{run}/comparison.json", [])
    if comparison:
        table = pd.DataFrame(comparison).drop(columns=["last_fold", "selection_score"])
        tcn_meta = read_json("artifacts/tcn_training/tcn_metadata.json", {})
        if tcn_meta:
            row = {"model": "tcn", "threshold": tcn_meta["threshold"]}
            row.update({k: v for k, v in tcn_meta["selection"].items() if k in table.columns})
            table = pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values("mean_validation_sharpe", ascending=False)
        st.write("Chọn model và threshold bằng các fold validation theo thời gian")
        st.dataframe(table, width="stretch", hide_index=True)
    if metadata:
        st.write("Model artifact và lineage")
        st.json({k: metadata[k] for k in ["model", "type", "threshold", "sha256", "training_rows", "features_sha256", "deployment_status"] if k in metadata})
        st.write("Feature dùng chung khi train và inference")
        st.code(", ".join(metadata.get("features", [])))
    st.info("TCN: notebooks/02_Kaggle_TCN.ipynb · LoRA: notebooks/03_Kaggle_Chronos_LoRA.ipynb")
    st.caption("Train PyTorch và fine-tune pretrained là hai thí nghiệm khác nhau. Chronos được giữ ở mức exploratory khi chưa xác minh dữ liệu pretraining.")
with paper_tab:
    st.write("Danh mục mô phỏng · vốn khởi tạo 10.000 USDT")
    if st.button("Lấy dữ liệu mới và chạy một bước paper", disabled=not bool(metadata)):
        try:
            from quantstream.paper import paper_step
            with st.spinner("Cập nhật dữ liệu thị trường công khai..."):
                paper = paper_step(ROOT, cfg)
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Paper step chưa hoàn tất: {e}")
    st.caption("Để chạy liên tục: python -m quantstream paper --watch. App đọc trạng thái khi tải lại trang; không giả lập giao dịch trong lúc máy tắt.")
    if paper:
        a, b = st.columns(2)
        a.metric("Paper equity", f"{paper['equity']:,.2f} USDT")
        b.metric("Thay đổi từ vốn đầu", f"{paper['equity'] / paper['initial_capital'] - 1:+.2%}")
        st.caption(f"Cập nhật: {paper['updated_at']}")
        assets = pd.DataFrame.from_dict(paper["assets"], orient="index")
        st.dataframe(assets.drop(columns=["pending"], errors="ignore"), width="stretch")
        st.dataframe(pd.DataFrame(paper["events"]).tail(100), width="stretch", hide_index=True)
        if "monitoring" in paper:
            st.write("Theo dõi thay đổi phân phối feature · 7 ngày gần nhất")
            st.dataframe(pd.DataFrame(paper["monitoring"]["feature_drift_last_7_days"]), width="stretch", hide_index=True)
            st.json(paper["monitoring"]["prediction_quality"])
            st.caption(paper["monitoring"]["note"])
    else:
        st.info("Chưa chạy paper trading. Việc mở dashboard không tự phát sinh lệnh mô phỏng.")
with data_tab:
    st.json(quality)
    if not bars.empty:
        import duckdb
        with duckdb.connect() as conn:
            table = conn.execute("SELECT symbol, count(*) AS candles, min(timestamp) AS first_candle, max(timestamp) AS last_candle FROM read_parquet(?) GROUP BY symbol", [str(ROOT / "data/processed/bars.parquet")]).df()
        st.dataframe(table, width="stretch", hide_index=True)
    st.write("Quy tắc: kiểm tra checksum ZIP, OHLC hợp lệ, timestamp UTC, phát hiện thiếu nến và payload trùng xung đột.")
    st.caption("Dataset sử dụng dữ liệu thị trường thật. Dữ liệu tạo trong tests/ chỉ dùng để kiểm thử logic.")
