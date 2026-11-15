"""Local market terminal."""
from __future__ import annotations
from html import escape
import os
from pathlib import Path
import time

from filelock import FileLock
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from . import terminal
from . import model_ui
from .common import write_json

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("QUANTSTREAM_ROOT", PROJECT / "workspace")).resolve()
GREEN, BLUE, GOLD, RED = "#48dda6", "#8ca6ff", "#e9b778", "#fb7185"
CSS = """<style>
.stApp{background:#0b121d;color:#dbe5ef;font-family:'Segoe UI',sans-serif}
.block-container{max-width:1520px;padding:1.6rem 2.4rem 2rem}
[data-testid=stHeader]{height:0;background:transparent}[data-testid=stToolbar]{display:none}
[data-testid=stSidebar]{background:#0e1826;border-right:1px solid #253044}
h1,h2,h3{letter-spacing:-.035em;font-weight:600!important}h1{font-size:2rem!important}h2{font-size:1.25rem!important}h3{font-size:1.05rem!important}
.brand{display:flex;align-items:center;gap:11px;margin:3px 0 28px}.logo{background:#48dda6;color:#102921;width:36px;height:36px;border-radius:9px;text-align:center;font-size:24px;font-weight:750;padding-top:1px}.brand-name{font-size:20px;font-weight:700;letter-spacing:-.8px}.sub{font:10px monospace;letter-spacing:1.7px;color:#8395aa}
.top{display:flex;justify-content:space-between;border-bottom:1px solid #253044;padding:0 0 15px;margin-bottom:22px;align-items:center}.crumb{font:11px monospace;color:#879bb1}.pill{font:10px monospace;letter-spacing:.7px;border:1px solid #2b4b40;color:#82e6bb;background:#142b25;border-radius:20px;padding:5px 11px}.pill.amber{border-color:#66502f;background:#2d261b;color:#e9c18b}
.eyebrow{font:10px monospace;letter-spacing:1.8px;color:#879bb1;margin-bottom:8px}.hero{display:flex;justify-content:space-between;align-items:flex-start;margin:22px 0}.hero h1{padding:0;margin:0}.hero p{color:#8da0b4;font-size:13px;margin:5px 0}
[data-testid=stMetric]{background:linear-gradient(120deg,#132132,#111b2a);border:1px solid #29374b;border-radius:11px;padding:16px 18px;min-height:110px}[data-testid=stMetricLabel]{color:#93a7bd;font-size:12px}[data-testid=stMetricValue]{font-family:Consolas,monospace;font-size:26px!important;letter-spacing:-1px}
.forecast{background:linear-gradient(140deg,#172f2d,#132033);border:1px solid #31544b;padding:23px;border-radius:13px}.number{font:500 43px Consolas,monospace;letter-spacing:-2px;margin:8px 0}.small{font-size:12px;color:#91a8b9;line-height:1.75}.detail{display:flex;justify-content:space-between;border-bottom:1px solid #2a4046;padding:12px 0;font-size:11px}.detail span{color:#91a8b9}.detail b{color:#d6e4eb;font-weight:500}.source{font:10px monospace;color:#758aa0;border-top:1px solid #253044;padding-top:10px}
.stButton>button{border-radius:8px;border-color:#34465c;font-size:12px}.stButton>button[kind=primary]{background:#48dda6;color:#0a261b;border-color:#48dda6;font-weight:650}div[role=radiogroup]{gap:8px}div[role=radiogroup]>label{border:1px solid #2b394c;background:#121e2e;border-radius:7px;padding:5px 10px!important}[data-testid=stDataFrame]{border:1px solid #253044;border-radius:10px}footer{display:none}
@media(max-width:800px){.block-container{padding:1rem}.hero{display:block}.number{font-size:34px}}
</style>"""


@st.cache_data(show_spinner=False, ttl=300, max_entries=30)
def parquet(path, modified):
    return pd.read_parquet(path)


def frame(path):
    path = Path(path)
    return parquet(str(path), path.stat().st_mtime_ns) if path.exists() else pd.DataFrame()


def chart(fig, height=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#aabed2", family="Segoe UI, sans-serif", size=11), height=height,
                      margin=dict(l=5, r=10, t=28, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.13, font_size=10),
                      xaxis=dict(gridcolor="#203044", zeroline=False), yaxis=dict(gridcolor="#203044", zeroline=False))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def date(value):
    return pd.Timestamp(value).strftime("%d/%m %H:%M UTC") if value else "—"


def refresh():
    try:
        with st.spinner("Đồng bộ thị trường…"):
            terminal.local_step(ROOT)
        st.cache_data.clear()
        st.session_state.last_poll = time.monotonic()
    except Exception as error:
        st.error(str(error))


def main():
    st.set_page_config(page_title="QuantStream | Market terminal", page_icon="Q", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    cfg = terminal.config(ROOT)
    registry = terminal.read_json(ROOT / "artifacts/local_registry.json", {})
    meta = terminal.read_json(ROOT / "artifacts/champion.json", {})
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">Q</div><div><div class="brand-name">QuantStream</div><div class="sub">MARKET TERMINAL</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
        symbol = st.selectbox("Thị trường", cfg.symbols, format_func=lambda s: s.replace("USDT", " / USDT"))
        names = list(registry) or ["ridge"]
        selected = st.selectbox("Model", names, index=names.index(meta.get("model", names[0])),
                                format_func=lambda n: model_ui.label(registry.get(n, {"model":n})), disabled=not registry)
        if registry and selected != meta.get("model"):
            state=terminal.read_json(ROOT/"paper/state.json",{})
            different=bool(state and state.get("model_sha256")!=registry[selected]["sha256"])
            open_positions=any(a.get("qty") for a in state.get("assets",{}).values())
            if different and open_positions:
                st.warning("Còn vị thế paper. Dừng lệnh mới ở trang Danh mục và chờ vị thế thoát trước khi đổi model.")
            title="Lưu phiên và dùng model này" if different else "Dùng model này"
            if st.button(title,width="stretch",disabled=different and open_positions):
                try:
                    with FileLock(str(ROOT/"terminal.lock"),timeout=2):
                        terminal.activate(ROOT,selected,archive_empty=different)
                    terminal.local_step(ROOT,refresh=not (ROOT/"data/processed/bars.parquet").exists())
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
        st.divider()
        auto = st.toggle("Tự đồng bộ mỗi phút", value=False, disabled=not registry)
        if st.button("Đồng bộ thị trường", width="stretch", disabled=not registry):
            refresh()
        st.caption("Nến 1 giờ · dự báo lợi suất 4 giờ")
        st.divider()
        st.markdown('<span class="pill amber">PAPER MODE</span><p class="small">Dự báo và mô phỏng. Không kết nối tài khoản hay đặt lệnh tiền thật.</p>', unsafe_allow_html=True)
        st.caption("Local v1.0 · BTC / ETH")
    st.markdown('<div class="top"><span class="crumb">WORKSPACE / THỊ TRƯỜNG SỐ</span><span class="pill">● LOCAL WORKSPACE</span></div>', unsafe_allow_html=True)
    if not registry:
        st.title("Thị trường, dữ liệu, dự báo.")
        st.caption("Nạp 90 ngày nến thật, fit Ridge và XGBoost trên CPU để mở terminal.")
        if st.button("Kết nối dữ liệu và tạo model", type="primary"):
            try:
                with st.spinner("Đang chuẩn bị workspace…"):
                    terminal.initialize(ROOT)
                st.rerun()
            except Exception as error:
                st.error(str(error))
        model_ui.import_panel(ROOT,PROJECT)
        return
    if not meta:
        st.info("Model đã nạp. Chọn model ở sidebar rồi bấm Dùng model này để bắt đầu dự báo.")
        model_ui.import_panel(ROOT,PROJECT)
        return
    route = st.radio("Trang", ["Thị trường", "Dự báo", "Kiểm định", "Danh mục", "Dữ liệu"], horizontal=True, label_visibility="collapsed")
    st.session_state.setdefault("last_poll", time.monotonic())

    @st.fragment(run_every=60 if auto else None)
    def content():
        if auto and time.monotonic() - st.session_state.last_poll >= 55:
            refresh()
        current = terminal.read_json(ROOT / "artifacts/champion.json", {})
        bars = frame(ROOT / "data/processed/bars.parquet")
        latest = terminal.read_json(ROOT / "latest_forecasts.json", {}).get("forecasts", [])
        health = terminal.read_json(ROOT / "health.json", {})
        if health.get("status") == "error":
            st.warning("Lần đồng bộ gần nhất gặp lỗi: " + health["error"])
        if bars.empty:
            st.error("Không tìm thấy nến đã lưu. Cần nạp lại dữ liệu workspace.")
            return
        if route == "Thị trường":
            market(bars, symbol, latest, current)
        elif route == "Dự báo":
            predictions(symbol, current)
        elif route == "Kiểm định":
            research(current)
        elif route == "Danh mục":
            portfolio(cfg, current)
        else:
            data_page(bars, current)
    content()


def market(bars, symbol, forecasts, meta):
    asset = bars[bars.symbol.eq(symbol)].sort_values("timestamp")
    last = asset.iloc[-1]
    forecast = next((r for r in forecasts if r["symbol"] == symbol and r.get("model_sha256") == meta.get("sha256")), {})
    quote = terminal.read_json(ROOT / "quotes.json", {}).get("quotes", {}).get(symbol, {})
    if quote and pd.Timestamp.now(tz="UTC") - pd.Timestamp(quote["received_at"]) > pd.Timedelta(minutes=2):
        st.warning("Giá ticker đã quá 2 phút. Bấm Đồng bộ thị trường để cập nhật.")
    fresh = 0 <= (pd.Timestamp.now(tz="UTC") - last.timestamp - pd.Timedelta(hours=1)).total_seconds() < 3720
    label = "DỮ LIỆU MỚI" if fresh else "DỮ LIỆU ĐÃ LƯU"
    st.markdown(f'<div class="hero"><div><div class="eyebrow">MARKET OVERVIEW</div><h1>{escape(symbol.replace("USDT"," / USDT"))}</h1><p>Nến đã đóng, dòng tiền và model local.</p></div><span class="pill">● {label}</span></div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    change = quote.get("priceChangePercent", (last.close/asset.iloc[-25].close-1)*100)
    a.metric("Giá gần nhất · USDT" if quote else "Đóng nến · USDT", f"{quote.get('lastPrice',last.close):,.2f}", f"{change:+.2f}% / 24h")
    b.metric("Khối lượng 24h · USDT", f"{quote.get('quoteVolume',asset.tail(24).quote_volume.sum())/1e6:,.1f}M")
    c.metric("Model đang dùng", model_ui.label(meta))
    d.metric("Nến đóng gần nhất", (last.timestamp+pd.Timedelta(hours=1)).strftime("%H:%M UTC"))
    st.write("")
    left,right = st.columns([3,1.1], gap="large")
    with left:
        a,b = st.columns([3,1])
        a.subheader("Giá và khối lượng")
        period = b.selectbox("Khoảng", ["48 giờ","7 ngày","30 ngày"], index=1,label_visibility="collapsed")
        view = asset.tail({"48 giờ":48,"7 ngày":168,"30 ngày":720}[period])
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.025,row_heights=[.8,.2])
        fig.add_trace(go.Candlestick(x=view.timestamp,open=view.open,high=view.high,low=view.low,close=view.close,
                                    increasing_line_color=GREEN,decreasing_line_color=RED,showlegend=False),row=1,col=1)
        fig.add_trace(go.Bar(x=view.timestamp,y=view.volume,marker_color=[GREEN if c>=o else RED for c,o in zip(view.close,view.open)],opacity=.4,showlegend=False),row=2,col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_yaxes(title="USDT",row=1,col=1)
        chart(fig,460)
        st.markdown(f'<div class="source">BINANCE SPOT · 1H · {len(asset):,} NẾN · {escape(date(quote.get("received_at")))}</div>',unsafe_allow_html=True)
    with right:
        value=forecast.get("prediction")
        text=f"{value:+.2%}" if value is not None else "—"
        color=GREEN if value is not None and value>=0 else RED
        details="".join(f'<div class="detail"><span>{label}</span><b>{escape(date(forecast.get(key)))}</b></div>' for label,key in [("Nến tín hiệu","decision_time"),("Vào giả định","entry_time"),("Hết khung dự báo","horizon_end")])
        st.markdown(f'<div class="forecast"><div class="eyebrow">MODEL FORECAST · 4H</div><div class="number" style="color:{color}">{text}</div><p class="small">Lợi suất dự báo, không phải lệnh mua/bán.</p>{details}<p class="small" style="margin-top:16px">Khung dự báo 4 giờ, chờ vào 1 giờ theo cấu hình kiểm định.</p></div>',unsafe_allow_html=True)
        st.caption(f"Model fit: {date(meta['created_at'])}")
        from .forecast_contract import allows_paper_entries
        if not allows_paper_entries(meta):
            st.caption("Chưa đạt điều kiện chấp nhận · dự báo để theo dõi nghiên cứu.")
        validation=meta.get("validation") or {}
        ratio=validation.get("mse_ratio_to_zero")
        if ratio is not None:
            st.caption("Validation MSE / zero return: " + f"{ratio:.4f}" + (" · chưa vượt baseline" if ratio>=1 else " · thấp hơn baseline"))
        else:
            st.caption("Checkpoint đã nhập. Xem nguồn và loss lịch sử ở trang Kiểm định.")


def predictions(symbol,meta):
    st.subheader("Dự báo đã ghi")
    st.caption("Chỉ ghi khi nến còn mới. Không dùng model vừa fit để dựng dự báo quá khứ.")
    hist=terminal.forecast_history(ROOT)
    if not hist.empty:
        hist=hist[hist.symbol.eq(symbol)&hist.model_sha256.eq(meta["sha256"])].sort_values("decision_time")
    if hist.empty:
        st.info("Chưa có dự báo đủ mới cho model này.")
    else:
        outcomes = terminal.build_features(frame(ROOT / "data/processed/bars.parquet"), terminal.config(ROOT))
        outcomes = outcomes[["symbol", "decision_time", "target_return"]]
        hist = hist.merge(outcomes, on=["symbol", "decision_time"], how="left", validate="many_to_one")
        mature = hist.dropna(subset=["target_return"])
        if not mature.empty:
            st.metric("MAE forecast đã đủ nhãn", f"{(mature.prediction-mature.target_return).abs().mean():.3%}")
            st.caption(f"{len(mature)} cửa sổ đã đủ thời gian; mẫu ít chưa kết luận hiệu quả đầu tư.")
        fig=px.scatter(hist,render_mode="svg",x="decision_time",y="prediction",color_discrete_sequence=[GREEN])
        limit = max(.001, float(hist.prediction.abs().max()) * 1.2)
        fig.update_yaxes(range=[-limit, limit])
        if len(hist) == 1:
            point = hist.decision_time.iloc[0]
            fig.update_xaxes(range=[point-pd.Timedelta(hours=1), point+pd.Timedelta(hours=1)])
        fig.update_yaxes(title="Dự báo return 4h",tickformat=".2%")
        chart(fig,280)
        st.dataframe(hist[["symbol","decision_time","prediction","target_return","model","recorded_at"]],width="stretch",hide_index=True)
        st.download_button("Tải lịch sử dự báo CSV",hist.to_csv(index=False).encode(),"forecasts.csv","text/csv")
    st.divider()
    from .risk_ui import panel as interval_panel
    interval_panel(ROOT, PROJECT, symbol, chart)
    st.divider()
    if meta.get("hyperparameters",{}).get("family") == "zero_return":
        st.info("Baseline return bằng 0 là mốc đối chiếu, không có feature importance hoặc tín hiệu đã học.")
        return
    if meta.get("type")=="tcn" or meta.get("hyperparameters",{}).get("family") in ("tcn","gru"):
        try:
            model_ui.neural_inputs(ROOT,meta,symbol,chart)
        except Exception as error:
            st.error(str(error))
        return
    st.subheader("Feature và dự báo")
    values=terminal.feature_contributions(ROOT,symbol).head(10).sort_values("value")
    chart(go.Figure(go.Bar(x=values.value,y=values.feature,orientation="h",marker_color=[GREEN if x>=0 else GOLD for x in values.value])),340)
    st.caption("Ridge: hệ số × feature đã chuẩn hóa. XGBoost: độ quan trọng tổng quát; không phải giải thích nhân quả.")
    if meta.get("residual_abs_q80") is not None:
        st.caption(f"80% sai số OOF nằm trong ±{meta['residual_abs_q80']:.2%}; không bảo đảm coverage tương lai.")


def research(meta):
    if meta.get("type") in ("tcn","portable"):
        model_ui.history(ROOT,meta,chart)
        return
    run=ROOT/meta["run"]
    st.subheader("Kiểm định theo thời gian")
    st.caption("Ba fold mở rộng, purge nhãn chồng lấn. Validation đã dùng so model, chưa phải test đầu tư độc lập.")
    table=pd.DataFrame(terminal.read_json(run/"validation_metrics.json",[]))
    a,b,c=st.columns(3)
    row=table[table.model.eq(meta["model"])].iloc[0]
    a.metric("MAE validation",f"{row.mae:.3%}")
    b.metric("MSE / zero return",f"{row.mse_ratio_to_zero:.4f}")
    c.metric("Đúng hướng",f"{row.directional_accuracy:.1%}")
    a,b=st.columns([1.1,1])
    with a:
        fig=px.bar(table,x="model",y="mae",color="model",color_discrete_sequence=[GREEN,BLUE,"#8494aa",GOLD])
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickformat=".2%",title="MAE · thấp hơn tốt hơn")
        chart(fig,290)
    b.dataframe(table,width="stretch",hide_index=True)
    oof=frame(run/"validation_predictions.parquet")
    values=oof[oof.model.eq(meta["model"])]
    a,b=st.columns(2)
    with a:
        fig=px.scatter(values,render_mode="svg",x="target_return",y="prediction",color="symbol",opacity=.4,color_discrete_sequence=[GREEN,BLUE])
        fig.update_xaxes(title="Return thực tế",tickformat=".1%")
        fig.update_yaxes(title="Return dự báo",tickformat=".1%")
        chart(fig,300)
    with b:
        chart(px.histogram(values.assign(error=values.prediction-values.target_return),x="error",color="symbol",nbins=45,barmode="overlay",opacity=.6,color_discrete_sequence=[GREEN,BLUE]),300)
    curve=frame(run/(meta["model"]+"_equity.parquet"))
    if not curve.empty:
        st.subheader("Mô phỏng trên validation")
        chart(px.line(curve,render_mode="svg",x="timestamp",y=["model","buy_hold","ema"],color_discrete_sequence=[GREEN,BLUE,GOLD]))
        report=terminal.read_json(run/(meta["model"]+"_backtest.json"),{})
        st.dataframe(pd.DataFrame([{"strategy":k,**v} for k,v in report.items()]).drop(columns="note",errors="ignore"),width="stretch",hide_index=True)
        st.caption("Cùng phí/slippage giả định. Đây không phải PnL tiền thật; số lệnh và exposure cần đọc cùng lợi nhuận.")
    with st.expander("Artifact và cấu hình"):
        st.json({k:meta.get(k) for k in ("model","sha256","training_rows","created_at","fit_seconds","sklearn_version","deployment_status")})
        st.code(", ".join(meta["features"]),language=None)


def portfolio(cfg,meta):
    from .forecast_contract import allows_paper_entries
    accepted = allows_paper_entries(meta)
    st.subheader("Danh mục mô phỏng")
    if not accepted:
        st.info("Chưa có model đạt điều kiện validation. Tiếp tục theo dõi dự báo; tạm dừng nhận lệnh mới.")
    st.caption("10.000 USDT mô phỏng. App/worker dừng thì mô phỏng không tiếp tục chạy.")
    controls=terminal.read_json(ROOT/"controls.json",{"paper_enabled":False})
    a,b=st.columns([1,3])
    if a.button("Bắt đầu mô phỏng",disabled=controls["paper_enabled"] or not accepted,type="primary"):
        write_json(ROOT/"controls.json",{"paper_enabled":True})
        refresh()
        st.rerun()
    if b.button("Dừng nhận lệnh mới",disabled=not controls["paper_enabled"]):
        write_json(ROOT/"controls.json",{"paper_enabled":False})
        refresh()
        st.rerun()
    state=terminal.read_json(ROOT/"paper/state.json",{})
    st.caption(f"Vốn mỗi lệnh: {cfg.position_fraction:.0%} mỗi tài sản · phí/slippage {cfg.fee_bps+cfg.slippage_bps:g} bps mỗi chiều · ngưỡng {meta['threshold']:.2%}")
    if not state:
        st.info("Chưa bắt đầu phiên paper. Không có lệnh hoặc PnL tạo sẵn.")
        return
    a,b,c=st.columns(3)
    a.metric("Equity mô phỏng",f"{state['equity']:,.2f} USDT")
    b.metric("Từ vốn đầu",f"{state['equity']/state['initial_capital']-1:+.2%}")
    c.metric("Lệnh đã khớp",str(sum(e["side"] in ("BUY","SELL") for e in state["events"])))
    chart(px.line(pd.DataFrame(state["equity_history"]),render_mode="svg",x="time",y="equity",color_discrete_sequence=[GREEN]),280)
    st.dataframe(pd.DataFrame.from_dict(state["assets"],orient="index").drop(columns="pending",errors="ignore"),width="stretch")
    events=pd.DataFrame(state["events"])
    st.dataframe(events.tail(100),width="stretch",hide_index=True)
    if not events.empty:
        st.download_button("Tải sổ lệnh CSV",events.to_csv(index=False).encode(),"paper_orders.csv","text/csv")
    with st.expander("Lưu phiên paper"):
        st.caption("Giữ lại phiên cũ. Cần chờ vị thế mở thoát trước khi đổi model.")
        if st.button("Lưu phiên và dừng mô phỏng"):
            try:
                terminal.archive_paper(ROOT)
                st.rerun()
            except Exception as error:
                st.error(str(error))


def data_page(bars,meta):
    st.subheader("Nguồn dữ liệu và vận hành")
    st.caption("Binance Spot public REST · UTC · chỉ nến đã đóng")
    st.dataframe(bars.groupby("symbol").agg(candles=("timestamp","size"),first=("timestamp","min"),last=("timestamp","max")),width="stretch")
    quality=terminal.read_json(ROOT/"data/processed/live_quality.json",{})
    a,b,c=st.columns(3)
    a.metric("Thiếu nến",str(sum(r["missing_bars"] for r in quality.get("gaps",[]))))
    b.metric("Nến loại bỏ",str(quality.get("quarantine_rows",0)))
    c.metric("Model trên đĩa",f"{meta['artifact_bytes']/1024:.1f} KB")
    with st.expander("Nguồn và health log"):
        st.json(terminal.read_json(ROOT/"data/processed/source_manifest.json",{}))
        st.json(terminal.read_json(ROOT/"health.json",{}))
    st.download_button("Tải nến CSV",bars.to_csv(index=False).encode(),"candles.csv","text/csv")
    st.divider()
    st.subheader("Fit lại model local")
    st.caption("Dùng nến đã lưu và tham số cố định; cần lưu phiên paper cũ trước.")
    state=terminal.read_json(ROOT/"paper/state.json",{})
    if state:
        st.info("Đang có phiên paper. Lưu phiên ở trang Danh mục trước khi fit lại; nạp checkpoint không cần fit lại.")
    if st.button("Fit lại Ridge và XGBoost",disabled=bool(state)):
        try:
            with FileLock(str(ROOT/"terminal.lock"),timeout=2),st.spinner("Đang fit trên CPU…"):
                terminal.fit_local(ROOT)
            terminal.local_step(ROOT,refresh=False)
            st.rerun()
        except Exception as error:
            st.error(str(error))

    st.divider()
    model_ui.import_panel(ROOT,PROJECT)
