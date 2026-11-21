"""Interval view alongside the existing point-forecast model."""

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import write_json
from .features import build_features
from .risk_reference import (
    empirical_interval,
    save_forward_interval,
    forward_interval_scores,
)
from . import terminal


def panel(root, project, symbol, chart):
    st.subheader("Khoảng tham chiếu return 4 giờ")
    try:
        frame = build_features(
            pd.read_parquet(Path(root) / "data/processed/bars.parquet"),
            terminal.config(root),
        )
        reference = empirical_interval(frame, symbol)
    except ValueError as error:
        st.info(str(error))
        return
    save_forward_interval(root, reference)
    a, b, c = st.columns(3)
    a.metric("Phân vị 10%", f"{reference['q10']:+.2%}")
    b.metric("Trung vị lịch sử", f"{reference['q50']:+.2%}")
    c.metric("Phân vị 90%", f"{reference['q90']:+.2%}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[reference["q10"], reference["q90"]],
            y=[0, 0],
            mode="lines+markers",
            line={"width": 12, "color": "#38bdf8"},
            name="Khoảng tham chiếu 10–90%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[reference["q50"]],
            y=[0],
            mode="markers",
            marker={"size": 13, "color": "#fbbf24"},
            name="Trung vị",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="Return 4h", tickformat=".1%")
    chart(fig, 190)
    st.caption(
        f"{reference['history_rows']} nhãn đã đủ thời gian, cửa sổ 28 ngày; "
        "chuẩn hóa theo biến động quá khứ. Khoảng tham chiếu không bảo đảm coverage tương lai."
    )
    age = (
        pd.Timestamp.now(tz="UTC") - pd.Timestamp(reference["decision_time"])
    ).total_seconds()
    if not 0 <= age <= 3720:
        st.caption("Dữ liệu đã lưu; khoảng này không được ghi thành dự báo mới.")
    scores = forward_interval_scores(root, frame)
    if not scores.empty:
        st.dataframe(scores, hide_index=True, width="stretch")
        st.caption("Coverage chỉ tính từ các khoảng đã ghi trước khi nhãn xuất hiện.")
    with st.expander("Mô hình dự báo mới"):
        st.caption(
            "Các model thử nghiệm trả thêm khoảng phân vị; không thay model đang chạy paper."
        )
        name = st.selectbox(
            "Model thử nghiệm", ["Timestral-5M", "Timestral-28M", "TiRex-2"]
        )
        folder = Path(project) / "artifacts/forecast" / name
        if not (folder / "model.safetensors").exists():
            st.info(
                "Chưa có weights local. Xem mục model tùy chọn trong docs/VAN_HANH_VI.md để chuẩn bị Timestral-5M."
            )
        elif st.button("Tính dự báo thử nghiệm"):
            try:
                from .modern_models import predict_latest_distribution

                with st.spinner("Đang tính trên CPU…"):
                    records = predict_latest_distribution(frame, folder, name)
                for record in records:
                    save_forward_interval(root, record)
                write_json(Path(root) / "latest_experimental_interval.json", records)
            except (ImportError, OSError, ValueError, RuntimeError) as error:
                st.error(str(error))
        path = Path(root) / "latest_experimental_interval.json"
        if path.exists():
            records = json.loads(path.read_text())
            selected = [
                r for r in records if r["symbol"] == symbol and r["source"] == name
            ]
            if selected:
                r = selected[0]
                st.write(
                    f"Trung vị: **{r['q50']:+.2%}** · khoảng 10–90%: **{r['q10']:+.2%} → {r['q90']:+.2%}**"
                )
                st.caption(
                    "Nến dự báo: "
                    + r["decision_time"]
                    + ". Chỉ dùng để theo dõi nghiên cứu."
                )
                if r["decision_time"] != reference["decision_time"]:
                    st.caption(
                        "Kết quả thử nghiệm cũ; bấm tính để dùng cửa sổ nến hiện tại."
                    )
