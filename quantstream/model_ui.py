"""Imported checkpoints and their recorded training evidence."""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .tcn_import import (
    historical_report,
    import_bundle,
    import_checkpoint,
    normalized_window,
)


def label(metadata):
    return metadata.get("origin", {}).get("label") or {
        "ridge": "Ridge · CPU",
        "xgboost_d3": "XGBoost · CPU",
    }.get(metadata.get("model"), metadata.get("model", "Model"))


def import_panel(root, project):
    st.subheader("Nạp model đã train")
    st.caption(
        "TCN: model.pt + tcn_metadata.json. Notebook 09/10/11: thư mục model bundle đã giải nén. Chỉ nạp, không train lại."
    )
    folder = st.text_input(
        "Thư mục checkpoint", value=str(Path(project) / "artifacts/tcn")
    )
    if "model_imported" in st.session_state:
        st.success(st.session_state.pop("model_imported"))
    if st.button("Kiểm tra và nạp model"):
        try:
            with st.spinner("Đối chiếu SHA và cấu hình…"):
                if (Path(folder) / "bundle_manifest.json").exists():
                    metadata = import_bundle(root, folder)
                else:
                    metadata = import_checkpoint(root, folder)
            st.session_state["model_imported"] = (
                label(metadata)
                + " · đã thêm vào danh sách. Chọn model ở sidebar để sử dụng."
            )
            st.rerun()
        except Exception as error:
            st.error(str(error))
    st.caption(
        "TCN cần PyTorch CPU. Xem README.md và nạp checkpoint ở trang Dữ liệu."
    )


def history(root, metadata, chart):
    original = historical_report(root, metadata)
    st.subheader("Checkpoint và quá trình học")
    st.info(label(metadata) + " · " + metadata.get("origin", {}).get("status", ""))
    a, b, c = st.columns(3)
    a.metric("Train đã ghi", str(metadata.get("training_rows", "—")) + " dòng")
    b.metric("Checkpoint", metadata["sha256"][:12])
    c.metric("Bước train khi import", "0")
    st.caption(
        "Báo cáo lịch sử đi kèm checkpoint. Không dùng checkpoint đã refit toàn development để dựng lại OOF trên các fold cũ."
    )
    if metadata.get("training_last_label"):
        st.caption("Nhãn train cuối (UTC): " + str(metadata["training_last_label"]))
    if metadata.get("artifact_role") == "all_observed_research_refit_no_independent_test":
        st.caption("Checkpoint này đã refit toàn bộ nhãn quan sát. Chưa có test độc lập cho chính các weights này.")
    if metadata["type"] == "tcn":
        loss = []
        for fold in original.get("history", []):
            for row in fold["epochs"]:
                loss.append({"fold": str(fold["fold"] + 1), **row})
        if loss:
            data = pd.DataFrame(loss)
            a, b = st.columns(2)
            with a:
                chart(
                    px.line(
                        data,
                        x="epoch",
                        y="train_huber",
                        color="fold",
                        markers=True,
                        render_mode="svg",
                    ).update_layout(title="Train Huber · nhãn × 100"),
                    300,
                )
            if "validation_mse" in data:
                with b:
                    chart(
                        px.line(
                            data,
                            x="epoch",
                            y="validation_mse",
                            color="fold",
                            markers=True,
                            render_mode="svg",
                        ).update_layout(title="Validation MSE lịch sử"),
                        300,
                    )
        selection = original.get("selection", {})
        if selection.get("folds"):
            st.dataframe(
                pd.DataFrame(selection["folds"]).drop(columns="note", errors="ignore"),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "Epoch của bản TCN cũ được chọn trên validation; không phải một test đầu tư độc lập."
        )
        params = {
            "family": "tcn",
            "width": 24,
            "lookback": original["config"]["lookback"],
            "activation": "GELU",
            "dilations": [1, 2, 4, 8],
        }
    else:
        params = original["hyperparameters"]
        records = []
        if original.get("validation"):
            records.append(
                {"stage": "Validation dùng lựa chọn", **original["validation"]}
            )
        if original.get("test"):
            records.append(
                {"stage": "Kiểm tra hồi cứu · đã từng quan sát", **original["test"]}
            )
        if records:
            st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
        st.caption(
            "Block lịch sử này là kiểm tra hồi cứu. Chưa phải dữ liệu tương lai độc lập."
        )
        gate = original.get("validation_gate")
        if gate is not None:
            st.dataframe(pd.DataFrame([gate]), width="stretch", hide_index=True)
            if not gate["beats_zero_on_mean_and_every_fold"]:
                st.warning(
                    "Model chưa vượt baseline return bằng 0 trên trung bình và từng fold validation."
                )
        folder = Path(root) / metadata["source_metadata_path"]
        curves = []
        for path in folder.parent.glob("history_*.csv"):
            curves.append(pd.read_csv(path).assign(stage=path.stem))
        if curves:
            data = pd.concat(curves, ignore_index=True)
            chart(
                px.line(
                    data,
                    x="epoch",
                    y="train_loss_scaled"
                    if "train_loss_scaled" in data
                    else "train_huber_scaled",
                    color="stage",
                    markers=True,
                    render_mode="svg",
                ),
                300,
            )
            if "inner_validation_mse" in data:
                chart(
                    px.line(
                        data.dropna(subset=["inner_validation_mse"]),
                        x="epoch",
                        y="inner_validation_mse",
                        color="stage",
                        markers=True,
                        render_mode="svg",
                    ),
                    300,
                )
    with st.expander("Kiến trúc, siêu tham số và nguồn"):
        st.json(params)
        st.json(
            {
                key: metadata.get(key)
                for key in [
                    "created_at",
                    "imported_at",
                    "source_control",
                    "training_features_sha256",
                    "parity_max_abs_delta",
                    "deployment_status",
                ]
            }
        )
        st.code(", ".join(metadata["features"]), language=None)
    train_config = metadata.get("training_config", {})
    risk = ["position_fraction", "drawdown_halt", "fee_bps", "slippage_bps"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "config": "Báo cáo train gốc",
                    **{k: train_config.get(k) for k in risk},
                },
                {
                    "config": "Paper workspace hiện tại",
                    **{k: metadata["config"].get(k) for k in risk},
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Paper dùng risk của workspace; các con số backtest lịch sử dùng cấu hình train được ghi riêng."
    )


def neural_inputs(root, metadata, symbol, chart):
    data = normalized_window(root, metadata, symbol)
    st.subheader("Cửa sổ đầu vào model")
    chart(
        go.Figure(
            go.Heatmap(
                z=data.to_numpy().T,
                x=data.index,
                y=data.columns,
                colorscale="RdBu",
                zmid=0,
                colorbar_title="z-score",
            )
        ),
        380,
    )
    st.caption(
        "Feature chuẩn hóa bằng scaler đã lưu trong checkpoint. Đây là đầu vào, không phải feature importance hay xác suất dự báo đúng."
    )
