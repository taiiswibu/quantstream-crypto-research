# Third-party sources

- FraudGuard by ItsLewis-DE: architectural inspiration only, at commit `5144d6d63dc38b4af0e0a68fd8638f369d35a7a4`. No source code from that repository is bundled.
- Binance Spot market data: downloaded from public Binance archives/API. Archive URLs and checksums are in `data/processed/source_manifest.json`. Dataset use is governed by the provider's terms: https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md .
- Chronos-2 by Amazon: original model/library licensed Apache-2.0; source https://github.com/amazon-science/chronos-forecasting ; model https://huggingface.co/amazon/chronos-2 . The bundle includes adapters trained by this project, not the pretrained base weights. The adapter must be used with the base model and its license. Recorded base-model revision and provenance are in each `chronos_report.json`.
- PyTorch, scikit-learn, XGBoost, pandas, NumPy, PyArrow, DuckDB, Streamlit, Plotly, Transformers, PEFT and other dependencies retain their respective licenses. Install them through package managers; their distributions are not bundled.

Original QuantStream application code in this package is licensed under MIT; this does not relicense market data or pretrained model artifacts.
