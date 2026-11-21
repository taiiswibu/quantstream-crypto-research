# Cài đặt và vận hành QuantStream

## Môi trường

App core chạy CPU với Python 3.12 64-bit. Dữ liệu lấy từ Binance public market-data API, không cần tài khoản sàn hay API key. GPU chỉ cần cho các thí nghiệm deep learning lớn; app Ridge/XGBoost không phụ thuộc CUDA.

Các lệnh bên dưới chạy tại thư mục có `pyproject.toml` và `app.py`. Dùng trực tiếp Python trong môi trường ảo để tránh nhầm Python của Conda hoặc hệ thống.

### Windows PowerShell

Nếu chưa có `.venv`:

```powershell
py -3.12 -m venv .venv
```

Cài dependency theo constraints đã ghi nhận:

```powershell
.\.venv\Scripts\python.exe -m pip install -e . -c requirements-tested-core.txt
```

Khởi tạo workspace mới, chỉ khi chưa có model local:

```powershell
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace setup --days 90
```

Nếu đã có `workspace/artifacts/local_registry.json`, bỏ qua bước setup. Lệnh setup sẽ từ chối khi đã có model, tránh fit đè ngoài ý muốn.

Mở app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Truy cập [http://127.0.0.1:8501/](http://127.0.0.1:8501/). Giữ tiến trình đang chạy. `Ctrl+C` dừng app; dữ liệu đã lưu vẫn còn.

### Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . -c requirements-tested-core.txt
.venv/bin/python -m quantstream.terminal --root workspace setup --days 90
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Lệnh setup cũng chỉ dùng cho workspace mới. Khi cập nhật code, giữ các thay đổi local và sao lưu workspace trước khi thay dependency hoặc model.

## Dữ liệu và worker

Đồng bộ một lượt:

```powershell
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace refresh
```

Worker đồng bộ liên tục:

```powershell
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace watch
```

Dùng một worker. App/worker dùng chung file lock. Laptop ngủ, tiến trình dừng hoặc mất mạng thì không có vận hành liên tục. Lệnh vào trễ bị kiểm tra theo quy tắc runtime, không được ghi lại thành một lần khớp đúng giờ.

Trong app, **Đồng bộ thị trường** nạp nến/ticker mới; **Tự đồng bộ mỗi phút** chỉ chạy khi phiên app còn hoạt động. Nạp thêm nến không tự cập nhật weights của model.

Gói lịch sử có `bars.parquet` và `history_manifest.json`. Nạp qua trang Dữ liệu hoặc CLI:

```powershell
.\.venv\Scripts\python.exe -m quantstream.history_import --root workspace --archive "C:\du-lieu\QuantStream_Training_History.zip"
```

App kiểm tra SHA, symbol, nến đóng, OHLCV và xung đột khóa. Một nến trùng khóa nhưng khác giá/volume làm thao tác dừng trước khi ghi. Feature được khởi động lại tại gap, không nội suy nến để che khoảng trống.

## Model và phiên paper

Fit lại Ridge/XGBoost từ nến đã lưu qua trang **Dữ liệu**. Ba fold đánh giá nằm trong 90 ngày gần nhất; mỗi fold được train bằng toàn bộ nhãn lịch sử hợp lệ trước ranh giới của nó. Sau CV, model phục vụ được refit trên các nhãn đã biết.

Nếu có phiên paper, dừng nhận lệnh mới, chờ vị thế đóng rồi dùng **Lưu phiên và dừng mô phỏng**. Không xóa state để bỏ guard hoặc giấu drawdown.

Lưu phiên bằng CLI khi không còn vị thế mở:

```powershell
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace archive-paper
```

Nạp bundle do Notebook 09/10/11 xuất, sau khi giải nén:

```powershell
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace import-model --folder "artifacts\run008"
```

Thư mục bundle phải có manifest, metadata và weights theo đúng định dạng. Import đối chiếu hash/config và parity; import không có bước optimizer và không phải lượt train mới. Chọn model ở sidebar sau khi nạp.

TCN cần PyTorch CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dl]" -c requirements-tested-core.txt
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace import-tcn --folder "artifacts\tcn"
```

Chỉ nạp artifact có nguồn tin cậy. `joblib` chứa đối tượng Python; không dùng weights không rõ nguồn. Dự án không kết nối đặt lệnh tiền thật.

## Notebook và ảnh minh chứng

Cài phần notebook/dev:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook]" -c requirements-tested-core.txt
```

Notebook 13 cần snapshot với hai file `bars.parquet` và `history_manifest.json` trong `research_input/`. Có thể chỉ định thư mục khác:

```powershell
$env:QUANTSTREAM_HISTORY_DIR = "D:\du-lieu\quantstream-history"
```

Snapshot trong lượt minh chứng có SHA256 `1bc980d79b7a5a9c689883aa8d065e67284d7850d605c353dc2ab040ffb1b653`. Nếu dùng dữ liệu khác, cần manifest khớp dữ liệu của mình và ghi kết quả như một lượt riêng; không coi đó là tái lập đúng bảng đã công bố.

Mở [Notebook 13](../notebooks/13_Project_Evidence_VI.ipynb) và chọn kernel của `.venv`. Notebook kiểm tra input, train CPU, lưu artifacts vào `notebook_work/portfolio_cpu`, và xuất bảng vào `reports/portfolio_cpu`. Không dùng workspace paper của app. Cell sơ đồ cần Graphviz với lệnh `dot` trong PATH.

Notebook 08 chỉ đọc artifacts để review. Để đọc lượt Notebook 13 thay vì workspace app:

```powershell
$env:QUANTSTREAM_REVIEW_ROOT = "$PWD\notebook_work\portfolio_cpu"
```

Xuất nguyên byte ảnh đã lưu trong notebook:

```powershell
.\.venv\Scripts\python.exe scripts/export_notebook_images.py notebooks/13_Project_Evidence_VI.ipynb --output reports/portfolio_cpu/figures
```

Manifest ảnh ghi notebook hash, cell index, execution count và image hash. Các ảnh này là output tính toán/diagram trong notebook, không phải ảnh chụp giao diện live trên máy Windows.

## Notebook Kaggle và môi trường tùy chọn

Notebook 02–07 và 09–12 giữ quy trình nghiên cứu riêng. Chọn đúng dataset, source SHA và snapshot manifest của từng run; không bỏ kiểm tra SHA hoặc tự thay phiên bản package để chạy qua lỗi. Các run đã xem kết quả không trở lại thành test độc lập khi chạy lại.

Với thí nghiệm Chronos/PEFT, dùng môi trường riêng theo protocol từng run. Sự cố torchao/NumPy đã có ghi nhận trong [hồ sơ thí nghiệm](EXPERIMENTS.md) và các file `docs/experiments`. Preflight smoke không tính là research fit.

Model foundation thử nghiệm là extension riêng. Đọc [phương pháp và điều kiện sử dụng](METHODS_2026_VI.md) trước khi cài. Nếu cần chuẩn bị Timestral-5M theo source hiện có:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[forecast]"
.\.venv\Scripts\python.exe -m quantstream.modern_models --model Timestral-5M --folder artifacts/forecast/Timestral-5M
```

Extension không thay champion paper. Weights có license riêng; không gộp quyền sử dụng của chúng vào MIT của ứng dụng.

## Sao lưu và xử lý lỗi

| Hiện tượng | Kiểm tra |
|---|---|
| Port 8501 đang dùng | Dừng phiên cũ hoặc chọn port khác bằng `--server.port` |
| Không tải được nến | Đọc HTTP error/health log; giữ dữ liệu đã lưu, không chuyển sang data giả |
| Model/hash không khớp | Dùng đúng bundle; không sửa metadata để bỏ kiểm tra |
| Bị chặn refit vì paper | Lưu phiên đúng quy trình, chờ đóng vị thế |
| Không thấy notebook chart | Chọn đúng kernel và mở output đã lưu; kiểm tra runtime error trước khi xuất ảnh |
| Metric khác README | So snapshot, horizon, config, cohort và phiên bản dependency trước khi kết luận |

Tắt app/worker rồi sao lưu toàn bộ `workspace`. Thư mục này chứa nến, registry, model, SQLite forecast, báo cáo validation và paper state. Các dữ liệu cá nhân/vận hành đó không được commit lên Git.


## Chụp ảnh giao diện làm minh chứng

### Chuẩn bị một phiên chụp

1. Mở app tại `http://127.0.0.1:8501/` và phóng to cửa sổ trình duyệt. Dùng một mức zoom cố định, bắt đầu từ 100%; nếu nhãn bị cắt thì giảm về 90% trước khi chụp cả bộ.
2. Chọn cùng một tài sản và một model cho bộ ảnh chính. Bấm **Đồng bộ thị trường** một lượt, chờ biểu đồ render rồi chụp. Nếu đồng bộ lỗi hoặc dữ liệu cũ, giữ trạng thái đó khi mô tả; không thay ảnh hoặc sửa thời gian để trông như dữ liệu mới.
3. Trên Windows, nhấn **Win + Shift + S**, chọn vùng app gồm sidebar và nội dung chính, rồi lưu **PNG**. Với trang dài, chụp hai khung thay vì thu nhỏ đến mức không đọc được chữ.
4. Giữ nguyên tên model/checkpoint, tài sản, mốc UTC, metric và đơn vị trục. Có thể cắt tab trình duyệt, taskbar hoặc cửa sổ chat nằm ngoài app. Không sửa số liệu, xóa cảnh báo liên quan đến chất lượng hoặc tạo lệnh giả để ảnh đẹp hơn.
5. Tạo thư mục `reports/app/` rồi lưu theo tên ở bảng dưới. Ảnh app dùng làm bằng chứng trạng thái tại thời điểm chụp, không thay cho file metrics hay validation.

| Tên PNG | Trang và khung cần chụp | Bằng chứng cần đọc được |
|---|---|---|
| `01_thi_truong.png` | Thị trường: sidebar, KPI, nến/volume và forecast | BTC/ETH, model, thời điểm nến, đơn vị giá/volume, horizon |
| `02_du_bao.png` | Dự báo: lịch sử forecast và phần feature | Forecast đã ghi, thời điểm, model và ghi chú sai số/OOF |
| `03_kiem_dinh_baseline.png` | Kiểm định: KPI và bảng so model/baseline | MAE, MSE/zero-return, DA, tên baseline và phạm vi validation |
| `04_kiem_dinh_fold.png` | Kiểm định: bảng từng fold và backtest | Fold ratios, kỳ dữ liệu, equity và số lệnh/drawdown nếu có |
| `05_danh_muc.png` | Danh mục: equity, lệnh và cấu hình chi phí | PAPER MODE, vốn giả lập, phí/slippage, số lệnh thật đang được ghi |
| `06_du_lieu.png` | Dữ liệu: nguồn, health và thông tin nến | Nguồn Binance, số dòng, ngày đầu/cuối, thiếu nến và cập nhật |
| `07_checkpoint.png` | Mở chi tiết checkpoint ở Kiểm định/Dữ liệu | Model hash rút gọn, ngày fit, training rows và trạng thái chấp nhận |

Không cần bắt đầu paper mới chỉ để có ảnh. Nếu hiện có 0 lệnh thì chụp đúng 0 lệnh và chú thích trạng thái. Không fit lại, import hay đổi model giữa các ảnh để chọn riêng những con số đẹp hơn.

### Ghi chú cho bộ ảnh

Lưu một ghi chú ngắn cạnh ảnh, ví dụ `reports/app/capture_notes.md`, với thời điểm chụp (kèm múi giờ), commit code nếu biết, tài sản, model/checkpoint và trạng thái đồng bộ/paper. Nếu chụp lại ở model hoặc thời điểm khác, ghi rõ bộ ảnh đã đổi.

Trước khi công khai, xem lại đường dẫn thư mục cá nhân, email, API key/token hoặc thông tin tài khoản xuất hiện trong phần metadata. Các vùng riêng tư có thể được cắt khỏi khung; giữ nguyên các metric và cảnh báo làm thay đổi cách hiểu kết quả.

Ảnh cần bổ sung cho README sẽ dùng đường dẫn `reports/app/<tên ảnh>.png`. Khi chưa có ảnh, không tạo hình minh họa giả hoặc để đường dẫn ảnh hỏng trong README. Mỗi ảnh chèn vào README cần có chú thích tài sản, model và thời điểm chụp.
