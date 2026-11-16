# Lịch sử train và kiểm định local

Lần cài nhanh chỉ nạp 90 ngày để mở ứng dụng sớm. Hai model CPU được fit trên dữ liệu đó; đây không phải cùng lịch sử với checkpoint từ notebook.

Trong trang Dữ liệu, mở **Nạp lịch sử huấn luyện**, chọn gói ZIP có `history_manifest.json` và `bars.parquet`, rồi bấm kiểm tra/nạp. App đối chiếu SHA, thời gian nến, symbol, OHLCV và các khóa trùng. Nếu một nến đã tồn tại nhưng có giá hoặc volume khác, thao tác dừng trước khi ghi dữ liệu. Nạp lịch sử không đổi weights, model đang hoạt động hoặc phiên paper.

Sau khi nạp, nút **Fit lại Ridge và XGBoost** dùng toàn bộ nhãn quá khứ khả dụng để train. Ba fold validation được xác định từ 90 ngày nến gần nhất, nên thêm lịch sử cũ không tự chuyển bài kiểm tra sang một giai đoạn cũ hơn. Mỗi fold vẫn purge nhãn chồng lấn. Các model notebook/TCN đã nhập được giữ lại trong registry.

Nếu có phiên paper, lưu phiên theo luồng hiện có trước khi refit; không xóa state để bỏ guard. Registry ghi số dòng train, mốc nhãn cuối, cửa sổ kiểm định và MSE ratio từng fold. Baseline CPU vẫn là model nghiên cứu paper-only; CV chưa phải xác nhận hiệu quả đầu tư. Cờ chấp nhận của checkpoint notebook và luồng paper hiện có được giữ nguyên.

CLI tương đương để chỉ nạp dữ liệu:

```powershell
.\.venv\Scripts\python.exe -m quantstream.history_import --root workspace --archive "C:\duong-dan\QuantStream_Training_History.zip"
```

Nhiều lịch sử hơn có thể giảm overfit, nhưng không bảo đảm có tín hiệu dự báo return. Chỉ so metric trên cùng symbol, mốc thời gian, horizon và chi phí; CV gần đây vẫn chưa phải bằng chứng prospective độc lập. Quyền sử dụng dữ liệu trong gói lịch sử giữ theo nguồn và manifest, không được thay bằng license của source code.
