# Run 007: kiểm tra cách tăng chất lượng dự báo

Notebook 10 so sánh thời gian train, loss, learning rate và độ gần của dữ liệu. Mục tiêu là tìm cải thiện có thể lặp lại trên validation; không hứa trước model sẽ chính xác hơn.

## Chạy

1. Trong thư mục dự án đang có, chạy `git pull --ff-only origin main`.
2. Tạo notebook Kaggle mới và import `notebooks/10_Kaggle_Extended_Training.ipynb`. Bật GPU và Internet.
3. Có thể Add Input output của notebook Run 006, hoặc một private Dataset chứa `QuantStream_Run006_Output.zip`. Notebook tự tìm, kiểm tra SHA và lấy lại nến/feature đã khóa; không dùng lại trọng số. Không có input thì tải dữ liệu cùng giai đoạn 2023-01 đến hết 2026-11-13 UTC.
4. Điền `NOTEBOOK_URL` trong cell đầu. Giữ mặc định tối đa 48 trial, 60 epoch mỗi neural fit, 14.400 giây cho job train. Chọn Save Version → Save & Run All.
5. Khi status là `completed`, tải `QuantStream_Run007_Output.zip` và `QuantStream_Run007_Model.zip`. Nếu job hết giờ hoặc lỗi, giữ ZIP partial; không xóa rồi chạy lại để chọn kết quả đẹp.

Bốn giờ là trần thời gian job train; thời gian tải dữ liệu và chuẩn bị ngoài job. Early stopping hoặc kết thúc số trial có thể làm job xong sớm. Nếu chạm ngân sách giữa một fit, trạng thái là incomplete và chỉ giữ partial evidence. Không gọi đó là lượt train hoàn tất.

## Các phép so sánh

14 trial đầu cố định: Ridge, XGBoost và sáu nhánh cho mỗi họ TCN/GRU. Sau đó Optuna tìm thêm cấu hình trong phần thời gian dành cho search. Các cặp đối chứng chung seed, dữ liệu, ba fold validation và tham số khác.

| Nhánh neural | Trần epoch | Learning rate | Loss | Dữ liệu train |
|---|---:|---|---|---|
| Đối chứng ngắn | 12 | Cố định | Huber | Expanding |
| Train dài | 60 | Cố định | Huber | Expanding |
| Giảm learning rate | 60 | Cosine | Huber | Expanding |
| Đổi loss | 60 | Cosine | MSE | Expanding |
| Dữ liệu gần đây | 60 | Cosine | Huber | Tối đa 730 ngày |
| Kết hợp | 60 | Cosine | MSE | Tối đa 730 ngày |

Patience=8 áp dụng cho cả đối chứng 12 và 60 epoch, để cặp này chỉ khác trần epoch. Cosine dùng cùng quỹ đạo learning rate khi refit đúng số epoch đã chọn. MSE và Huber là hai hàm khác nhau; không so trực tiếp độ lớn training loss giữa hai hàm để kết luận model nào tốt hơn.

Search còn xét width 16/32/64, dropout, learning rate, weight decay, activation GELU/ReLU/SiLU, cửa sổ 365/730 ngày hoặc toàn lịch sử. Không có LLM hoặc agent trong dự báo.

Các fold validation của Run 007 được đặt lại bên trong development của snapshot cũ. Vì ranh giới validation khác Run 006, không lấy hai điểm validation trung bình của hai run so trực tiếp rồi kết luận tiến bộ. Các nhánh đối chứng 12/60 của Run 007 mới cung cấp phép so sánh cùng tập.

Ba finalist được đánh giá trên seed 42/123/2026. Chọn bằng mean MSE ratio qua seed/fold, không chọn seed đẹp nhất. Seed xuất model cố định 42.

## Đọc kết quả

- `13-training-ablations.png` so các nhánh cố định cùng fold và seed 42. Đây là kết quả một seed của phép so sánh, không phải độ tin cậy thống kê.
- `14-learning-rates.png` hiển thị learning rate thực dùng từng epoch. Các biểu đồ giá, feature, chia tập, loss, seed, forecast/actual, equity vẫn được giữ.
- `selection.json` lưu mean/SD qua seed và `validation_gate`: mean MSE ratio <1 và trung bình từng fold <1 mới được ghi nhận vượt baseline trên các tiêu chí này. Đây không phải kiểm định ý nghĩa thống kê hoặc phê duyệt giao dịch.
- Training rows và thời gian đầu/cuối thực tế được ghi sau khi cắt cửa sổ rolling, không dùng số hàng toàn lịch sử để mô tả một model chỉ train gần đây.
- Block 2026 đã từng được quan sát, nên báo cáo sau lựa chọn vẫn là hồi cứu. Tuning không đọc block này, nhưng không thể gọi nó là test đầu tư độc lập mới.
- Nếu tăng epoch không giảm sai số validation, không ép model chạy đủ 60 epoch. Khi đó cần xem lại tín hiệu feature, target và khả năng thích nghi theo thời gian trước khi tăng ngân sách nữa.

Code được kiểm thử bằng fixture nhỏ trên CPU. Lượt Kaggle Run 007 đầy đủ chưa được thực thi khi bàn giao notebook; không có số liệu performance GPU Run 007 được dựng sẵn.

## Nạp model mới

Giải nén Model.zip vào `artifacts/run007`, kiểm tra thư mục trực tiếp có `bundle_manifest.json`, `metadata.json`, `parity.json` và trọng số. Sau khi cập nhật source, vào Dữ liệu → Nạp model đã train; model hiện nhãn Notebook 10. Bước import kiểm tra SHA và parity, không fit lại.

Nếu dùng TCN/GRU trên Windows, cài CPU PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
```

Giữ nguyên model và evidence Run 006 để đối chiếu. App vẫn chỉ hỗ trợ paper trading và phải lưu phiên paper cũ trước khi đổi model nếu phiên đã rỗng.

Tài liệu tham khảo: [PyTorch CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html), [chia tập theo thời gian](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
