# Notebook → model → ứng dụng local

Notebook 09 là quy trình mới từ dữ liệu thô đến model đem về chạy. Notebook 01–08 giữ lịch sử các thí nghiệm cũ. Không cần chạy lại toàn bộ notebook cũ để dùng notebook 09.

## Chạy trên Kaggle

1. Kéo bản `main` mới bằng `git pull --ff-only origin main` trong thư mục dự án đang dùng.
2. Tạo notebook Kaggle, Import Notebook và chọn `notebooks/09_Kaggle_EndToEnd_Training.ipynb` từ máy bạn. Bật Internet và chọn GPU khả dụng.
3. Giữ mặc định 24 cấu hình, tối đa 12 epoch mỗi fit neural, ngân sách 7.200 giây cho toàn job train. Chọn **Save Version → Save & Run All** để lưu notebook cùng output thực thi. Đừng chạy các cell train nhiều lần để chọn kết quả đẹp hơn.
4. Theo dõi tiến độ trong cell train. Early stopping có thể làm job hoàn tất sớm hơn ngân sách. Worker và watchdog đều có giới hạn thời gian; khi không đủ thời gian, ZIP giữ partial evidence và trạng thái chưa hoàn tất.
5. Khi trạng thái là `completed`, tải `QuantStream_Run006_Output.zip` và `QuantStream_Run006_Model.zip` từ output cuối notebook hoặc tab Output của Kaggle. Giữ đường dẫn phiên bản notebook đã lưu cùng hai ZIP.

Lượt GPU đầy đủ này chưa được chạy thay bạn khi bàn giao code. Những kiểm thử CPU nhỏ chỉ xác nhận đường đi phần mềm, không xác nhận độ chính xác của model nghiên cứu.

## Quy trình trong notebook

| Bước | Bằng chứng được lưu |
|---|---|
| Thu thập | 92 archive BTC/ETH từ 01/2023–10/2026, checksum gốc; REST bổ sung tới hết 13/11/2026 UTC |
| Chất lượng | UTC, nến đã đóng, duplicate conflict, quarantine, missing bars; không nội suy giá qua gap |
| Feature | 18 feature quá khứ, reset indicators ở gap, nhãn return 4 giờ sau độ trễ vào 1 giờ |
| Chia tập | Hai fold validation mở rộng, purge nhãn chồng biên; block kiểm tra hồi cứu riêng ở cuối |
| Chuẩn hóa | Scaler fit trên train của từng fold; neural có inner chronological early stopping |
| Huấn luyện | Ridge, XGBoost, TCN và GRU; seed/cấu hình được ghi; TCN dùng GELU, GRU dùng sigmoid/tanh gates |
| Tuning | Optuna, tối đa 24 cấu hình; top 2 × seed 42/123/2026; lưu tất cả trial và dự báo từng seed |
| Đánh giá | MAE, MSE/zero-return, directional accuracy, sai số theo thời gian; backtest sau phí và drawdown |
| Trực quan | Giá/return, missingness, correlation, split, scaler, cấu trúc model, loss, trial/seed, forecast/actual và equity |
| Xuất model | Trọng số, scaler, tham số, schema feature, SHA, dự báo parity để đối chiếu khi nhập local |

Đồ thị mô tả toàn lịch sử không dùng để fit scaler. Dữ liệu test không đi vào optimizer hoặc lựa chọn siêu tham số. Tuy nhiên, các giai đoạn lịch sử đã từng được quan sát ở các lượt nghiên cứu trước: block cuối được gọi là **kiểm tra hồi cứu**, không phải test đầu tư độc lập. Muốn đánh giá độc lập cần khóa model rồi ghi dự báo trên dữ liệu tương lai chưa quan sát.

Directional accuracy của zero-return chỉ thể hiện tỷ lệ lớp theo quy ước dấu, không phải dự báo có thông tin. Các lần lặp Ridge là deterministic; độ lệch giữa seed bằng 0 không chứng minh model ổn định ở thị trường tương lai. MSE ratio < 1 mới vượt zero-return về MSE; vượt baseline cũng chưa chứng minh có lợi nhuận sau chi phí. Train lâu hơn không bảo đảm dự báo tốt hơn.

## Đem model về app

Giải nén `QuantStream_Run006_Model.zip` vào `artifacts/run006`. Trong thư mục này phải có `bundle_manifest.json`, `metadata.json`, `parity.json` và file trọng số. Thư mục `artifacts` được Git bỏ qua.

Nếu chỉ muốn inference từ model notebook, không tạo thêm baseline local:

```powershell
.\INSTALL_WINDOWS.cmd --inference-only
```

Ridge được xuất thành hệ số/scaler JSON; XGBoost dùng model UBJSON. TCN/GRU cần PyTorch CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
```

Mở app bằng `START_WINDOWS.cmd`. Ở trang **Dữ liệu → Nạp model đã train**, nhập đường dẫn thư mục `artifacts/run006`, rồi bấm **Kiểm tra và nạp model**. Màn hình khởi tạo cũng có mục này nếu chưa có model local.

Nạp model chỉ đăng ký checkpoint, không đổi model đang dùng hoặc train lại. Sau khi nạp, chọn model có nhãn **Notebook 09** ở sidebar. Nếu phiên paper cũ còn vị thế: dừng nhận lệnh mới và chờ vị thế thoát. Nếu phiên đã rỗng: nút **Lưu phiên và dùng model này** lưu nguyên phiên cũ rồi chuyển model. Không xóa state để bỏ qua kiểm tra.

Lần đầu chọn model, app tự nạp nến gần đây nếu chưa có dữ liệu. Phiên paper dùng risk của workspace (mặc định 10% vốn mỗi tài sản/lệnh, drawdown halt 10%), giữ riêng cấu hình train gốc trong báo cáo.

Có thể nạp qua CLI:

```powershell
.\.venv\Scripts\python.exe -m quantstream.terminal --root workspace import-model --folder .\artifacts\run006
```

## TCN cũ đang có trong máy

`artifacts/tcn/model.pt` đi cùng `tcn_metadata.json`. Chạy:

```powershell
.\IMPORT_TCN_WINDOWS.cmd
```

App đối chiếu SHA với reference CPU và Run 001 đã audit. Cùng tên và dung lượng file không chứng minh model đến từ Kaggle. Checkpoint khác cần manifest training gốc. Bộ nạp dùng `weights_only=True`, kiểm tra scaler/lookback/feature và giữ nguyên file gốc. Nạp checkpoint không chạy optimizer.

Checkpoint TCN cũ không tự trở thành kết quả của lượt Run 006. Bản cũ đã train trên toàn development không được đem chạy lại các fold từng thấy rồi gọi đó là OOF.

## Đọc hai loại biểu đồ

Đường equity gần phẳng có thể do ít vào lệnh hoặc giữ tiền mặt, không phải độ chính xác 100%. Xem số lệnh, turnover và exposure cùng lợi nhuận.

Thanh Ridge là hệ số nhân feature đã chuẩn hóa; đó không phải xác suất dự báo đúng. Với TCN/GRU, app hiển thị cửa sổ feature chuẩn hóa làm đầu vào, không gọi heatmap này là feature importance. Độ chính xác phải xem MAE/MSE so baseline, forecast/actual và kết quả ngoài thời gian train.
