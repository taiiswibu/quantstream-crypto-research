# QuantStream Local Terminal

Web app chạy trên máy cá nhân: Streamlit + Plotly, model Ridge và XGBoost đã fit trên dữ liệu Binance Spot thật. Không dùng LLM, chatbot hay agent. GTX 1650 không cần thiết; inference chạy CPU.

## Kéo từ GitHub trên Windows

Cần Git và Python 3.12 64-bit. Trong PowerShell:

```powershell
git clone https://github.com/taiiswibu/quantstream-crypto-research.git
cd quantstream-crypto-research
.\INSTALL_WINDOWS.cmd
.\START_WINDOWS.cmd
```

Lần đầu installer nạp 90 ngày nến BTC/ETH thật rồi fit Ridge và XGBoost trên CPU. Không cần tải ZIP hay chạy notebook Kaggle. Nếu đã có model trong `workspace`, installer giữ lại model và phiên paper. Cần Internet cho lần cài/nạp dữ liệu đầu; lỗi mạng được báo và không đổi thành dữ liệu mẫu.

Đã clone trước đó: dừng app/worker, chạy `git pull --ff-only`, rồi `INSTALL_WINDOWS.cmd` sau lần cập nhật và `START_WINDOWS.cmd`. Không ghi đè file bạn đã sửa; nếu Git báo conflict, giữ thay đổi trước khi cập nhật.

Linux: `bash START_LINUX.sh`.

## Mở bản ZIP trên Windows

1. Cài **Python 3.12 bản 64-bit** từ https://www.python.org/downloads/windows/ nếu máy chưa có.
2. Giải nén toàn bộ `QuantStream_Local_Terminal.zip` vào một thư mục, ví dụ `D:\QuantStream`.
3. Mở `INSTALL_WINDOWS.cmd` một lần. Cần Internet để tải thư viện từ PyPI.
4. Mở `START_WINDOWS.cmd`. Trình duyệt mở app tại `http://localhost:8501`. Giữ cửa sổ lệnh mở.
5. Bấm **Đồng bộ thị trường**, rồi bật **Tự đồng bộ mỗi phút**. Nếu mạng Binance bị chặn, app báo lỗi và giữ dữ liệu đã lưu.

Model và 90 ngày nến đã có sẵn trong bản phát hành. Lần đầu không cần chạy lại notebook Kaggle. App không tự cài Torch hoặc tải model LLM.

Nếu cài đặt bị ngắt, chạy lại `INSTALL_WINDOWS.cmd`. Khi port 8501 đã có app khác sử dụng, đóng phiên cũ trước khi mở phiên mới. Không mở cổng này ra Internet.

## Sử dụng

| Trang | Nội dung |
|---|---|
| Thị trường | BTC/ETH, biểu đồ nến và volume, ticker 24 giờ, dự báo return 4 giờ |
| Dự báo | Lịch sử forecast phát sinh sau triển khai, đóng góp feature Ridge / importance XGBoost |
| Kiểm định | Baseline zero-return và naive, ba fold thời gian, sai số, chiến lược sau phí |
| Danh mục | Vốn giả lập 10.000 USDT, lệnh chờ, phí/slippage, equity và sổ lệnh CSV |
| Dữ liệu | Nguồn, thời gian, thiếu nến, checksum, health log và fit lại từ nến đã lưu |

Model trong app là baseline local mới, không phải adapter Chronos Run 004/005 hay champion TCN cũ. Không chọn model tốt nhất từ một lượt inference ngẫu nhiên. Kết quả Run 001–005 thuộc các thí nghiệm khác và không được gộp vào bảng kiểm định local.

Dự báo là return sau thời gian chờ 1 giờ, giữ 4 giờ. Giá hiện tại từ ticker; nến chỉ gồm nến đã đóng. Thời gian hiển thị UTC. Một forecast mỗi model/tài sản/nến; không tạo lại lịch sử dự báo bằng model fit sau đó.

Chỉ bắt đầu paper khi cần mô phỏng. **Dừng nhận lệnh mới** hủy lệnh chờ và vẫn theo dõi vị thế đã mở. Bật tự đồng bộ khi dùng app, hoặc mở `WORKER_WINDOWS.cmd` để tiếp tục đồng bộ mỗi phút khi đóng trình duyệt. Dùng một worker; laptop ngủ/tắt hoặc mất mạng thì không có khớp lệnh trong thời gian đó. Lệnh vào bị trễ quá 2 phút sẽ bị bỏ; thoát vị thế trễ được ghi lại.

Muốn đổi model hoặc fit lại: dừng nhận lệnh mới, chờ vị thế thoát, chọn **Lưu phiên và dừng mô phỏng**. Phiên cũ còn trong `workspace/paper`. Không xóa state để giấu drawdown.

## Kết quả và giới hạn

Validation dùng ba fold mở rộng với purge ở ranh giới. Scaler chỉ fit trên train của fold. Model phục vụ được refit trên tất cả nhãn đã biết sau kiểm định. Validation đã dùng để so sánh model, **không phải holdout đầu tư độc lập**.

Đọc MSE/zero-return của từng lần fit ở trang Kiểm định; tỷ lệ dưới 1 mới vượt baseline về MSE. Mục đích bản này là sản phẩm local có dữ liệu thật, inference thật và kiểm định đọc được; **chưa đủ bằng chứng để dùng vốn thật**. Equity validation là mô phỏng với chi phí giả định. Paper chưa bắt đầu sẽ không có lịch sử/PnL tạo sẵn.

Chưa có đăng nhập nhiều người, vận hành máy chủ 24/7, SLA, kiểm toán bảo mật hay kết nối đặt lệnh sàn. Muốn triển khai công khai cần phạm vi hosting, xác thực và đánh giá vận hành riêng.

## Notebook

Thư mục `notebooks` có các notebook nghiên cứu đã rút gọn thành code + comment. Code thí nghiệm và source SHA khóa giữ nguyên. Các notebook Kaggle cần đúng dataset/source của từng run; không Run All các thí nghiệm đã hoàn thành để chọn lại kết quả. Notebook 08 đọc artifacts local, không train lại. Bản Git chỉ giữ code; chạy notebook sau khi setup để xem kết quả trên workspace của bạn. Bản ZIP riêng có biểu đồ đã lưu.

Từ bản nguồn và môi trường dev: `python -m jupyter nbconvert --execute --to notebook --inplace notebooks/08_Local_Model_Review.ipynb`. Thư viện notebook/Matplotlib không cần để chạy web app. Các notebook Kaggle 02–07 không được chạy lại toàn bộ trong lần chỉnh giao diện này. Notebook 01 cần artifacts thí nghiệm cũ đã lưu trước đây; bản cài app chỉ kèm artifacts local cho Notebook 08.

## Lưu và khôi phục

Tắt app/worker trước khi sao chép toàn bộ thư mục `workspace`. Đây là dữ liệu, model, kiểm định, SQLite forecast và phiên paper của bạn. Model chỉ được tải từ artifacts do dự án tạo; không chép file joblib không rõ nguồn vào workspace. Không cần API key sàn.

Đã kiểm thử tự động và xem giao diện trên Linux. Script Windows được kiểm tra nội dung; chưa thực thi trên máy Windows của bạn.
