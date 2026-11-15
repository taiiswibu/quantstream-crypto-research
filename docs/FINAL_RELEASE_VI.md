# Bản cuối: nghiên cứu, model và ứng dụng local

Notebook 11 giữ mục tiêu dự báo return 4 giờ, trễ vào 1 giờ, BTC/ETH và snapshot đã khóa. Không đổi sang dự báo biến động rồi gọi đó là cải thiện dự báo giá.

## Thiết kế

- Tám cấu hình cố định: hai cấu hình cho mỗi họ Ridge, XGBoost, TCN, GRU. Search không dồn thêm lượt vào Ridge vì nó rẻ hoặc đang đứng đầu.
- Mỗi cấu hình dùng ba fold thời gian đã purge nhãn. Một finalist mỗi họ được kiểm tra trên seed 42/123/2026. Ridge deterministic được tái sử dụng, không train giả ba lần.
- So sánh target gốc với target chia cho biến động quá khứ 48 giờ. Đầu ra luôn được đổi lại đúng đơn vị return trước khi chọn checkpoint, tính sai số, backtest và phục vụ inference. Ridge chuẩn hóa dùng intercept bằng 0 để kiểm tra prior return bằng 0.
- Neural mặc định được thử tối thiểu 24 epoch, tối đa 96 epoch trong pha early stopping, patience 12. Giữ checkpoint validation tốt nhất, kể cả khi epoch đó sớm hơn 24. Refit chỉ dùng số epoch được chọn; không ép checkpoint xấu chạy đủ 96 rồi dùng nó.
- So với việc chỉ đặt trần 60 epoch, pha thử này buộc quan sát thêm quá trình học. Train loss thấp hơn vẫn không tự chứng minh dự báo tốt hơn.
- Ngân sách mặc định 10.800 giây cho job train, có watchdog. Đây là trần, không phải số giờ phải dùng hết. Progress phân biệt các pha: điều kiện cân bằng, seed, checkpoint audit và refit triển khai.

## Điều kiện chấp nhận

Zero-return được giữ làm mốc. Model chỉ được ghi `signal_approved=true` khi đồng thời đạt mean MSE ratio <1, trung bình từng fold <1, trung bình từng seed <1 và cận trên của khoảng bootstrap tuần nằm dưới 0.

Khoảng bootstrap lấy các cụm tuần UTC cùng lúc cho cả hai tài sản, riêng trong từng fold; thống kê là chênh lệch MSE tương đối so với zero. Đây là chẩn đoán xấp xỉ trên validation đã từng dùng nghiên cứu, không phải kiểm định đầu tư độc lập và không hiệu chỉnh toàn bộ lịch sử chọn mô hình.

Nếu không model nào đạt, kết luận giữ nguyên là chưa tìm thấy lợi thế. Gói model nghiên cứu vẫn có thể ghi dự báo mới; app chặn nhận lệnh paper mới. Vị thế paper đã mở vẫn được thoát theo lịch/risk. Baseline zero không được gắn điểm validation của model đã học.

## Hai bộ weights

| Gói | Dữ liệu fit | Điểm hồi cứu |
|---|---|---|
| `QuantStream_Run008_Audit_Model.zip` | Development trước biên test cũ | Thuộc đúng checkpoint audit này; lịch sử đã từng quan sát |
| `QuantStream_Run008_Model.zip` | Refit một lần trên mọi nhãn đã đủ thời gian đến cutoff snapshot | `test=null`: các weights refit này chưa được kiểm tra độc lập |

Refit chỉ diễn ra sau khi khóa lựa chọn cấu hình và hoàn tất chẩn đoán audit; không dùng điểm test để sửa cấu hình. Bộ refit giúp inference dùng dữ liệu quan sát mới hơn thay vì dừng ở tháng 02/2026. Không sao chép điểm test của weights cũ sang weights mới.

## Chạy notebook

Import `notebooks/11_Final_Research_and_Deployment.ipynb`. Trên Kaggle bật GPU và Internet, thêm Input snapshot Run 006 đầy đủ nếu có, điền URL notebook rồi Save Version → Save & Run All. Nếu không có snapshot Input, notebook thu thập lại cùng khoảng ngày cố định. File Diagnostics chỉ có báo cáo, không thay thế snapshot chứa nến/feature.

Notebook cũng chạy CPU ngoài Kaggle và ghi đúng runtime đó. Một bản notebook đã thực thi bằng CPU không được mô tả là kết quả Kaggle GPU. Giữ mọi cell output và các ZIP của đúng lượt chạy.

Kết thúc sẽ có Model.zip để nhập local, Audit_Model.zip để đối chiếu checkpoint lịch sử, Diagnostics.zip nhỏ và Output.zip đầy đủ. Lượt incomplete giữ evidence và không được mô tả là hoàn tất.

## Chạy local trong thư mục hiện có

```powershell
git pull --ff-only origin main
.\INSTALL_WINDOWS.cmd --inference-only
```

Nếu model là TCN/GRU:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
```

Giải nén Model.zip vào `artifacts/run008`, sao cho thư mục này trực tiếp có `bundle_manifest.json`, `metadata.json`, `parity.json` và trọng số. Chạy `START_WINDOWS.cmd`, vào Dữ liệu → Nạp model đã train → chọn thư mục đó, rồi chọn Notebook 11 ở sidebar. Không train lại lúc import. Đồng bộ nến mới trước khi đọc dự báo.

Nếu còn phiên paper cũ có vị thế, dừng nhận lệnh mới và chờ thoát trước khi đổi model. Nếu phiên đã rỗng, dùng nút lưu phiên và chuyển model. Không xóa state để bỏ qua kiểm tra.

Trang Kiểm định cho biết điều kiện validation và provenance. Trang Dự báo chỉ ghi forecast mới theo model SHA, không dựng lại lịch sử bằng model refit. Model chưa được chấp nhận vẫn có thể dùng để quan sát dự báo, nhưng không tạo lệnh paper mới.

Nguồn phương pháp: [Scikit-learn: tránh data leakage](https://scikit-learn.org/stable/common_pitfalls.html), [selection bias trong cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [bootstrap ghép cặp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Các nguồn này không chứng minh model của dự án có lợi thế thị trường.
