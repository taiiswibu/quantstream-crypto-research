# Run 004 — độ ổn định của LoRA qua nhiều seed

Trạng thái: **bản thiết kế, chưa chạy**. Không có metric mới hoặc model được promote. Thiết kế này nối tiếp Run 003; chưa phải bằng chứng ngoài mẫu độc lập.

## Câu hỏi và phạm vi

Run 003 cho thấy cả A/B/C đều có MAE cao hơn zero-shot trên một seed. Câu hỏi hẹp của Run 004: với **cùng cấu hình B**, mức thay đổi MAE so với zero-shot có đổi dấu khi thay seed không, và khác biệt nằm ở BTC hay ETH?

Chọn B vì nó có MAE thấp nhất trong ba adapter Run 003. Đây là lựa chọn sau khi xem development; phải ghi nhận selection bias. Thí nghiệm đo độ ổn định, không chứng minh seed là nguyên nhân duy nhất khiến fine-tuning kém và không tìm seed tốt nhất để triển khai.

## Ngân sách dự kiến, không mở rộng sau khi xem kết quả

| Thành phần | Giá trị |
| --- | --- |
| Base model | amazon/chronos-2 |
| Base revision | 29ec3766d36d6f73f0696f85560a422f50e8498c |
| Cấu hình LoRA | B: 200 steps, learning rate 1e-5 |
| Seeds | 42, 123, 2026 |
| Tổng training | 3 lần fit mới, tối đa 600 optimizer steps cộng lại |
| Context / batch / cửa sổ | 128 / 8 / cùng 512 cửa sổ Run 003 |
| Dữ liệu | cùng snapshot development; manifest SHA-256 c8e0112916e6cea733338ead4412d6e9553e4c5844a2d200d1fd38e31c747fd0 |
| So sánh | dự báo lợi suất 0, Chronos zero-shot, LoRA B ở cả ba seed |

Mỗi fit bắt đầu từ cùng pretrained checkpoint. Seed 42 chạy lại là kiểm tra tái lập, không được tính kết quả Run 003 cũ thành một lần lặp độc lập mới. Giữ nguyên training series, thứ tự sample, target, horizon và cách xấp xỉ entry của Run 003. Không thay rank/target modules hoặc tune threshold. Khóa cấu hình LoRA mặc định đã resolve trong plan trước fit, thay vì chỉ ghi “default”.

Nếu hết quota hoặc lỗi, giữ evidence, ghi trạng thái partial/failed và số step thật. Không thêm seed để bù cho kết quả xấu; không chạy lại trial lỗi mà xóa dấu vết.

## Đo lường và cách đọc kết quả

1. Xuất toàn bộ dự báo ở cả ba seed. Kiểm tra đúng cùng symbol/time/target và cùng baseline; không so sánh các tập hàng khác nhau.
2. Báo MAE, MSE, tỷ lệ so baseline dự báo 0 và directional accuracy cho từng seed, toàn bộ mẫu và từng tài sản. Báo cả ba kết quả, mean và độ phân tán; không chỉ dòng tốt nhất.
3. Chênh lệch chính là MAE LoRA trừ MAE zero-shot. Âm là giảm sai số trên mẫu này. Nếu đổi dấu giữa seed, kết luận kết quả nhạy với khởi tạo; nếu cả ba cùng dương, ghi chưa thấy cải thiện với cấu hình đã chọn.
4. Lưu train loss theo step và checkpoint metadata. Chỉ có endpoint development không đủ để xác định thời điểm overfitting; train loss giảm không tự chứng minh tổng quát hóa tốt hơn.
5. Khoảng bất định mô tả: paired block bootstrap 2.000 lần, RNG 2026, block theo tuần lịch UTC (thứ Hai 00:00), giữ BTC/ETH trong cùng block và giữ các block biên ngắn. Mỗi lượt lấy lại cùng số block có hoàn lại; tính tổng sai số / tổng số hàng. Tính chênh lệch lỗi trung bình qua ba seed trên mỗi hàng trước khi resample. Cửa sổ Run 003 không cách đều theo giờ, nên không coi 24 hàng liên tiếp là 24 giờ. Khoảng percentile 95% chỉ là diagnostic trên development đã xem, không phải xác nhận alpha hoặc kết quả vượt qua multiple testing.

Ba seed là ngân sách nhỏ, chưa đủ để mô tả đầy đủ phân phối khởi tạo. Không suy từ MAE sang PnL; Run 004 phần này không đổi chiến lược, không có backtest lợi nhuận mới.

## Điều kiện phải hoàn tất trước training

- Có runner/notebook Run 004 và regression checks cho seed isolation, sample alignment, artifact completeness.
- Commit code + protocol từ working tree sạch; ghi source SHA, snapshot hash và đầy đủ tham số đã resolve.
- Chạy preflight adapter injection/backward; chốt environment lock sau smoke test trên đúng Kaggle Python/PyTorch/CUDA. File pip freeze local CPU không đại diện cho Kaggle GPU.
- Ghi notebook version/URL, GPU, Python/package versions. Xuất trainer global_step/log_history, dự báo từng seed, adapter hashes và trạng thái hoàn tất/lỗi.
- Dùng output Run 004 riêng, không ghi đè Run 003 và không mở final holdout cũ.

Hiện các điều kiện runner và Kaggle environment lock **chưa hoàn tất**. Tài liệu này không xác nhận một thí nghiệm đã đăng ký và thực thi đầy đủ.

## Đánh giá độc lập tiếp theo

Không đổi tên các cửa sổ development đã xem thành test. Sau diagnostic nhiều seed, cần một protocol prospective riêng:

- Cố định model, ensemble/rule và chiến lược trước khi bắt đầu; không chọn seed bằng PnL.
- Chỉ bắt đầu sau khi code, môi trường và bộ ghi dự báo trước thời điểm có nhãn đã sẵn sàng. Ghi chính xác mốc UTC trong plan trước dự báo đầu tiên; dự kiến lấy 00:00 UTC kế tiếp sau khi các điều kiện đạt.
- Quan sát 28 ngày liên tiếp, không dừng sớm vì kết quả đẹp hoặc xấu. Mỗi dự báo phải được ghi trước entry và label_end; lỗi thu thập phải được ghi, không backfill dự báo rồi gọi là forward.
- Khóa tập đánh giá trước khi mở nhãn tổng hợp, giữ sample alignment, xử lý gap và chi phí đã khai báo. 28 ngày chỉ là pilot, chưa bảo đảm đủ số giao dịch hoặc statistical power.
- Chỉ báo PnL sau khi entry/exit, sizing, phí/slippage và risk rules đã được khóa. Không tự động chuyển sang giao dịch tiền thật.

Mốc ngày cụ thể và source SHA prospective chưa được chốt, vì collector và runner của vòng này chưa có. Không có forward evaluation nào đang được ngầm chạy.

## Tiêu chí kết thúc

Hoàn thành khi có đủ evidence của ba fit hoặc báo cáo partial có lý do, bảng đầy đủ seed/asset/baseline và quyết định keep/reject/inconclusive. Giữ `research_only_no_promotion` dù có seed thắng trên development. Mọi thay đổi cấu hình sau khi xem kết quả phải là vòng nghiên cứu khác.

