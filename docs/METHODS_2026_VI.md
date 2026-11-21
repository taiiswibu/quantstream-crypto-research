# Các phương pháp mới và phạm vi sử dụng

Nguồn được kiểm tra ngày 16/11/2026. Bảng này mô tả khả năng của model, không chứng minh hiệu quả trên BTC/ETH của dự án. Kết quả thực nghiệm nằm trong notebook đã thực thi, được bàn giao riêng cùng evidence.

| Phương pháp | Nguồn và thời điểm | Quyết định triển khai |
|---|---|---|
| Timestral 5M/28M | [Mistral, 15/10/2026](https://mistral.ai/news/timestral/) | Thử zero-shot trên CPU, có/không covariate. Các model này trả phân vị, không chỉ một đường dự báo |
| TiRex-2 | [Paper 01/07/2026](https://arxiv.org/abs/2607.01204), [code chính thức](https://github.com/NX-AI/tirex-2) | Thử context 128, dự báo đồng thời BTC/ETH; bản mở hỗ trợ inference, không gọi tính năng PRO là tính năng miễn phí |
| Chronos-2 | [Amazon](https://github.com/amazon-science/chronos-forecasting) | Dự án đã thử LoRA trước đây; không coi chạy lại cùng model là bằng chứng của một phương pháp mới |
| TimesFM 3 | [Google, 31/08/2026](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [thông báo license](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) | Không đưa downloaded weights vào bản local thương mại; model weights giới hạn non-commercial/non-production, khác với dịch vụ Google Cloud được cấp phép |
| RLS cập nhật trễ | [Delay-aware audit, 08/10/2026](https://arxiv.org/abs/2610.12232) | Đối chứng nhẹ: chỉ cập nhật khi nhãn đã đủ thời gian. Đây là triển khai của dự án lấy cảm hứng từ phương pháp, không phải tái lập nguyên xi mọi thí nghiệm của paper |
| ORCA | [Paper sửa 03/10/2026](https://arxiv.org/abs/2606.14222) | Tham khảo ý tưởng hiệu chỉnh sai số theo ngữ cảnh. Chưa tuyên bố đã tái lập ORCA |
| OFTER | [Bài báo 16/10/2026](https://doi.org/10.1007/s41060-026-01302-7) | Tham khảo hướng hồi quy phi tham số cho tài chính; chưa đưa vào các phương pháp đã chạy |

## Mục tiêu và thời gian nhãn

Giữ return 4 giờ, chờ vào 1 giờ. Nhãn của quyết định t xuất hiện sau 5 giờ. Đầu vào foundation model chỉ dùng nhãn có `label_end_time < t`; thêm một giờ biên ingest nên target history kết thúc ở t−6. Model dự báo 6 bước để lấy đúng return tại t. Covariate bridge chỉ kéo đến t, không kéo đến t+1 hoặc các giá tương lai.

Funding và premium được lấy từ kho ZIP công khai của Binance. Mỗi ZIP được so checksum sidecar. Funding chỉ được dùng sau event một giờ; premium chỉ dùng sau khi nến đóng thêm một giờ. Dữ liệu thiếu được ghi nhận và lọc theo cùng cohort khi so sánh, không thay bằng 0 để giữ đủ số dòng.

Thử nghiệm foundation là screening theo ngày tại 00:00 UTC. Không so trực tiếp MSE của cohort này với toàn bộ dự báo mỗi giờ. Các model được đánh giá zero-shot, không ghi số bước optimizer hoặc epoch giả. So sánh dữ liệu và RLS dùng cohort hourly riêng. Validation đã dùng nghiên cứu trước đó; không gọi là test độc lập. Chưa loại trừ được dữ liệu pretraining trùng lịch sử nghiên cứu.

## Chỉ tiêu tách riêng

- Dự báo điểm: MSE/MAE và ratio so với zero-return trên đúng cùng dòng dự báo.
- Khoảng phân vị: coverage thực nghiệm, độ rộng và interval score. Coverage tốt không chứng minh dự báo hướng hoặc lợi nhuận tốt.
- Paper: vẫn theo điều kiện chấp nhận của model đang hoạt động. Các khoảng tham chiếu và model thử nghiệm không thay tín hiệu paper.

Ứng dụng mặc định có khoảng tham chiếu từ các return đã đủ nhãn trong 28 ngày gần nhất, chuẩn hóa theo biến động quá khứ. Đây là thống kê tham chiếu, không phải xác suất tương lai được bảo đảm. Chỉ ghi evidence mới khi nến còn mới, rồi chấm khi nhãn đã xuất hiện.

## Chuẩn bị model local

Cài phần phụ thuộc forecast và chuẩn bị weights bằng các lệnh Python trong [hướng dẫn vận hành](VAN_HANH_VI.md#notebook-kaggle-và-môi-trường-tùy-chọn). Model chạy CPU, pin revision và kiểm tra config/weights. App core chạy độc lập với extension.

Trong app: Dự báo → Mô hình dự báo mới → Tính dự báo thử nghiệm. Các model này được ghi rõ là thử nghiệm; không có tự chuyển model paper.

Có thể chuẩn bị thêm model bằng lệnh, sau khi extension đã cài:

```powershell
.\.venv\Scripts\python.exe -m quantstream.modern_models --model Timestral-28M --folder artifacts\forecast\Timestral-28M
.\.venv\Scripts\python.exe -m quantstream.modern_models --model TiRex-2 --folder artifacts\forecast\TiRex-2
```

## License

Timestral code dùng Apache-2.0; weights dùng Modified MIT. License weights có điều kiện doanh thu hợp nhất theo tháng: công ty hoặc chủ sử dụng vượt 20 triệu USD tháng trước cần điều khoản thương mại khác. File LICENSE được tải cùng model và phải giữ khi phân phối weights. TiRex-2 dùng Apache-2.0. Những ghi chú này tóm tắt nguồn model; cần đọc nguyên văn license khi dùng cho doanh nghiệp.

Kho dữ liệu lịch sử có [Binance Vision Dataset Terms, cập nhật 26/08/2026](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md), quy định CC BY-NC-SA 4.0 và điều kiện với dữ liệu dẫn xuất. Bản nghiên cứu này không tự cấp quyền dùng dữ liệu hoặc model dẫn xuất cho kinh doanh; triển khai thương mại cần nguồn dữ liệu và quyền sử dụng tương ứng.

Dữ liệu thị trường trong app dùng [public market-data-only endpoint](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md), không cần API key hoặc quyền đặt lệnh. Độ mới của nến vẫn được kiểm tra trước khi ghi forecast.

## Diễn giải kết quả

[FinVerse, 08/2026](https://arxiv.org/abs/2608.03259) cho thấy benchmark forecasting chung chưa đủ để kết luận hữu ích cho quyết định tài chính. [Một preprint ngày 09/11/2026](https://arxiv.org/abs/2611.12141) cũng ghi nhận kết quả âm với một nhóm foundation model trên intraday index futures. Các nghiên cứu đó không chứng minh BTC/ETH 4 giờ không dự báo được; chúng là lý do phải giữ baseline và kiểm tra riêng trên dữ liệu của dự án.
