# QuantStream: từ nghiên cứu đến ứng dụng local

Đây là phạm vi và các bước **đề xuất**, không phải báo cáo những tính năng đã hoàn tất hoặc lời hứa có lợi nhuận. Giữ mục tiêu đã chốt: BTC/ETH, nến 1 giờ, dự báo lợi suất 4 giờ, train trên Kaggle và triển khai paper trading local với chi phí thấp.

## Hiện tại đã biết gì?

- Run 002 đã thử Ridge, XGBoost, TCN và GRU. TCN được chọn trên development nhưng kết quả phụ thuộc seed; diagnostic nhạy với phí/slippage. Chưa có model được promotion.
- Run 004 đã chạy 3 seed × 200 steps. Mean LoRA MAE 0.007815294, cao hơn zero-return 0.007534907 khoảng 3,72% trên cùng 512 cửa sổ development. Không thể gọi LoRA hiện tại là model tốt nhất hoặc model sinh lời.
- Run 005 chỉ đối chứng inference với cùng trọng số; không train, không tìm siêu tham số. Chờ output Kaggle để xem control/reference có khớp hay không.
- Repo đã có ingestion dữ liệu thực, kiểm tra chất lượng, causal features, backtest có chi phí, dashboard và paper worker. Những thành phần này vẫn cần kiểm tra tích hợp trên máy triển khai của người dùng.
- Chưa biết cấu hình máy local hoặc ngân sách GPU cụ thể. Chưa chốt số trial/epoch/steps, model LLM hay cam kết thời gian hoàn tất.

## Khi nào chốt model?

Đợt tìm kiếm phải có ngân sách và quy tắc dừng được ghi trước khi chạy. Chạy hết ngân sách vẫn là một kết quả hợp lệ; không kéo dài đợt cũ chỉ để tìm score đẹp hơn.

| Cổng | Bằng chứng cần có |
| --- | --- |
| Dữ liệu | Nguồn, timestamp, khoảng thời gian, checksum, missing/duplicate/gap report; chỉ dùng thông tin có tại thời điểm dự báo |
| Chọn cấu hình | Các fold theo thời gian với purge theo label end; inner validation chọn epoch/siêu tham số; cùng population và baseline phù hợp |
| Chốt ứng viên | So seed/asset/regime, uncertainty có xét phụ thuộc theo thời gian; lưu cả trial tốt, xấu, lỗi và pruned |
| Đánh giá mới | Khóa thiết kế trước giai đoạn dữ liệu chưa dùng để tuning; dữ liệu test/development đã xem không trở lại thành test độc lập |
| Chiến lược | Kiểm tra net return, drawdown, số lệnh, exposure, turnover có đơn vị rõ, chi phí/slippage stress và quy tắc xử lý tín hiệu 4h chồng lấn |
| Vận hành | Reload artifact trên local, thống nhất preprocessing/forecast, đo latency/RAM/VRAM, xử lý stale data, restart, rollback và giới hạn paper risk |

Tiêu chí chọn dự báo và tiêu chí chọn chiến lược phải tách biệt. MAE tốt hơn không đảm bảo có lợi nhuận; một đoạn backtest có lời cũng không chứng minh dự báo tốt toàn bộ population. Các ngưỡng số cụ thể cần đăng ký trước đợt mới, chưa được xác định bởi tài liệu này.

Ứng dụng vẫn có thể hoàn thiện chức năng dù chưa model nào đạt cổng triển khai. Khi đó hiển thị rõ trạng thái research/no eligible signal; không ép chọn một model yếu để có lệnh.

## So sánh model và phương pháp huấn luyện

| Nhánh | Vai trò và đối chứng | Siêu tham số/kỹ thuật cần ghi |
| --- | --- | --- |
| Zero-return, naive, Ridge | Baseline dự báo; fit preprocessing trong train | Lookback/features, regularization, chuẩn hóa |
| Buy-and-hold, cash/no-trade, EMA | Baseline chiến lược với cùng giai đoạn, vốn và cách tính phí | Quy tắc tín hiệu, exposure, thời gian giữ, phí/slippage |
| XGBoost | Học đặc trưng bảng; so với Ridge và zero-return | Depth, learning rate, số cây, regularization, sampling; không có activation neural để tùy ý chọn |
| TCN/GRU nhỏ | Học chuỗi có thứ tự thời gian | Lookback, width/layers, dropout, learning rate, batch, weight decay, gradient clipping, early stopping |
| Chronos zero-shot | Foundation-model baseline chưa fine-tune | Revision, context, quantiles, precision, model mode, batch inference |
| Chronos LoRA | Fine-tune phần adapter, giữ base weights cố định | Rank, alpha, target modules, LoRA dropout, steps, learning rate và schedule |
| Full fine-tuning/QLoRA | Chỉ nghiên cứu thêm nếu tài nguyên và giả thuyết phù hợp | Không coi là mặc định tốt hơn LoRA; cần kiểm tra hỗ trợ thư viện, numerical parity và ablation riêng |

TCN hiện tại dùng GELU trong causal blocks và head hồi quy tuyến tính. GRU có gate sigmoid/tanh theo kiến trúc của nó. Kiến trúc/activation của Chronos phải đọc từ checkpoint/config thực tế; đổi activation của pretrained model là thay đổi thí nghiệm, không phải nút chỉnh mặc định.

Không biến mọi tổ hợp thành một grid khổng lồ. Đề xuất Optuna TPE và pruning/early stopping, sau đó kiểm tra nhiều seed cho shortlist đã chọn. Số trial, giới hạn từng fit và thời gian GPU vẫn cần chốt trước đợt mới. Không tái mở hay mở rộng ngân sách Run 002/004 đã kết thúc.

## Mỗi notebook nghiên cứu phải cho xem gì?

| Phần | Nội dung trực quan và evidence |
| --- | --- |
| Dữ liệu | Nguồn/khoảng thời gian, phân phối return, missing/gap, số mẫu từng tài sản |
| Chia tập | Timeline train/inner validation/outer validation, purge và vùng test chưa mở |
| Kiến trúc | Input/output shape, layer/activation, số parameters/trainable parameters, config thực tế |
| Quá trình học | Raw training loss, smoothing có ghi rõ, fixed-validation curve nếu đã ghi; learning rate, gradient norm, early-stop/checkpoint |
| Tuning | Bảng tất cả trial, objective theo trial, runtime/tài nguyên, tham số/seed; không chỉ top score |
| Dự báo | Actual vs predicted theo thời gian, residual/distribution, MAE/MSE/direction theo asset/regime/seed; baseline cùng sample |
| Chiến lược | Equity, drawdown, exposure/turnover, số lệnh, phí/slippage stress; không suy PnL từ MAE |
| Kết luận | Điều kiện đạt/chưa đạt, uncertainty, hạn chế, model/data/source hashes, output tải được |

Notebook 07 bổ sung biểu đồ từ log/metric Run 004 đã xác minh và output inference Run 005 khi có. Run 004 không có step-wise validation scores, vì vậy không vẽ thêm đường validation giả hoặc kết luận thời điểm overfit. Run 005 không train nên không có learning curve mới. Những notebook cũ chưa được nâng cấp đồng loạt bởi thay đổi này.

## Thu thập dữ liệu và ứng dụng local

Dữ liệu nến nên lấy bằng public archive/API/WebSocket chính thức, không scrape giao diện HTML. Ingestion hiện có hỗ trợ historical Binance archives/public REST và capture nến đóng qua WebSocket. Cần kiểm tra độ trễ, dedup, thiếu nến, reconnect/catch-up, revision và nguồn gốc dữ liệu trước khi dùng vận hành.

Snapshot dùng train/tuning phải được đóng băng. Dòng dữ liệu live và vùng quan sát mới phải có ranh giới rõ với dữ liệu đã dùng chọn model. Không đưa tin tức, bài đăng hoặc dữ liệu có thời điểm công bố không xác minh được vào feature lịch sử.

Đề xuất bản local đầu tiên:

1. Worker cập nhật nến đóng, kiểm tra chất lượng và ghi dữ liệu.
2. Inference service nạp model đã đăng ký, scaler/features và version phù hợp; ghi prediction + model/data version.
3. Signal/risk logic xác định vị thế paper và quy tắc 4h chồng lấn, không để model tự quyết toàn bộ quy trình.
4. Paper worker ghi lệnh/vị thế, phí/slippage, PnL, lỗi và trạng thái khôi phục.
5. Dashboard cho xem live data, prediction, lịch sử lệnh, hiệu năng model, độ mới dữ liệu và cảnh báo.
6. Gói triển khai có dependencies, cấu hình, hướng dẫn khởi động/dừng, health checks, log, restart và rollback.

Artifact đem về local gồm model hoặc adapter **kèm base revision cần thiết**, preprocessing, feature order, config, model card, benchmark và checksum. Adapter riêng không phải model hoàn chỉnh. Chọn gói nhỏ nhất đạt yêu cầu sau benchmark trên máy thật; có thể ưu tiên model tabular/TCN nhỏ nếu Chronos quá chậm.

Đây là mục tiêu kỹ thuật hướng sản phẩm; một demo local không tự trở thành hệ thống doanh nghiệp nhiều người dùng có SLA.

## LLM / agent nên làm gì?

Đề xuất thêm sau khi pipeline số liệu ổn định. Agent hỗ trợ đọc báo cáo, giải thích feature/metric có nguồn, truy vấn experiment và tóm tắt cảnh báo qua các tool chỉ đọc. Các con số phải được tính bởi code và trả về từ tool, không để LLM tự bịa kết quả.

Qwen3-4B là một ứng viên LLM local cần benchmark sau khi biết RAM/VRAM; Ollama có tool-calling API. Đây không phải lựa chọn đã chốt và không bảo đảm phù hợp máy người dùng. Chưa cần fine-tune LLM: trước hết dùng retrieval/tool calling với báo cáo nội bộ và đánh giá câu trả lời. Agent không được tự mở lệnh tiền thật, thay risk limits hoặc tự launch một đợt train tốn tài nguyên.

## Thứ tự thực hiện

1. Hoàn tất Run 005 hoặc review lỗi môi trường/inference nếu có.
2. Chốt GPU budget và local hardware; thiết kế đợt nghiên cứu mới với data/tuning/evaluation boundaries, baseline và các đồ thị nêu trên.
3. Chạy budgeted search, review tất cả outputs, freeze shortlist và đánh giá mới theo protocol đã đăng ký.
4. Đóng gói candidate đủ điều kiện, benchmark local và chạy paper observation; nếu không có candidate đạt thì công bố kết quả đó.
5. Hoàn thiện vận hành/dashboard rồi bổ sung LLM assistant chỉ đọc nếu còn giá trị sử dụng.

Nguồn kỹ thuật chính thức: [Binance API](https://developers.binance.com/en/docs/products/spot/rest-api), [Optuna sampling/pruning](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html), [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora), [Chronos](https://github.com/amazon-science/chronos-forecasting), [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B), [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling).
