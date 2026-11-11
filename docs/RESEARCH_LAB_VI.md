# QuantStream Research Lab — học bằng thí nghiệm có kiểm soát

## Bạn đang ở đâu?

TCN đã train thành công trên CUDA. Run 001 chọn epoch 3/4/2 trong ba fold; thêm epoch làm train loss giảm nhưng chưa cải thiện validation. Số liệu sau phí trên development không chứng minh chiến lược có alpha. Final holdout cũ đã xem và không được dùng để chọn cấu hình mới.

Một researcher cần biết **câu hỏi, thiết kế kiểm tra, kết quả và giới hạn**, thay vì chỉ có nhiều file model. Giữ cả thí nghiệm không cải thiện trong Git.

Run 002 và Run 003 đã hoàn tất: xem [model search](experiments/run-002-model-search.md) và [LoRA A/B/C](experiments/run-003-lora-lab.md). Không cần chạy lại hai vòng này để tiếp tục dự án. [Run 004](experiments/run-004-seed-stability-protocol.md) mới là bản thiết kế kiểm tra độ ổn định qua seed; chưa có runner hoàn chỉnh hoặc kết quả training.

## Phân biệt các thao tác

| Thao tác | Ví dụ trong dự án | Bạn học gì? |
| --- | --- | --- |
| Training từ đầu | TCN / GRU với weights khởi tạo ngẫu nhiên | loss, optimizer, batch, sequence, epoch |
| Hyperparameter tuning | Optuna tìm learning rate, width, dropout, lookback | search space, objective, budget, selection bias |
| Fine-tuning pretrained | Chronos-2 + LoRA | base checkpoint, adapter, learning rate nhỏ, before/after |
| Backtesting | dự báo → tín hiệu → khớp lệnh → phí → equity | độ trễ, chi phí, drawdown, số giao dịch |
| Forward evaluation | dự báo được lưu trước khi có giá tương lai | kiểm tra ngoài mẫu mới thực sự |

Không dùng checkpoint TCN của run 001 để khởi tạo các fold quá khứ: checkpoint đó đã học toàn development, sẽ làm lộ dữ liệu các validation fold.

## Run 002 — so sánh và tuning, notebook 04

1. Đặt giả thuyết: các họ model phức tạp có giảm sai số so Ridge và dự báo 0 trên dữ liệu causal không?
2. Khóa features/target/costs; giữ BTC/ETH, nến 1h, delay 1h, hold 4h. Không trộn tuning model với đổi nhãn hoặc risk rule.
3. Dùng hai outer fold đầu cho tìm kiếm. Trong mỗi outer train, 20% mốc thời gian cuối làm inner early stopping; label crossing ranh giới bị purge. Hai tài sản dùng cùng mốc UTC.
4. Neural model chọn epoch trên inner validation, rồi khởi tạo lại và refit outer train với số epoch cố định. Outer validation không được chọn epoch hoặc scaler.
5. Optuna tìm tối đa **12 trial tổng**, 8 epoch/lượt fit; bốn trial đầu gồm Ridge alpha=10, XGBoost depth=3/LR=.04, một TCN và một GRU. Cấu hình tiếp theo do TPE đề xuất; không đảm bảo số trial bằng nhau cho từng họ.
6. Objective là mean của MSE chia MSE dự báo lợi suất 0 ở từng fold; nhỏ hơn 1 mới thắng baseline 0 về sai số. Strategy threshold 0.006 đã cố định từ lịch sử run 001; không tune threshold trong run này.
7. Top 2 cấu hình thử ba seed 42/123/2026 trên hai fold đầu; chọn theo mean objective, không chọn seed đẹp nhất. Ridge deterministic nên variance seed bằng 0 không có nghĩa chắc chắn hơn về tương lai.
8. Lưu `selection.json`, khóa trước khi chấm fold 3. Chỉ chấm candidate đã chọn; sau đó tính sensitivity chi phí 1x/1.5x/2x. Fold 3 **đã được xem trước đây**, nên đây là diagnostic development, không được gọi là test độc lập.

Neural và tabular được chấm cùng tập timestamp có ít nhất 48 giờ feature liên tục; các cửa sổ qua gap bị loại. Lookback của neural là 24 hoặc 48; TCN dùng bốn causal block, GRU một lớp, width 16/32, head dropout 0/.15/.3; AdamW LR 1e-4..2e-3 và weight decay 1e-5..1e-2. Batch 256/512. XGBoost depth 2..4, LR từ 0.02 đến 0.1, 120 cây; Ridge alpha từ 0.1 đến 100.

Các con số này là **ngân sách dự kiến**, chưa phải kết quả đạt được. Timeout 7200 giây chỉ dừng bắt đầu trial search mới, không ngắt trial đang chạy hay giai đoạn seed/diagnostic. Nếu quota ít, chọn 4 trial / 4 epoch ngay trước lần chạy đầu và ghi rõ vào protocol.

### Output và resume

- `plan.json`: tham số cố định, mã nguồn checksum, source Git SHA, dataset hash, GPU/Python/package versions.
- `study.sqlite3`, `trials.csv`, `trials/*`: toàn bộ trial và prediction, giữ lịch sử Optuna.
- `stability/*`, `selection.json`: nhiều seed và quyết định trước diagnostic.
- `diagnostic.started.json`, `diagnostic.json`, `diagnostic_predictions.parquet`: chặn âm thầm chạy diagnostic lặp lại.
- `result.json`: kết quả tổng hợp và giới hạn; không tự refit checkpoint phục vụ dashboard hoặc promote champion.

Giữ nguyên output để resume và Save Version trên Kaggle. Nếu diagnostic đã bắt đầu mà bị lỗi, giữ ZIP evidence gửi về để review; không đổi thư mục nhằm xóa dấu vết rồi chọn kết quả đẹp. Resume TPE giữ các trial/parameter đã lưu; RNG sampler khởi động lại khi process restart, nên đường tìm kiếm tiếp theo có thể khác chạy liền mạch.

## Run 003 — fine-tuning thật, notebook 05

Chạy riêng sau run 002, khi GPU còn quota. Dùng cùng 512 cửa sổ, base Chronos revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, context 128, batch 8, seed 42.

| Cấu hình | Steps | Learning rate | So sánh cần học |
| --- | ---: | ---: | --- |
| A | 100 | 1e-5 | baseline LoRA ngắn |
| B | 200 | 1e-5 | B so A: thêm bước có làm overfit? |
| C | 200 | 3e-6 | C so B: LR thấp có ổn định hơn? |

Mỗi cấu hình khởi tạo lại từ cùng base, không nối adapter của cấu hình trước. `finetuned_mae - zero_shot_mae < 0` là cải thiện trên những cửa sổ này. Không suy diễn sang PnL, không gọi dữ liệu này là test độc lập. Pretraining overlap chưa xác minh. API dùng LoRA của Chronos/PEFT; rank/target-module giữ mặc định thư viện và chưa phải biến được tune ở vòng này.

ZIP output giữ adapter, dự báo từng hàng và metadata. Giữ trainer log ở Kaggle để xem quá trình học. Không train full model hoặc mở rộng hàng trăm trial khi chưa phân tích vòng nhỏ.

### Môi trường Kaggle và lỗi torchao

Run 003 đã hoàn thành; xem [báo cáo](experiments/run-003-lora-lab.md). Không cần train lại để sửa môi trường. Notebook 05 mới dành cho lần tái lập có chủ đích:

1. Restart Session rồi Run All, Internet On và GPU khả dụng. Nếu đã import PEFT/Chronos/Transformers/torchao, cell đầu dừng trước khi thay đổi thư viện và yêu cầu restart.
2. Cell gỡ optional `torchao`, cài Chronos `2.3.2` và PEFT `0.21.2`, giữ PyTorch/CUDA của Kaggle. Lab dùng float32, không cần torchao quantization. Không nâng torchao tùy ý vì còn phụ thuộc phiên bản PyTorch.
3. Trước khi tải checkpoint, thử gắn LoRA vào một lớp Linear và chạy backward trên CPU. Kiểm tra này phát hiện đường lỗi adapter injection; chưa kiểm chứng toàn bộ Chronos/GPU training.
4. Lưu `environment.json`, `pip-freeze.txt` và trainer JSON/JSONL/log nếu được tạo. Các package còn lại có khoảng phiên bản, nên đây **chưa phải environment lock đầy đủ** hoặc bảo đảm bitwise reproducibility.
5. Lần chạy mới dùng `run-003-lora-lab-rerun` và `QuantStream_LoRA_Rerun_Output.zip`, giữ nguyên evidence Run 003 cũ. ZIP vẫn xuất khi lỗi; cần đủ A/B/C và `comparison.json` mới xem là hoàn tất.

Nguồn lỗi: [PEFT issue #3964](https://github.com/huggingface/peft/issues/3964). Upstream đã sửa trên main; notebook pin PEFT theo phiên bản được ghi trong artifact Run 003 và xử lý dependency cũ rõ ràng, không dựa vào main chưa cố định.

## Cách tự đọc một thí nghiệm

Sau mỗi lượt chạy, trả lời bằng số liệu:

1. Mô hình có thắng dự báo lợi suất 0 theo MSE/MAE không? Thắng Ridge không?
2. Train loss giảm trong khi inner validation tăng từ epoch nào? Early stopping chọn gì?
3. Đổi seed có đảo thứ hạng? Cải thiện có tập trung ở một fold/tài sản không?
4. Sau phí còn bao nhiêu giao dịch, drawdown bao nhiêu, kết quả có chịu được cost 2x không?
5. Đã chọn bao nhiêu cấu hình trên cùng development? Có đủ dữ liệu mới để đánh giá ngoài mẫu chưa?
6. Thí nghiệm tiếp theo kiểm tra đúng một giả thuyết nào? Kết quả âm sẽ được ghi ra sao?

Khi có ZIP thật, cập nhật experiment log và commit kết quả ở một commit riêng. Không ghi trial dự kiến thành số model đã train. Chưa có independent alpha evidence thì để trạng thái `research_only`.

## Vòng nghiên cứu sau khi đọc kết quả

Các hướng có thể đăng ký trước: bỏ từng nhóm feature (ablation), dữ liệu rolling vs expanding, phân tích theo asset/regime, uncertainty bằng block bootstrap, rank-based signal và portfolio/risk sizing. Những việc này **chưa được triển khai trong run 002/003**. Mỗi thay đổi cần ghi giả thuyết, ngân sách và tiêu chí trước khi chạy; nhiều thử nghiệm trên cùng development làm tăng nguy cơ overfit.

## Tài liệu gốc

- [TimeSeriesSplit và gap — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [TPESampler — Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html)
- [Saving/loading/warm-start — PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Chronos forecasting source](https://github.com/amazon-science/chronos-forecasting)
