# Phần cứng local và ngân sách nghiên cứu kế tiếp

Trạng thái: **đã chốt mục tiêu tài nguyên; chưa triển khai hoặc chạy đợt train mới**. Người dùng cung cấp cấu hình và ủy quyền lựa chọn ngân sách theo hiệu quả. Không lưu các mã định danh thiết bị/sản phẩm vì chúng không cần cho việc thiết kế hay benchmark.

## Máy triển khai

| Thành phần | Thông tin người dùng cung cấp |
| --- | --- |
| CPU | AMD Ryzen 7 4800H, 2,90 GHz |
| RAM | 16 GB, khoảng 15,4 GB usable |
| GPU rời | NVIDIA GeForce GTX 1650, VRAM 4 GB |
| GPU tích hợp | AMD Radeon Graphics; không cộng bộ nhớ này thành VRAM của NVIDIA |
| Dung lượng | 477 GB tổng, đã dùng 348 GB; suy ra còn khoảng 129 GB tại thời điểm cung cấp |
| Hệ thống | x64; chưa xác minh phiên bản OS, NVIDIA driver hay CUDA runtime |

Đánh giá kỹ thuật ban đầu: đây là cấu hình mục tiêu hợp lý để thử dashboard, ingestion và inference model nhỏ. Chưa có benchmark trực tiếp để bảo đảm độ trễ, mức RAM/VRAM hoặc khả năng chạy ổn định dài hạn. Fine-tuning neural/foundation model ưu tiên Kaggle; Ridge/XGBoost có thể fit bằng CPU mà không dùng quota GPU.

## Ngân sách đã chọn

**Trần 6 giờ notebook bật GPU trên Kaggle cho toàn đợt**, không phải mục tiêu bắt buộc sử dụng hết. Mục tiêu vận hành là 2–4 giờ nếu hoàn thành đủ kiểm tra sớm. Không mở song song nhiều notebook GPU. Không giả định tài khoản đang có bao nhiêu quota; quota khả dụng và khả năng cấp GPU phải xem tại thời điểm chạy.

Đơn vị ở đây là thời gian notebook có GPU được cấp. Đây không phải khẳng định cách Kaggle quy đổi quota khi máy có nhiều GPU. Dùng một device cho job; không nhân đôi fits hoặc chuyển sang distributed training.

| Phần | Phân bổ tối đa | Giới hạn công việc |
| --- | ---: | --- |
| Setup, kiểm tra môi trường và Run 005 | 30 phút | Đối chứng inference hiện tại; 0 optimizer steps. Mismatch thì giữ evidence và dừng, không cài lại/chạy lại nhiều lần để đuổi score |
| Tìm cấu hình ML/DL mới | 180 phút | Tối đa 24 cấu hình tổng, Ridge/XGBoost/TCN/GRU; hai fold tuning; neural fit tối đa 12 epoch và early stopping inner validation |
| Kiểm tra shortlist | 60 phút | Top 2 cấu hình × seed 42/123/2026; tái dùng evidence seed 42 nếu đúng cùng phép đo. Lưu dự báo của tất cả seed, không chỉ summary |
| LoRA có điều kiện | 60 phút | Tối đa 3 cấu hình mới trong protocol riêng, chỉ sau khi review Run 005. Steps/LR/rank chưa được khóa; không khởi chạy nhánh này chỉ vì còn quota |
| Dự phòng kỹ thuật/xuất evidence | 30 phút | Lưu output khi lỗi/ngắt, thu gom artifact; không dùng để bổ sung trial sau khi đã thấy kết quả |
| Tổng | **360 phút** | Bỏ phần không cần thiết; không tự kéo dài hoặc bù thêm seeds/trials cho kết quả không thuận lợi |

Chuẩn bị dữ liệu, kiểm tra baseline CPU, đọc báo cáo, vẽ lại biểu đồ và benchmark local nên thực hiện khi không cấp GPU nếu có thể. Nếu Run 005 đã được thực hiện từ một phiên trước thì ghi thời gian thực tế và không chạy lại; phần phân bổ dư không biến thành trial bổ sung.

24 trial là 24 cấu hình, **không phải 24 lượt fit**: từng fold có fit riêng và neural có inner pilot/refit. Giới hạn số trial/epoch và thời gian đều phải được runner ghi vào plan trước fit đầu tiên. Nếu thời gian không đủ cho kiểm tra shortlist, giữ kết quả là incomplete thay vì công bố một winner đã kiểm tra đủ.

### Timeout chưa được triển khai cho đợt mới

Runner `quantstream.search.run_search` hiện tại giới hạn thời gian bắt đầu trial tìm kiếm, **không** chặn toàn bộ job: fit đang chạy, seed stability và diagnostic có thể tiếp tục. Vì vậy không coi việc truyền `timeout_seconds` vào runner cũ là đã bảo đảm trần 6 giờ.

Runner kế tiếp cần timer theo từng phần và watchdog toàn tiến trình, checkpoint/progress ghi thường xuyên, archive partial evidence khi hết giờ; thời gian setup/download cũng phải được tính hoặc ghi riêng rõ ràng. Tắt accelerator sau khi lưu evidence. Tài liệu này chốt ngân sách và yêu cầu triển khai, chưa khẳng định watchdog/notebook nghiên cứu mới đã tồn tại.

## Model local ưu tiên benchmark

| Vai trò | Ưu tiên | Quy tắc lựa chọn |
| --- | --- | --- |
| Dự báo dạng bảng | Ridge, XGBoost trên CPU | So cùng sample/baseline, latency và tài nguyên thực; độ đơn giản là lợi thế khi chất lượng tương đương |
| Dự báo chuỗi | TCN/GRU nhỏ, batch inference 1 | Benchmark CPU và GTX 1650; không bắt buộc GPU nếu CPU đáp ứng nhịp nến 1 giờ |
| Foundation forecast | Chronos-2/adapter đã đánh giá, thử batch 1 | Không khẳng định 4 GB là không chạy được. Chỉ giữ nếu numerical parity, latency và tài nguyên đạt trên máy thật |
| LLM trợ lý báo cáo | Ollama `qwen3:1.7b` Q4_K_M | Context bắt đầu 2.048 token, output tối đa 512 token, một request; benchmark GPU offload/CPU fallback và chất lượng tool calling |

Trang Ollama công bố package `qwen3:1.7b` Q4_K_M khoảng 1,4 GB; runtime còn có KV cache, workspace, context và phần mềm khác. Không coi model file size là tổng VRAM/RAM cần dùng. Qwen3-4B là ứng viên phụ sau benchmark, chưa phải mặc định trên GPU 4 GB.

Không fine-tune LLM trong ngân sách nghiên cứu này. Dùng retrieval và tool chỉ đọc trên báo cáo/experiment có nguồn; tính metric bằng code. Agent không tự đặt lệnh tiền thật, thay risk limits hoặc khởi chạy các fit ngoài plan.

## Tiêu chí benchmark mục tiêu

Các số sau là **mục tiêu thiết kế**, chưa phải số đo trên máy người dùng:

- Nhịp inference theo nến 1 giờ đã đóng; đo cả feature building + model inference, ngoài cold start/download. Mục tiêu p95 không quá 30 giây cho một lượt gồm BTC và ETH, trên một tập replay cố định.
- Mục tiêu tổng các process của ứng dụng dùng không quá 8 GiB RAM ở steady state; đo cả giai đoạn peak. Chạy forecast và LLM tuần tự ở bản đầu nếu cùng tranh VRAM.
- Nếu dùng NVIDIA, mục tiêu peak VRAM toàn GPU không quá 3 GiB để còn chỗ cho desktop/process khác; VRAM thực sự khả dụng phải đo lại. Không tự loại model chỉ vì số VRAM danh nghĩa; dùng số đo và fallback CPU.
- Artifact/cache của ứng dụng mục tiêu không quá 20 GB, không giữ mọi optimizer checkpoint ở local. Không tự xóa file/cache của người dùng để đạt mục tiêu.
- Forecast parity phải so với bản export cùng precision/preprocessing; nếu quantize hoặc merge adapter, đó là một phép chuyển đổi cần kiểm tra lại, không mặc định giữ nguyên dự báo.
- Đo latency/RAM/VRAM, restart, stale data và replay determinism trước khi tuyên bố local-ready. Sau đó mới đo paper observation; chưa bắt đầu giai đoạn quan sát mới bằng tài liệu này.

Vẫn giữ `research_only_no_promotion` cho các model hiện có. Một hệ thống paper trading có thể hoàn thiện chức năng với trạng thái no eligible signal, kể cả khi ứng viên chưa đạt cổng chất lượng/chiến lược.

## Việc tiếp theo

1. Nhận và review `QuantStream_Run005_Output.zip` hoặc traceback/partial ZIP nếu môi trường khác; chưa có output mới được xác minh tại thời điểm ghi kế hoạch.
2. Chuẩn bị protocol/source/data boundary và notebook đợt nghiên cứu mới, có watchdog thực, learning/validation curves và toàn bộ baseline/seed predictions. Không sửa ngân sách của Run 002/004 đã kết thúc.
3. Chạy đợt mới trong trần đã chọn; giữ tất cả kết quả và lỗi. Thời điểm model được chốt phụ thuộc các cổng đã đăng ký và đánh giá mới, không phụ thuộc việc dùng hết giờ GPU.
4. Đóng gói ứng viên đủ điều kiện và benchmark trên máy người dùng theo các mục tiêu trên.

Nguồn chính thức được kiểm tra khi chọn LLM: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Ollama model package](https://ollama.com/library/qwen3:1.7b), [Ollama hardware support](https://docs.ollama.com/gpu). Chưa xác minh driver trên máy người dùng; không suy ra GPU offload đã hoạt động chỉ từ tên card.
