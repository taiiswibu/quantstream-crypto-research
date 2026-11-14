# Run 005 — NumPy metadata mismatch trước inference

Status: **probe received and verified; bootstrap repair implemented; full Kaggle inference pending**. Artifact `QuantStream_Run005_Runtime_Probe_20261113T180032Z_y__qpkpj.zip`, SHA-256 `547404c9cc40228e0578b54726a506f8a54e183cd476a0aa1f3cb21c5eb20347`. Tóm tắt kiểm tra: [verification.json](../../reports/experiments/run-005-runtime-probe/verification.json). Chưa có output forecast Run 005 được review.

## Những gì đã xác minh

- Traceback dừng tại `importlib.metadata.version('numpy')`, trả `2.1.3`, khác giá trị `1.26.4` trong lock Run 004. Lỗi xảy ra trước lệnh pip cài/gỡ package và trước runner inference trong notebook.
- Lock đã review thực sự ghi Python `3.13.16` và NumPy `1.26.4`. [Release notes NumPy 1.26.4](https://numpy.org/doc/stable/release/1.26.4-notes.html) chỉ nêu hỗ trợ Python 3.9–3.12; [NumPy 2.1](https://numpy.org/doc/stable/release/2.1.0-notes.html) bổ sung Python 3.13.
- ZIP xác nhận có hai metadata NumPy, theo thứ tự `/usr/local/lib/python3.13/dist-packages`: **2.1.3**, rồi `/usr/lib/python3/dist-packages`: **1.26.4**. Lock cũ lấy giá trị cuối trong dict comprehension; `metadata.version` của cell setup lấy match đầu.
- Cả kernel và subprocess sạch đều import NumPy **2.1.3** từ `/usr/local/.../numpy/__init__.py`. NumPy dot và PyTorch↔NumPy bridge đều trả tổng 2.0. Không có import error của bảy module được probe.
- Hai inventory đều có 937 tên package. Chênh lệch duy nhất so với lock: thiếu `chronos-forecasting==2.3.2`, và có thêm `torchao==0.10.0`. Python 3.13.16, Torch 2.11.0+cu130, CUDA 13.0, cuDNN 92700 và Tesla T4/capability 7.5 khớp các trường đã ghi.
- Run 004 không lưu `numpy.__version__` và đường dẫn module đã import. Vì vậy lock metadata chưa đủ để xác định NumPy thực tế đã thực thi ở Run 004.

Đã xác định bootstrap đọc metadata NumPy không cùng quy tắc với lock. Điều này không chứng minh Run 004 đã import NumPy 1.26.4 hay 2.1.3: loaded version/path lịch sử không được ghi. Numerical/adapter audits Run 004 vẫn giữ nguyên; metadata identity không được gọi là bằng chứng tái tạo loaded-module identity lịch sử.

## Quyết định xử lý

Giữ nguyên lock, manifest và tất cả điểm số lịch sử. Đăng ký bootstrap revision **`run-005-inference-v1-runtime-fix-001`**, giữ nguyên checkpoint/adapter/data, pass order, tolerance và ngân sách 0 optimizer steps. Chưa có inference nào hoàn tất trước revision này.

Notebook 07 Retry001 giữ nguyên NumPy/PyTorch và các package nền; không tạo constraint file từ toàn bộ last-wins inventory. Bootstrap chỉ được gỡ `torchao==0.10.0` và cài Chronos 2.3.2 còn thiếu bằng `--no-deps --only-binary=:all:`. Mọi drift khác đều dừng trước pip. Sau sửa phải khớp toàn bộ legacy inventory, đồng thời xác minh first-match core pins, dependency closure của stack research, loaded module versions của probe và GPU/runtime. `pip check` toàn Kaggle có các lỗi ứng dụng khác; không tự cập nhật chúng.

Thử một PEFT forward trên Linear CPU tổng hợp, freeze parameters và inference mode; không backward, fit hoặc tải checkpoint. Sau đó runner vẫn kiểm tra full environment cũ và thêm `effective-runtime.json`/plan cho module thật trong kernel trước checkpoint. ZIP giữ `setup/setup.json` với hành động pip, metadata trước/sau và các kiểm tra.

Thư mục mới: `/kaggle/working/quantstream_run005_retry001`; output **`QuantStream_Run005_Retry001_Output.zip`**, giữ nguyên ZIP lỗi cũ. Import Notebook 07 cập nhật, session mới, Tesla T4/Internet và hai input cũ; Run All một lần rồi gửi ZIP. Bản sửa đã qua local checks, chưa được thực thi với full pretrained model trên Kaggle.

[Notebook 07b](../../notebooks/07b_Kaggle_Runtime_Probe.ipynb) vẫn có thể dùng nếu phát sinh drift mới; không cần chạy lại probe đã review. Notebook không cần snapshot/Run 004 ZIP hoặc Internet, không cài/gỡ package, không tải model, không fit, không đọc/xóa output cũ. Nó ghi:

1. Python executable/version, sys.path và các module đã nạp trong kernel.
2. Tất cả distribution metadata candidates và duplicates, cả first-match query và legacy last-wins inventory.
3. Phiên bản và file của NumPy/pandas/PyArrow/SciPy/sklearn/Matplotlib/PyTorch từ subprocess sạch; NumPy dot và PyTorch↔NumPy bridge.
4. `pip check`, NVIDIA driver/GPU nếu khả dụng, stdout/stderr/import errors. Probe hoàn tất không có nghĩa mọi package đã tương thích.

ZIP dùng tên riêng `QuantStream_Run005_Runtime_Probe_<UTC>_<unique>.zip`; giữ nguyên `QuantStream_Run005_Output.zip` thất bại. Cần tải rồi **đính kèm trực tiếp** probe ZIP để review; link `/kaggle/working` không phải file đã được chuyển sang cuộc trò chuyện.

## Nếu còn drift sau sửa

- NumPy duplicate đã được đối chiếu cụ thể bằng cả version và đường dẫn; duplicate/drift khác chưa được cho phép. Không chỉnh số trong lock để làm phép so sánh pass.
- Nếu runtime thực sự khác, cần protocol revision tách biệt cho diagnostic dưới môi trường mới, ghi đầy đủ drift và kiểm soát mode/RNG trong cùng môi trường. Không gọi đó là đã khôi phục lịch sử hoặc tái lập chính xác Run 004.
- Nếu có lỗi binary/ABI, chọn môi trường tương thích dựa trên error và phiên bản thực tế. Không tải model hoặc chạy inference trước khi kiểm tra nền thành công.

## Kiểm tra local

Probe được kiểm tra giữ đủ duplicate metadata, lưu import/dependency errors và tạo archive riêng không đè output trước. Regression tests bootstrap mô phỏng đúng hai sửa đã review, xác nhận không pip cài NumPy/PyTorch, từ chối drift ngoài plan, phân biệt imported NumPy 2.1.3 với legacy metadata 1.26.4 và kiểm tra dependency closure. CPU PEFT smoke và eight-pass Chronos control dùng tiny model tổng hợp, không có model download/fit; không thay thế lần chạy Kaggle đang chờ.
