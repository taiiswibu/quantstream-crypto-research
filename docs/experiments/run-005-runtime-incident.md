# Run 005 — NumPy metadata mismatch trước inference

Status: **reported setup failure; Kaggle diagnostic evidence pending**. Không có output forecast Run 005 được review. Người dùng gửi traceback và link tải trong phiên Kaggle, chưa gửi bytes ZIP có thể đọc tại đây.

## Những gì đã xác minh

- Traceback dừng tại `importlib.metadata.version('numpy')`, trả `2.1.3`, khác giá trị `1.26.4` trong lock Run 004. Lỗi xảy ra trước lệnh pip cài/gỡ package và trước runner inference trong notebook.
- Lock đã review thực sự ghi Python `3.13.16` và NumPy `1.26.4`. [Release notes NumPy 1.26.4](https://numpy.org/doc/stable/release/1.26.4-notes.html) chỉ nêu hỗ trợ Python 3.9–3.12; [NumPy 2.1](https://numpy.org/doc/stable/release/2.1.0-notes.html) bổ sung Python 3.13.
- Lock cũ thu thập metadata bằng dict comprehension trên mọi distribution và giữ giá trị cuối khi trùng tên. Cell setup mới dùng `metadata.version`, có thể trả match khác. Test synthetic tái hiện được hai cách đọc khác nhau trên cùng danh sách metadata.
- Run 004 không lưu `numpy.__version__` và đường dẫn module đã import. Vì vậy lock metadata chưa đủ để xác định NumPy thực tế đã thực thi ở Run 004.

Đây chưa chứng minh môi trường Kaggle đã đổi, metadata đang trùng hoặc Run 004 dùng NumPy phiên bản nào. Không coi metadata version là bằng chứng độc lập của loaded module. Numerical/adapter audits Run 004 vẫn là những phép kiểm tra đã thực hiện; giới hạn tái tạo runtime cần ghi riêng.

## Quyết định xử lý

Giữ nguyên lock, manifest, protocol và tất cả điểm số lịch sử. Không ép cài NumPy 1.26.4 vào Python 3.13, không bỏ guard để gọi đó là tái lập cùng môi trường. Phần setup Notebook 07 được sửa thông báo để chuyển sang kiểm tra runtime, chưa mở khóa inference.

Chạy [Notebook 07b](../../notebooks/07b_Kaggle_Runtime_Probe.ipynb), một Run All, CPU cũng được. Notebook không cần snapshot/Run 004 ZIP hoặc Internet, không cài/gỡ package, không tải model, không fit, không đọc/xóa output cũ. Nó ghi:

1. Python executable/version, sys.path và các module đã nạp trong kernel.
2. Tất cả distribution metadata candidates và duplicates, cả first-match query và legacy last-wins inventory.
3. Phiên bản và file của NumPy/pandas/PyArrow/SciPy/sklearn/Matplotlib/PyTorch từ subprocess sạch; NumPy dot và PyTorch↔NumPy bridge.
4. `pip check`, NVIDIA driver/GPU nếu khả dụng, stdout/stderr/import errors. Probe hoàn tất không có nghĩa mọi package đã tương thích.

ZIP dùng tên riêng `QuantStream_Run005_Runtime_Probe_<UTC>_<unique>.zip`; giữ nguyên `QuantStream_Run005_Output.zip` thất bại. Cần tải rồi **đính kèm trực tiếp** probe ZIP để review; link `/kaggle/working` không phải file đã được chuyển sang cuộc trò chuyện.

## Sau khi có evidence

- Nếu metadata trùng/mâu thuẫn với loaded version, xác minh import paths và cách snapshot inventory trước khi sửa bootstrap. Không chỉnh số trong lock để làm phép so sánh pass.
- Nếu runtime thực sự khác, cần protocol revision tách biệt cho diagnostic dưới môi trường mới, ghi đầy đủ drift và kiểm soát mode/RNG trong cùng môi trường. Không gọi đó là đã khôi phục lịch sử hoặc tái lập chính xác Run 004.
- Nếu có lỗi binary/ABI, chọn môi trường tương thích dựa trên error và phiên bản thực tế. Không tải model hoặc chạy inference trước khi kiểm tra nền thành công.

## Kiểm tra local

Probe được kiểm tra giữ đủ duplicate metadata, lưu import/dependency errors và tạo archive riêng không đè output trước. Notebook schema/Python cells được kiểm tra. Có chạy probe trên môi trường local của agent để xác nhận JSON/ZIP và NumPy↔PyTorch bridge; đó không phải môi trường Kaggle của người dùng và không thay thế ZIP đang chờ.
