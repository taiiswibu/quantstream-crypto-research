# Quy trình Git

Commit đầu nhập bản MVP đã có trước khi theo dõi bằng Git. Các commit sau là thay đổi thật: kiểm tra snapshot, CI, Git SHA cho training và tài liệu. Không tạo lại ngày làm việc cũ.

Xem lịch sử bằng `git log --oneline --graph --all`; xem một thay đổi bằng `git show <SHA>`. Chưa push thì người khác chưa xem được commit trên GitHub.

## Xuất bản lần đầu

Tạo repo tên gợi ý `quantstream-crypto-research` trong tài khoản `taiiswibu`. Chọn Public nếu muốn nhà tuyển dụng xem. Để repo trống: không chọn thêm README, .gitignore hoặc license.

```powershell
git remote add origin https://github.com/taiiswibu/quantstream-crypto-research.git
git push -u origin main
git push origin --tags
```

Nếu có origin rồi, xem `git remote -v` trước; không force-push repo cũ không liên quan. Git có thể yêu cầu đăng nhập trên trình duyệt.

Có thể dùng helper thay cho ba lệnh xuất bản sau khi đã tạo repo trống:

```powershell
powershell -File scripts/publish_github.ps1 -RepoUrl https://github.com/taiiswibu/quantstream-crypto-research.git
```

Helper yêu cầu branch main sạch, không thay origin khác và không force-push. Nếu chính sách PowerShell trên máy không cho chạy script, dùng các lệnh Git trực tiếp bên trên; không cần đổi chính sách hệ thống. Script được cung cấp cho Windows, chưa chạy trên Windows trong môi trường tạo gói.

Gói Git-ready có cả lịch sử. Có thể áp dụng source lên thư mục MVP hiện tại nếu nó CHƯA có Git; các dữ liệu/model trong đường dẫn ignored được giữ. Nếu thư mục đã có Git, dùng thư mục mới thay vì chép đè lịch sử cũ.

## Một phần việc thật

```powershell
git switch -c feat/kaggle-tcn-run-001
# Sửa cấu hình/code, tạo hypothesis record thật trước khi train.
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check quantstream app.py tests scripts
git status --short
git diff
git add config.json docs/experiments/run-001.md
git diff --staged
git commit -m "feat: define first Kaggle TCN experiment"
git push -u origin feat/kaggle-tcn-run-001
```

Chỉ add các file bạn đã tạo/sửa. Sau commit sạch, xuất input Kaggle; manifest ghi SHA đó. Khi có kết quả thật, commit report và quyết định keep/reject/inconclusive. Mở PR vào main để người xem theo dõi hypothesis, diff và kiểm tra.

| Prefix | Công việc |
|---|---|
| feat | Khả năng/feature mới |
| fix | Sửa lỗi tái hiện được |
| test | Kiểm tra hành vi có rủi ro |
| docs | Hướng dẫn hoặc kết luận |
| ci | Kiểm tra tự động |
| refactor | Đổi cấu trúc, giữ hành vi |
| chore | Bảo trì repo |

Commit sau mỗi phần việc hoàn chỉnh, không cần commit theo giờ. Khi nhờ sửa tiếp, gửi link repo và branch đang dùng để việc sửa nối tiếp lịch sử đã push.
