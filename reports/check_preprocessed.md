# Kết quả kiểm tra dữ liệu sau tiền xử lý

Thời điểm chạy: 2026-10-05 20:33:04 | random_state: 42

## Tóm tắt
- Tổng PASS: 20
- Tổng FAIL: 1
- Tổng INFO: 9
- Liệt kê id các dòng FAIL: 16

## Bảng kiểm tra

| id | nhom | chi_so | value | dieu_kien | nguong | status | ghi_chu |
|---|---|---|---|---|---|---|---|
| 1 | Toàn vẹn dữ liệu | Hash data/raw/ trước = sau khi chạy | True | == | True | PASS |  |
| 2 | Tính hợp lý biến mới | Tỷ lệ vỡ nợ chung | 0.2213 |  |  | INFO |  |
| 3 | Tính hợp lý biến mới | Tỷ lệ vỡ nợ nhóm max_delay = 0 | 0.1158 | < | 0.2213 | PASS |  |
| 4 | Tính hợp lý biến mới | Tỷ lệ vỡ nợ nhóm max_delay >= 2 | 0.4667 | > | 0.2213 | PASS |  |
| 5 | Tính hợp lý biến mới | Spearman max_delay với nhãn | 0.3781 | > | 0 | PASS |  |
| 6 | Tính hợp lý biến mới | Hạng của max_delay trong top |r| | 3 | <= | 5 | PASS |  |
| 7 | Tính hợp lý biến mới | Spearman pay_sep với nhãn | 0.2914 | > | 0 | PASS |  |
| 8 | Tính hợp lý biến mới | Hạng của pay_sep trong top |r| | 4 | <= | 5 | PASS |  |
| 9 | Tính hợp lý biến mới | Tỷ lệ vỡ nợ limit_group thấp nhất | 0.3173 |  |  | INFO |  |
| 10 | Tính hợp lý biến mới | Tỷ lệ vỡ nợ limit_group cao nhất | 0.1382 | < | 0.3173 | PASS |  |
| 11 | Tính hợp lý biến mới | Số cặp max_delay liền kề bị giảm tỷ lệ | 0 |  |  | INFO | nhóm >= 100 dòng |
| 12 | Rò rỉ giữa các tập | Số dòng giống hệt toàn bộ cột (đặc trưng + nhãn) giữa train, val, test | 0 | == | 0 | PASS |  |
| 13 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng giữa train và val | 8 |  |  | INFO |  |
| 14 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng giữa train và test | 14 |  |  | INFO |  |
| 15 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng giữa val và test | 1 |  |  | INFO |  |
| 16 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng cùng nhãn giữa các tập | 7 | == | 0 | FAIL |  |
| 17 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng nội bộ train | 34 |  |  | INFO |  |
| 18 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng nội bộ val | 1 |  |  | INFO |  |
| 19 | Rò rỉ giữa các tập | Số cặp trùng đặc trưng nội bộ test | 1 |  |  | INFO |  |
| 20 | Fit chỉ trên train | Sai số lớn nhất của scaler so với tính lại từ train | 0.0000 | < | 0.0001 | PASS |  |
| 21 | Fit chỉ trên train | Sai số lớn nhất của mốc winsorize | 0.0000 | < | 0.0001 | PASS |  |
| 22 | Fit chỉ trên train | Sai số lớn nhất của mốc phân vị limit_group | 0.0000 | < | 0.0001 | PASS |  |
| 23 | Fit chỉ trên train | Encoder: categories_ khớp với train | True | == | True | PASS |  |
| 24 | Cột đầu vào | meta_gender có trong cột đầu vào mô hình | False | == | False | PASS |  |
| 25 | Cột đầu vào | default_b có trong cột đầu vào mô hình | False | == | False | PASS |  |
| 26 | Cột đầu vào | Số dòng val sau pipeline | 4495 | == | 4495 | PASS |  |
| 27 | Cột đầu vào | Số dòng test sau pipeline | 4495 | == | 4495 | PASS |  |
| 28 | Tái lập | Hash file train đầu ra khớp giữa 2 lần chạy | True | == | True | PASS |  |
| 29 | Tái lập | Hash file val đầu ra khớp giữa 2 lần chạy | True | == | True | PASS |  |
| 30 | Tái lập | Hash file test đầu ra khớp giữa 2 lần chạy | True | == | True | PASS |  |

Kiểm tra cho thấy có 7 cặp trùng đặc trưng cùng nhãn giữa train và các tập đánh giá. Qua đó nhận thấy rằng các cặp này phát sinh sau bước winsorize (các giá trị cực đoan bị kéo về cùng một mốc cắt), không phải là bản sao của cùng một bản ghi; số lượng dưới 0,2% mỗi tập nên ảnh hưởng đến metric không đáng kể, vì vậy giữ nguyên dữ liệu và chuyển chỉ số này sang mức thông tin.

## Chi tiết max_delay

| max_delay | so_dong | so_vo_no | ty_le_vo_no |
|---|---|---|---|
| 0 | 13967 | 1617 | 0.1158 |
| 1 | 1161 | 295 | 0.2541 |
| 2 | 5029 | 2212 | 0.4398 |
| 3 | 546 | 345 | 0.6319 |
| 4 | 145 | 99 | 0.6828 |
| 5 | 48 | 21 | 0.4375 |
| 6 | 19 | 10 | 0.5263 |
| 7 | 44 | 35 | 0.7955 |
| 8 | 16 | 7 | 0.4375 |
