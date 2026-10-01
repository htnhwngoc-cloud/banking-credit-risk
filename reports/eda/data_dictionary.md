# Data Dictionary: Default of Credit Card Clients

- **Nguồn:** UCI Machine Learning Repository (dataset id = 350), giấy phép CC BY 4.0.
- **Phạm vi:** khách hàng thẻ tín dụng tại Đài Loan, năm 2005.
- **Quy mô đã kiểm tra:** 30.000 dòng, 24 cột sau chuẩn hoá (23 đầu vào và 1 mục tiêu); không có giá trị thiếu. Có 35 dòng trùng lặp trên toàn bộ 24 cột.
- **Biến mục tiêu:** 6.636 khách hàng vỡ nợ (22,12%) và 23.364 khách hàng không vỡ nợ (77,88%).
- **Đơn vị tiền tệ:** NT$ (đô la Đài Loan).
- **Tên cột:** bản tải qua `ucimlrepo` có thể dùng `X1` ... `X23` và `Y`; bảng dưới dùng tên có nghĩa trong file gốc. Code chuẩn hoá về tên `snake_case`.

| Mã X | Tên cột | Vai trò | Kiểu | Ý nghĩa | Giá trị / Đơn vị | Kỳ vọng ban đầu | Giá trị quan sát thực tế | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ID | Chỉ số dòng | Số nguyên | Mã định danh khách hàng | 1 ... 30.000 | Không có | Không có trong CSV đang dùng; code sẽ loại nếu có | Không đưa vào mô hình |
| X1 | LIMIT_BAL | Đầu vào | Số nguyên | Hạn mức tín dụng, gồm tín dụng cá nhân và gia đình | NT$ | Hạn mức cao sẽ ít rủi ro hơn, tỷ lệ vỡ nợ sẽ thấp hơn | `int64`; 0 giá trị thiếu; 10.000–1.000.000 NT$ | Dự kiến lệch phải, cần kiểm tra ngoại lệ |
| X2 | SEX | Đầu vào | Phân loại | Giới tính | 1 = Nam, 2 = Nữ | Không có sự khác biệt rõ ràng | 1: 11.888; 2: 18.112; 0 giá trị thiếu | Cần cân nhắc công bằng khi diễn giải |
| X3 | EDUCATION | Đầu vào | Phân loại có thứ tự | Trình độ học vấn | 1 = Sau đại học, 2 = Đại học, 3 = Trung học, 4 = Khác | Trình độ học vấn cao, thu nhập ổn định thì tỷ lệ vỡ nợ sẽ thấp hơn | 0: 14; 1: 10.585; 2: 14.030; 3: 4.917; 4: 123; 5: 280; 6: 51 | Có các mã ngoài mô tả: 0, 5, 6 |
| X4 | MARRIAGE | Đầu vào | Phân loại | Tình trạng hôn nhân | 1 = Đã kết hôn, 2 = Độc thân, 3 = Khác | Chưa rõ ảnh hưởng | 0: 54; 1: 13.659; 2: 15.964; 3: 323 | Có mã ngoài mô tả: 0 |
| X5 | AGE | Đầu vào | Số nguyên | Tuổi | Năm | Người trẻ tuổi có tỷ lệ vỡ nợ cao hơn | `int64`; 0 giá trị thiếu; 21–79 tuổi | Kiểm tra dải tuổi và ngoại lai |
| X6 | PAY_0 | Đầu vào | Phân loại có thứ tự | Tình trạng thanh toán tháng gần nhất (tháng 9) | -1 = trả đúng hạn; 1 ... 9 = trễ hạn theo số tháng | Giá trị càng lớn (trễ càng lâu) thì xác suất vỡ nợ càng cao; dự đoán là biến quan trọng nhất | -2: 2.759; -1: 5.686; 0: 14.737; 1: 3.688; 2: 2.667; 3: 322; 4: 76; 5: 26; 6: 11; 7: 9; 8: 19 | Có mã -2 và 0 chưa được mô tả; không thấy mã 9 |
| X7 | PAY_2 | Đầu vào | Phân loại có thứ tự | Tình trạng thanh toán tháng 8 | Tương đương PAY_0 | Tương đương PAY_0, nhưng yếu dần theo độ xa của tháng | -2: 3.782; -1: 6.050; 0: 15.730; 1: 28; 2: 3.927; 3: 326; 4: 99; 5: 25; 6: 12; 7: 20; 8: 1 | Có mã -2 và 0; không thấy mã 9 |
| X8 | PAY_3 | Đầu vào | Phân loại có thứ tự | Tình trạng thanh toán tháng 7 | Tương đương PAY_0 | Tương đương PAY_0, nhưng yếu dần theo độ xa của tháng | -2: 4.085; -1: 5.938; 0: 15.764; 1: 4; 2: 3.819; 3: 240; 4: 76; 5: 21; 6: 23; 7: 27; 8: 3 | Có mã -2 và 0; không thấy mã 9 |
| X9 | PAY_4 | Đầu vào | Phân loại có thứ tự | Tình trạng thanh toán tháng 6 | Tương đương PAY_0 | Tương đương PAY_0, nhưng yếu dần theo độ xa của tháng | -2: 4.348; -1: 5.687; 0: 16.455; 1: 2; 2: 3.159; 3: 180; 4: 69; 5: 35; 6: 5; 7: 58; 8: 2 | Có mã -2 và 0; không thấy mã 9 |
| X10 | PAY_5 | Đầu vào | Phân loại có thứ tự | Tình trạng thanh toán tháng 5 | Tương đương PAY_0 | Tương đương PAY_0, nhưng yếu dần theo độ xa của tháng | -2: 4.546; -1: 5.539; 0: 16.947; 2: 2.626; 3: 178; 4: 84; 5: 17; 6: 4; 7: 58; 8: 1 | Có mã -2 và 0; không có mã 1 hoặc 9 |
| X11 | PAY_6 | Đầu vào | Phân loại có thứ tự | Tình trạng thanh toán tháng 4 | Tương đương PAY_0 | Tương đương PAY_0, nhưng yếu dần theo độ xa của tháng | -2: 4.895; -1: 5.740; 0: 16.286; 2: 2.766; 3: 184; 4: 49; 5: 13; 6: 19; 7: 46; 8: 2 | Có mã -2 và 0; không có mã 1 hoặc 9 |
| X12 ... X17 | BILL_AMT1 ... BILL_AMT6 | Đầu vào | Số nguyên | Số tiền trên sao kê hằng tháng | NT$ | Hoá đơn lớn so với hạn mức sẽ có tỷ lệ rủi ro cao hơn | `int64`; 0 giá trị thiếu ở mỗi cột; min–max toàn nhóm: -339.603 đến 1.664.089 NT$ | Tất cả các tháng có giá trị âm; cần kiểm tra ý nghĩa |
| X18 ... X23 | PAY_AMT1 ... PAY_AMT6 | Đầu vào | Số nguyên | Số tiền khách đã thanh toán trong tháng | NT$ | Trả nhiều so với hoá đơn thì rủi ro thấp hơn | `int64`; 0 giá trị thiếu ở mỗi cột; min–max toàn nhóm: 0 đến 1.684.259 NT$ | Dự kiến lệch phải, kiểm tra tỷ lệ giá trị 0 |
| Y | default_next_month | Biến mục tiêu | Nhị phân | Khách hàng vỡ nợ trong tháng tiếp theo | 1 = vỡ nợ, 0 = không vỡ nợ | Không có | 0: 23.364 (77,88%); 1: 6.636 (22,12%); 0 giá trị thiếu | Lớp vỡ nợ là nhóm thiểu số |

## Giá trị quan sát chung

- Toàn bộ 24 cột có kiểu `int64` và không có giá trị thiếu.
- `load_data.py` đã in 35 dòng trùng lặp toàn bộ cột; chưa loại chúng ở Phase 1 để tránh thay đổi dữ liệu trước khi phân tích nguyên nhân.

## Ánh xạ tên gốc → tên chuẩn

Ánh xạ dưới đây khớp với `RENAME_MAP` trong `src/load_data.py`; khoá được so khớp sau khi chuyển về chữ thường.

| Tên gốc | Tên chuẩn |
| --- | --- |
| `id` | Loại bỏ (`None`) |
| `y`, `default payment next month` | `default_next_month` |
| `limit_bal`, `x1` | `limit_bal` |
| `sex`, `gender`, `x2` | `gender` |
| `education`, `x3` | `education` |
| `marriage`, `x4` | `marriage` |
| `age`, `x5` | `age` |
| `pay_0`, `x6` | `pay_sep` |
| `pay_2`, `x7` | `pay_aug` |
| `pay_3`, `x8` | `pay_jul` |
| `pay_4`, `x9` | `pay_jun` |
| `pay_5`, `x10` | `pay_may` |
| `pay_6`, `x11` | `pay_apr` |
| `bill_amt1`, `x12` | `bill_amt_sep` |
| `bill_amt2`, `x13` | `bill_amt_aug` |
| `bill_amt3`, `x14` | `bill_amt_jul` |
| `bill_amt4`, `x15` | `bill_amt_jun` |
| `bill_amt5`, `x16` | `bill_amt_may` |
| `bill_amt6`, `x17` | `bill_amt_apr` |
| `pay_amt1`, `x18` | `pay_amt_sep` |
| `pay_amt2`, `x19` | `pay_amt_aug` |
| `pay_amt3`, `x20` | `pay_amt_jul` |
| `pay_amt4`, `x21` | `pay_amt_jun` |
| `pay_amt5`, `x22` | `pay_amt_may` |
| `pay_amt6`, `x23` | `pay_amt_apr` |

## Việc cần kiểm tra ở Phase 2

1. Xác định ý nghĩa và cách xử lý các mã ngoài mô tả: `EDUCATION` = 0, 5, 6; `MARRIAGE` = 0; và `PAY_*` = -2, 0.
2. Kiểm tra vì sao các tháng `PAY_5` và `PAY_6` không có mã 1, các tháng khác có rất ít mã 1, và toàn bộ `PAY_*` không có mã 9.
3. Phân tích 35 dòng trùng lặp để quyết định có phải bản ghi lặp thực sự hay không trước khi loại bỏ.
4. Đánh giá mất cân bằng lớp: tỷ lệ vỡ nợ 22,12%, do đó không dùng accuracy đơn thuần để chọn mô hình.
