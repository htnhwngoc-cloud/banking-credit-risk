# Problem Statement: Dự đoán rủi ro vỡ nợ thẻ tín dụng

## 1. Bối cảnh

Dự án sử dụng bộ dữ liệu **Default of Credit Card Clients** gồm các khách hàng thẻ tín dụng tại Đài Loan trong năm 2005. Dữ liệu chứa thông tin nhân khẩu học, hạn mức tín dụng, lịch sử tình trạng thanh toán, số dư sao kê và số tiền đã thanh toán trong sáu tháng gần nhất.

Mục tiêu là hỗ trợ nhận diện sớm khách hàng có rủi ro không thanh toán đúng hạn, để việc theo dõi và quản trị rủi ro thẻ tín dụng có cơ sở hơn. Đây là bài tập học thuật trên dữ liệu UCI; không dùng trực tiếp để ra quyết định tín dụng thực tế.

## 2. Bài toán

Đây là bài toán **phân loại nhị phân**. Với thông tin của một khách hàng tại thời điểm quan sát, mô hình dự đoán liệu khách hàng đó có **vỡ nợ trong tháng tiếp theo** hay không.

- Biến mục tiêu: `default_next_month` (`1` = vỡ nợ, `0` = không vỡ nợ).
- Biến đầu vào: `limit_bal`, `gender`, `education`, `marriage`, `age`, các trạng thái thanh toán `pay_*`, số dư sao kê `bill_amt_*` và số tiền thanh toán `pay_amt_*`.
- Quy mô sau chuẩn hoá: 30.000 quan sát, 24 cột, không có giá trị thiếu.
- Phân bố mục tiêu: 6.636 ca vỡ nợ (22,12%) và 23.364 ca không vỡ nợ (77,88%).

## 3. Câu hỏi nghiên cứu

1. Những đặc điểm nào liên quan mạnh nhất đến rủi ro vỡ nợ trong tháng tiếp theo?
2. Lịch sử thanh toán trễ ở các tháng gần đây làm thay đổi tỷ lệ vỡ nợ như thế nào?
3. Mô hình có nhận diện được phần đáng kể trong 22,12% khách hàng vỡ nợ mà vẫn hạn chế cảnh báo nhầm hay không?

## 4. Giả thuyết

Các giả thuyết dưới đây lấy từ cột **Kỳ vọng ban đầu** trong data dictionary.

- **H1:** Khách hàng có `limit_bal` cao hơn có tỷ lệ vỡ nợ thấp hơn.
- **H2:** Khách hàng có trình độ học vấn cao hơn có tỷ lệ vỡ nợ thấp hơn.
- **H3:** Khách hàng trẻ tuổi có tỷ lệ vỡ nợ cao hơn.
- **H4:** Giá trị `pay_*` càng lớn (trễ hạn càng lâu), đặc biệt ở các tháng gần nhất, thì xác suất vỡ nợ càng cao.
- **H5:** Số dư sao kê cao so với hạn mức tín dụng đi kèm rủi ro vỡ nợ cao hơn.
- **H6:** Khách hàng thanh toán nhiều so với số dư sao kê có rủi ro vỡ nợ thấp hơn.

Giới tính không có giả thuyết hướng tác động rõ ràng; ảnh hưởng của tình trạng hôn nhân sẽ được khám phá thay vì khẳng định trước.

## 5. Tiêu chí thành công

Do tỷ lệ vỡ nợ thực tế chỉ là **22,12%**, accuracy không phải thước đo chính: một mô hình luôn dự đoán “không vỡ nợ” đã đạt 77,88% accuracy nhưng bỏ sót toàn bộ ca vỡ nợ.

| Chỉ số | Mục tiêu |
| --- | --- |
| AUC-ROC trên tập test | ≥ 0,75 |
| Recall lớp vỡ nợ | ≥ 60% số ca vỡ nợ thực tế (tương ứng nhận diện ít nhất khoảng 60% của nhóm chiếm 22,12%) |
| Precision lớp vỡ nợ | > 22,12%, cao hơn tỷ lệ nền của lớp vỡ nợ |
| Balanced accuracy | ≥ 65% |
| Báo cáo so sánh | So sánh với baseline dự đoán toàn bộ là không vỡ nợ (accuracy 77,88%, recall vỡ nợ 0%) |

Ngưỡng dự đoán cuối cùng sẽ được chọn theo đánh đổi giữa recall và precision, thay vì tối ưu accuracy đơn thuần.

## 6. Phạm vi và hạn chế

- Dữ liệu phản ánh khách hàng thẻ tín dụng tại Đài Loan năm 2005; không đại diện cho khách hàng hoặc chính sách tín dụng hiện tại ở Việt Nam.
- Nhãn chỉ phản ánh kết quả trong **tháng tiếp theo**, không phải toàn bộ vòng đời khoản nợ.
- `education`, `marriage` và `pay_*` có mã ngoài mô tả; cần xử lý và giải thích ở giai đoạn EDA trước khi mô hình hoá.
- Có 35 dòng trùng lặp toàn bộ cột; cần xác minh trước khi quyết định giữ hay loại bỏ.
- Dữ liệu có thuộc tính nhạy cảm như giới tính, hôn nhân và học vấn. Các kết quả cần được kiểm tra thiên lệch, và chỉ phục vụ mục đích học tập.
