# Nhật Ký Quyết Định Tiền Xử Lý Dữ Liệu (Preprocessing Decisions)

Tài liệu này ghi lại các quyết định xử lý dữ liệu ở giai đoạn tiền xử lý, dựa trên kết quả phân tích khám phá dữ liệu (EDA) tại `reports/eda/data_issues.md`.

## 1. Loại bỏ dòng trùng lặp
- **Vấn đề**: Dữ liệu có 35 dòng trùng lặp hoàn toàn trên toàn bộ 24 cột (bao gồm cả nhãn).
- **Bằng chứng từ EDA**: Mục 5 trong EDA cho thấy có 35 nhóm bản ghi trùng lặp chính xác (0.12% dữ liệu).
- **Quyết định**: Loại bỏ các dòng trùng lặp bằng cách giữ lại bản ghi đầu tiên của mỗi nhóm trước khi chia dữ liệu (drop duplicates). Tổng số dòng còn lại sau khi loại bỏ là 29.965 dòng.
- **Lý do**: Loại bỏ trước khi chia dữ liệu (train/val/test) nhằm đảm bảo không có rò rỉ dữ liệu (data leakage) do bản sao nằm rải rác ở cả hai tập huấn luyện và đánh giá.

## 2. Gộp mã ngoài mô tả cho `education` và `marriage`
- **Vấn đề**: `education` chứa các mã 0, 5, 6 và `marriage` chứa mã 0 không có trong mô tả gốc.
- **Bằng chứng từ EDA**: 
  - `education`: Mã 0 (0.00% vỡ nợ), mã 5 (6.43%), mã 6 (15.69%). Nhóm "khác" (4) có tỷ lệ 5.69%. 
  - `marriage`: Mã 0 có tỷ lệ vỡ nợ 9.26%, trong khi nhóm "khác" (3) có 26.01%.
- **Quyết định**: Gộp mã 0, 5, 6 của `education` vào nhóm 4 ("khác"). Gộp mã 0 của `marriage` vào nhóm 3 ("khác").
- **Lý do**: Dù EDA cho thấy có sự chênh lệch tỷ lệ vỡ nợ, nhóm các mã này có kích thước mẫu quá nhỏ (ví dụ mã 0 của education chỉ có 14 dòng). Gộp vào nhóm "khác" giúp đơn giản hóa phân phối và gom nhóm các thông tin không xác định.

## 3. Xử lý mã -2 và 0 cho các cột `pay_*`
- **Vấn đề**: Các cột lịch sử thanh toán `pay_sep` đến `pay_apr` chứa mã -2 và 0.
- **Bằng chứng từ EDA**: Các mã -2, -1, 0 chiếm tỷ trọng cực kỳ lớn (hơn 10-20% cho mỗi mã). Các biến này có tương quan lớn nhất với nhãn.
- **Quyết định**: Giữ nguyên các mã -2 và 0, không xóa và không gộp mù quáng. Khi tính các biến mới (như số tháng trễ, mức trễ), chỉ coi các giá trị > 0 là có trễ hạn (mã -2, -1, 0 được coi là không trễ).
- **Lý do**: Đây có thể là những trạng thái thanh toán bình thường nhưng không được liệt kê (ví dụ: 0 là nợ quay vòng đang trả, -2 là không có giao dịch/dư nợ). Do số lượng lớn và có ý nghĩa thống kê cao, việc xóa hoặc gộp sẽ làm hỏng cấu trúc dữ liệu và đánh mất tín hiệu.

## 4. Xử lý giá trị thiếu (Missing values)
- **Vấn đề**: Dữ liệu có thể chứa giá trị thiếu (NaN).
- **Bằng chứng từ EDA**: Không phát hiện giá trị NaN.
- **Quyết định**: Không thực hiện gán giá trị thiếu (imputation). Bổ sung hàm kiểm tra (assert) ở cuối pipeline để báo lỗi nếu dữ liệu mới có chứa NaN.
- **Lý do**: Không cần can thiệp nếu dữ liệu không có giá trị bị khuyết.

## 5. Giá trị âm ở `bill_amt_*` và ngoại lai tiền tệ
- **Vấn đề**: Nhiều biến sao kê `bill_amt_*` có giá trị âm. Đặc trưng tiền tệ có đuôi lệch phải dài (ngoại lai).
- **Bằng chứng từ EDA**: EDA ghi nhận có từ 590-688 dòng mỗi tháng mang sao kê âm. Các biến tiền tệ có giá trị max rất lớn (cực trị dương).
- **Quyết định**: 
  - Giữ nguyên các số tiền âm cho `bill_amt_*`, đồng thời tạo thêm 1 biến cờ (flag) chung `has_negative_bill`. 
  - Áp dụng kỹ thuật Winsorize (cắt ở phân vị 1% và 99%) cho toàn bộ biến tiền tệ (limit_bal, bill_amt_*, pay_amt_*, avg_bill_amt, avg_pay_amt).
  - Ngưỡng Winsorize (p1, p99) được tính chỉ dựa trên tập `train`.
- **Lý do**: Giá trị âm có thể do khách hàng trả dư hoặc được hoàn tiền (nghiệp vụ hợp lệ). Việc Winsorize giúp kiềm chế các giá trị quá cực đoan, bảo vệ độ ổn định của các mô hình học máy (nhất là nhóm mô hình tuyến tính).

## 6. Tạo đặc trưng mới: Tỷ lệ trả nợ (pay_amt / bill_amt)
- **Vấn đề**: Tính tỷ lệ số tiền đã trả trên tổng sao kê, nhưng mẫu số (`bill_amt`) có thể bằng 0 hoặc âm.
- **Quyết định**: Đặt tỷ lệ này là `1.0` nếu `bill_amt <= 0`, ngược lại là `pay_amt / bill_amt`.
- **Lý do**: Khách hàng có số dư sao kê bằng 0 hoặc âm đồng nghĩa với việc không có khoản nợ nào trong tháng đó, nên trạng thái thanh toán được xem như hoàn hảo (100% trả nợ). Ngăn chặn lỗi chia cho 0 và giá trị vô cực.

## 7. Mất cân bằng lớp (Class Imbalance)
- **Vấn đề**: Lớp thiểu số (vỡ nợ) chiếm khoảng 22%.
- **Bằng chứng từ EDA**: Dữ liệu mất cân bằng nhẹ đến trung bình.
- **Quyết định**: Tính toán trọng số lớp (`class_weight`) dựa trên thuật toán `balanced` của scikit-learn từ tập train. Lưu trọng số này vào một file JSON.
- **Lý do**: Việc sử dụng Class Weight hỗ trợ trực tiếp từ thuật toán sẽ bảo toàn chất lượng dữ liệu tốt hơn là dùng SMOTE (tránh rủi ro tạo điểm nhiễu và dễ ứng dụng trong thực tế).

## 8. Tạo các đặc trưng về trễ hạn
- **Quyết định**: Từ các cột `pay_*`, tính số tháng trễ (đếm số tháng có giá trị > 0), mức trễ tối đa `max_delay` (giá trị lớn nhất trong sáu tháng), trung bình mức trễ `avg_delay` và xu hướng trễ `delay_trend` ([công thức, ví dụ: chênh lệch giữa mức trễ các tháng gần nhất và các tháng xa hơn]).
- **Lý do**: Các biến này tóm tắt hành vi thanh toán trong sáu tháng, nên có tín hiệu mạnh hơn từng cột `pay_*` riêng lẻ. Kiểm tra sau xử lý cho thấy `max_delay` và `pay_sep` có tương quan Spearman dương với nhãn (0,378 và 0,291).

## 9. Tạo các đặc trưng về hóa đơn, số tiền trả và hạn mức
- **Quyết định**: Tính `avg_bill_amt`, `avg_pay_amt` (trung bình sáu tháng), tỷ lệ sử dụng hạn mức ([công thức, ví dụ: trung bình hóa đơn chia cho `limit_bal`, đặt 0 khi `limit_bal <= 0`]), nhóm tuổi `age_group` ([các mốc tuổi]) và nhóm hạn mức `limit_group` ([số nhóm], mốc phân vị tính chỉ từ train).
- **Lý do**: Tỷ lệ sử dụng hạn mức cho biết mức độ phụ thuộc vào tín dụng, còn các biến nhóm giúp mô hình bắt được quan hệ phi tuyến theo tuổi và hạn mức.

## 10. Mã hóa biến phân loại và loại bỏ giới tính
- **Quyết định**: Biến danh nghĩa (`education`, `marriage`) mã hóa one-hot với encoder chỉ fit trên train và `handle_unknown="ignore"`; biến có thứ tự (`age_group`, `limit_group`) giữ mã số theo thứ tự tăng dần. Cột `meta_gender` giữ lại trong bảng dữ liệu nhưng không đưa vào đầu vào mô hình.
- **Lý do**: One-hot tránh áp đặt thứ tự giả cho biến danh nghĩa. Loại giới tính khỏi đầu vào để mô hình không dự đoán dựa trên đặc điểm nhạy cảm, đồng thời vẫn có thể dùng cột này để kiểm tra độ công bằng sau khi huấn luyện.