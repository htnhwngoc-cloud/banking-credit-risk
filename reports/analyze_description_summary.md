# Tổng kết Phase 4 - Phần A: Phân tích mô tả và Kiểm chứng giả thuyết

Dựa trên dữ liệu `train.csv` (14,683 dòng) từ kết quả tiền xử lý (Phase 3), chúng tôi đã tiến hành đánh giá chi tiết đặc trưng và tỷ lệ vỡ nợ, đồng thời kiểm định thống kê cho 10 giả thuyết đã đề ra.

## 1. Kết quả kiểm chứng 10 giả thuyết
Toàn bộ 10 giả thuyết (H1-H10) đề ra ban đầu **đều có ý nghĩa thống kê** (p-value < 0.05 sau hiệu chỉnh đa kiểm định FDR). 

Những điểm nổi bật nhất:
- **(H1) Mức trễ tối đa (`max_delay`)**: Nhóm khách hàng có `max_delay` từ 3 tháng trở lên có rủi ro vỡ nợ vượt trên 60%. Đồng thời `num_delayed_months` cũng là biến phân tách nợ xấu cực kỳ mạnh mẽ.
- **(H2) Hạn mức tín dụng (`limit_group`)**: Khách hàng hạn mức cực thấp (Q1) có tỷ lệ vỡ nợ lên tới hơn 31%, cao gấp đôi so với nhóm cao nhất (Q4).
- **(H6) Mức độ quan trọng của tháng gần nhất (`pay_sep`)**: Trong số các biến gốc, `pay_sep` là biến có sức ảnh hưởng mạnh nhất tới khả năng vỡ nợ (Cramér's V = 0.42). Càng xa hiện tại (như `pay_apr`), sức mạnh phân loại càng yếu đi (Cramér's V = 0.24). 
- **(H8) Tỷ lệ trả nợ (`pay_ratio_sep`)**: Tỷ lệ trả thực tế trên số dư thấp làm gia tăng rõ rệt khả năng vỡ nợ.
- **(H9 & H10) Biến phái sinh (`delay_trend`, `num_zero_pay`)**: Các khách hàng có xu hướng gia tăng mức độ trễ theo thời gian (dấu hiệu suy thoái tín dụng) hoặc có nhiều tháng không đóng một đồng nào đều có tỷ lệ vỡ nợ cao vượt bậc, chứng tỏ các feature engineering mới là chính xác và có giá trị lớn.

## 2. Kiểm tra tương quan và tính chồng chéo (Sơ bộ)
Trong số các đặc trưng mạnh nhất:
- `pay_sep` (Ordinal category), `max_delay` (Continuous/Ordinal), `num_delayed_months` mang thông tin về hành vi trễ hạn, do đó chúng có độ tương quan khá lớn với nhau.
- Các đặc trưng về tiền tệ (`limit_group` hoặc `limit_bal`, `pay_ratio_sep`) đem lại luồng thông tin khác so với nhóm hành vi trễ hạn.

## 3. Đề xuất cho Phase 4 - Phần B (Mô hình hóa)
Qua Part A, chúng ta có danh sách các đặc trưng (features) cốt lõi cần phải đưa vào mô hình hoặc làm trọng tâm cho Feature Selection:
1. Nhóm mạnh nhất: `max_delay`, `pay_sep` (Có thể xem xét tính đa cộng tuyến để giữ lại 1 trong 2, hoặc cho mô hình Tree-based tự xử lý).
2. Nhóm hành vi: `num_delayed_months`, `delay_trend`, `num_zero_pay`.
3. Nhóm nhân khẩu/hạn mức: `limit_bal` (hoặc `limit_group`), `age_group`, `education`, `marriage`.
4. Nhóm tài chính: `pay_ratio_sep`, `bill_ratio_sep`, `avg_bill_amt`.

Đề xuất trong Phần B, chúng ta sẽ áp dụng các mô hình Tree-based (như LightGBM, Random Forest, XGBoost) do các biến quan trọng nhất đều có tính chất là phân loại (categorical), thứ bậc (ordinal) hoặc phân phối không chuẩn. Tree-based models sẽ xử lý tốt sự phi tuyến tính và tương quan giữa các nhóm biến này mà không cần biến đổi log hay loại bỏ quá gắt gao.

*(File notebook chi tiết và toàn bộ biểu đồ, bảng số liệu đã được lưu tại `reports/phase4/analyze/tables` và `reports/phase4/analyze/figures`).*
