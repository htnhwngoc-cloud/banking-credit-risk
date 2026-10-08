# Báo Cáo Tổng Hợp Dự Án Dự Báo Rủi Ro Tín Dụng Thẻ (Credit Risk Prediction)

**Dự án:** UCI Default of Credit Card Clients  
**Giai đoạn:** Hoàn thiện và Bàn giao  

## 1. Mục Tiêu Dự Án
Xây dựng một hệ thống phân tích và dự báo khả năng vỡ nợ của khách hàng sử dụng thẻ tín dụng trong tháng tiếp theo. Mục đích nhằm:
1. Xác định các yếu tố, hành vi dẫn đến rủi ro tín dụng.
2. Cung cấp mô hình học máy để cảnh báo sớm.
3. Hỗ trợ ra quyết định thông qua ứng dụng giao diện.

## 2. Dữ Liệu & Tiền Xử Lý
- **Nguồn:** Tập dữ liệu của các ngân hàng Đài Loan (2005) với 30,000 khách hàng.
- **Biến mục tiêu:** `default.payment.next.month` (1: Vỡ nợ, 0: Trả đủ).
- **Tiền xử lý:**
  - Chuyển đổi mã hóa các cột categorical (`EDUCATION`, `MARRIAGE`).
  - Tạo các biến kỹ thuật số học (Feature Engineering) quan trọng: 
    - `bill_ratio` (Tỷ lệ sử dụng hạn mức).
    - `pay_ratio` (Tỷ lệ trả nợ).
    - `delay_trend` (Xu hướng trễ hạn - đường xu hướng tuyến tính 6 tháng).
    - Các biến tổng hợp (Trung bình hóa đơn, trung bình đã trả, số tháng trễ, mức trễ tối đa).
  - Loại bỏ hoàn toàn biến giới tính (`SEX`) để đảm bảo tính công bằng (Fairness).
  - Xử lý các giá trị cực trị (Outliers) bằng kỹ thuật Winsorization dựa trên phân vị (P1-P99) của tập Train.
  - Phân tách dữ liệu: 70% Train, 15% Validation, 15% Test.

## 3. Khám Phá Dữ Liệu (EDA) & Phân Tích 
Phân tích mô tả đã xác nhận mạnh mẽ các giả thuyết kinh doanh:
- Hành vi thanh toán trong tháng gần nhất (`pay_sep`) là chỉ báo quan trọng nhất. Nhóm trễ hạn > 2 tháng có rủi ro vỡ nợ vượt 60-70%.
- Tỷ lệ sử dụng hạn mức (`bill_ratio`) càng cao, rủi ro vỡ nợ càng tăng (đặc biệt khi dùng sát hạn mức).
- Khách hàng có xu hướng trễ hạn kéo dài (`delay_trend` dương) đối diện rủi ro vỡ nợ 37.36%.

## 4. Xây Dựng & Đánh Giá Mô Hình 
Chúng tôi đã chọn và tối ưu mô hình **LightGBM** (sử dụng Optuna) làm mô hình cuối cùng vì tốc độ huấn luyện nhanh, xử lý tốt dữ liệu phi tuyến, và dễ diễn giải (qua SHAP).

### 4.1. Hiệu Suất Mô Hình (Trên tập Test)
Mô hình hoàn thành xuất sắc mục tiêu đặt ra:
- **AUC-ROC:** 0.7862 (Vượt mục tiêu 0.77).
- **AUC-PR:** 0.5602 (Vượt mục tiêu 0.54).
- **Recall (Khả năng bắt ca vỡ nợ):** ~78.27% tại ngưỡng 0.3763.

### 4.2. Tối Ưu Hóa Ngưỡng    
Mô hình đã được tối ưu hóa dựa trên **Chi phí kỳ vọng**, giả định Chi phí bỏ sót (False Negative) = 5, và Chi phí cảnh báo nhầm (False Positive) = 1.
- Ngưỡng tối ưu xác định trên Validation Set là **0.3763**.
- Tổng chi phí kỳ vọng trên tập Test đạt 2515 đơn vị (tốt hơn mức 2619 của mô hình Logistic Regression cơ sở).

### 4.3. Giải Thích Tính Năng (Feature Importance)
Ứng dụng SHAP, mô hình nhận diện 5 yếu tố cốt lõi quyết định rủi ro:
1. `pay_sep`: Trạng thái thanh toán tháng gần nhất.
2. `avg_delay`: Trung bình mức độ trễ hạn trong 6 tháng.
3. `pay_ratio_sep`: Tỷ lệ thanh toán nợ tháng gần nhất.
4. `limit_bal`: Hạn mức tín dụng được cấp.
5. `delay_trend`: Xu hướng trễ hạn.

## 5. Bàn Giao & Ứng Dụng   
- **Dashboard Streamlit:** Xây dựng tại `app/streamlit_app.py`, cung cấp giao diện trực quan cho nhân viên tín dụng nhập thông tin, xem dự báo xác suất, và nhận giải thích lý do (SHAP) của từng quyết định.
- **Khuyến Nghị Chính Sách:** Xác định các nhóm khách hàng cảnh báo sớm và đưa ra phân khúc rủi ro (4 dải: Thấp, Trung Bình, Cao, Rất Cao) cùng các hành động can thiệp tương ứng. Chi tiết xem tại `reports/recommendations.md`.

## 6. Kết Luận
Dự án đã thành công chuyển đổi dữ liệu thô thành một giải pháp phân tích tín dụng có giá trị kinh doanh rõ ràng. Bằng việc kết hợp Feature Engineering sâu sát, mô hình Machine Learning mạnh mẽ, và cách tiếp cận định lượng hóa chi phí rủi ro, hệ thống hoàn toàn sẵn sàng cho các pha triển khai thực tế tiếp theo.