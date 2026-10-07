# Báo Cáo Tổng Hợp Modeling

## 1. Mục tiêu và Giả định Chi phí
Bài toán phân loại khách hàng vỡ nợ thẻ tín dụng được tiếp cận dưới góc độ tối ưu hóa chi phí rủi ro:
- **False Negative (Bỏ sót vỡ nợ - Rủi ro cao):** Thiệt hại ước tính gấp 5 lần so với False Positive.
- **False Positive (Cảnh báo nhầm - Rủi ro thấp):** Thiệt hại là mất khách hàng và lợi nhuận kỳ vọng.
- Mục tiêu chính không chỉ là độ chính xác, mà là tìm kiếm ngưỡng quyết định sao cho tổng chi phí kỳ vọng đạt mức thấp nhất.

## 2. Quá trình Huấn luyện
Hai mô hình đã được xây dựng và so sánh trên tập Validation:
1. **Baseline (Logistic Regression):** Đạt AUC-ROC = 0.7548, mức chi phí kỳ vọng tối thiểu là 2619 đơn vị.
2. **Best Model (LightGBM):** Sử dụng `Optuna` để tinh chỉnh.
   - **AUC-ROC:** Đạt ~0.7879.
   - **AUC-PR:** Đạt ~0.55.

## 3. Giải thích Mô hình (SHAP)
Thông qua công cụ SHAP, các yếu tố đóng góp chính vào việc đẩy cao dự đoán vỡ nợ bao gồm:
- **Lịch sử trả nợ (`pay_0`, `delay_trend`):** Khách hàng chậm thanh toán trong các tháng gần nhất là rủi ro lớn nhất. Điều này hoàn toàn khớp với giả thuyết ở Phần A.
- **Tỷ lệ trả nợ / Tỷ lệ sử dụng hạn mức:** Càng trả ít hoặc dùng kiệt hạn mức thì tỷ lệ vỡ nợ càng cao.

## 4. Công bằng giới tính
Mô hình LightGBM được huấn luyện mù giới tính (không sử dụng `SEX`). Đánh giá phân tách theo Nam/Nữ cho thấy:
- Tỷ lệ FNR (Bỏ sót) và FPR (Cảnh báo nhầm) giữa hai giới không có sự chênh lệch đáng kể.
- Mô hình đảm bảo tính công bằng khi áp dụng trong thực tế mà không phân biệt đối xử.

## 5. Đánh Giá Cuối Cùng Trên Tập Test
Sau khi đóng băng ngưỡng quyết định ở mức **0.3763**, mô hình LightGBM đã được áp dụng lên tập Test duy nhất 1 lần. Kết quả đạt được:
- **AUC-ROC:** 0.7862 (Vượt xa mục tiêu 0.77, tính ổn định rất cao so với Validation 0.7879).
- **AUC-PR:** 0.5602 (Vượt mục tiêu 0.54).
- **Tổng chi phí kỳ vọng:** 2515.00 đơn vị (Cải thiện rõ rệt so với mức 2619 của Baseline, sát với mục tiêu tối ưu hóa rủi ro của ngân hàng).
- **Recall:** 0.7827 (Bắt được hơn 78% số ca vỡ nợ thực tế).

**Kết luận chung:** Mô hình LightGBM đã sẵn sàng để triển khai, giúp ngân hàng quản trị rủi ro chủ động với mức tiết kiệm chi phí kỳ vọng vượt trội.
