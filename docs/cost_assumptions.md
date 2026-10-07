# Giả định Chi phí (Cost Assumptions)

Trong bài toán phân loại vỡ nợ tín dụng, mục tiêu kinh doanh không chỉ là có mô hình chính xác nhất (Accuracy cao), mà là giảm thiểu rủi ro và tổn thất tài chính. Ngân hàng đối mặt với hai loại sai lầm chính khi dự đoán khách hàng:

1.  **False Negative (FN - Bỏ sót vỡ nợ):** Mô hình dự đoán khách hàng "Không vỡ nợ", nhưng thực tế họ "Vỡ nợ". 
    *   **Hậu quả:** Ngân hàng cấp tín dụng và đối mặt với rủi ro mất vốn (tiền gốc, chi phí thu hồi nợ, mất thời gian xử lý).
    *   **Mức độ rủi ro:** Rất lớn (gây thiệt hại trực tiếp nặng nề nhất).

2.  **False Positive (FP - Cảnh báo nhầm):** Mô hình dự đoán khách hàng "Vỡ nợ", nhưng thực tế họ "Không vỡ nợ". 
    *   **Hậu quả:** Ngân hàng có thể từ chối cấp tín dụng hoặc giảm hạn mức.
    *   **Mức độ rủi ro:** Vừa phải (mất đi khách hàng tốt, mất cơ hội thu lợi nhuận từ tiền lãi/phí và giảm trải nghiệm khách hàng).

## 1. Kịch bản Cơ sở (Baseline Scenario)
Dựa trên nguyên tắc thiệt hại do mất vốn (FN) luôn lớn hơn nhiều so với việc mất lợi nhuận kỳ vọng từ một khách hàng (FP), chúng ta đặt giả định tỷ lệ chi phí cơ sở là **5:1**.

*   **Chi phí 1 ca FN ($C_{FN}$):** 5 đơn vị.
*   **Chi phí 1 ca FP ($C_{FP}$):** 1 đơn vị.

Mục tiêu của mô hình là tìm ra ngưỡng quyết định (threshold) sao cho **Tổng chi phí kỳ vọng (Expected Cost)** đạt mức tối thiểu:
`Expected Cost = (FN * 5) + (FP * 1)`

## 2. Kịch bản Độ nhạy (Sensitivity Scenarios)
Nhằm đánh giá sự thay đổi và linh hoạt của mô hình dưới các chiến lược/chính sách rủi ro khác nhau của ngân hàng:

*   **Kịch bản Thắt chặt (Conservative - 10:1):** Ưu tiên quản trị rủi ro tuyệt đối, thà giết nhầm còn hơn bỏ sót (thường áp dụng trong giai đoạn kinh tế suy thoái, nợ xấu tăng cao).
    *   Chi phí 1 ca FN: 10 đơn vị.
    *   Chi phí 1 ca FP: 1 đơn vị.
    *   *Chiến lược:* Tối đa hóa Recall. Ngưỡng quyết định sẽ thấp xuống.

*   **Kịch bản Mở rộng (Aggressive - 3:1):** Ưu tiên tăng trưởng dư nợ tín dụng, mở rộng tệp khách hàng. Mức phạt cho việc đánh mất khách hàng tiềm năng cao hơn tương đối.
    *   Chi phí 1 ca FN: 3 đơn vị.
    *   Chi phí 1 ca FP: 1 đơn vị.
    *   *Chiến lược:* Tối đa hóa Precision hơn so với kịch bản cơ sở, ngưỡng quyết định sẽ được đẩy cao lên một chút để tránh chặn nhầm người tốt.
