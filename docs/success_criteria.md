# Tiêu chí Thành công (Success Criteria)

Dựa trên kết quả đánh giá của mô hình Baseline (Logistic Regression) trên tập Validation, chúng ta thiết lập các tiêu chí thành công cho mô hình học máy nâng cao (như LightGBM, XGBoost hoặc Random Forest) ở các bước tiếp theo.

## 1. Kết quả Baseline (Logistic Regression)
Mô hình Logistic Regression kết hợp chuẩn hóa (StandardScaler) và cân bằng lớp (`class_weight='balanced'`) đạt được các chỉ số sau trên tập Validation:

- **AUC-ROC:** 0.7548
- **AUC-PR (Average Precision):** 0.5184
- **Tại ngưỡng mặc định (0.5000):**
  - Precision: 0.4294
  - Recall: 0.6141
  - F1-Score: 0.5054
- **Tại ngưỡng tối ưu chi phí (0.3961):**
  - Precision: 0.3762
  - Recall: 0.7085
  - F1-Score: 0.4915
  - **Chi phí kỳ vọng (Expected Cost):** 2619.00 đơn vị (giả định $C_{FN}=5, C_{FP}=1$)

## 2. Tiêu chí Thành công (Mục tiêu Cải thiện)
Mô hình mạnh hơn (sẽ được tinh chỉnh siêu tham số) phải vượt qua mức cơ sở này. Cụ thể, các tiêu chí đánh giá thành công trên tập Validation (và đối chiếu lại trên tập Test ở bước cuối) được đề xuất như sau:

1.  **Cải thiện Khả năng Phân biệt (Discriminative Power):**
    - **AUC-ROC:** Phải đạt tối thiểu **0.77** (cải thiện > 0.015 so với baseline).
    - **AUC-PR:** Phải đạt tối thiểu **0.54** (cải thiện > 0.02 so với baseline).

2.  **Giảm thiểu Chi phí Rủi ro (Cost Reduction):**
    - **Tổng Chi phí Kỳ vọng (Expected Cost):** Phải **nhỏ hơn 2500 đơn vị** tại ngưỡng tối ưu (giảm ít nhất ~4.5% tổng chi phí kỳ vọng).

3.  **Tính Ổn định và Công bằng:**
    - Không có hiện tượng quá khớp (Overfitting) trầm trọng. AUC-ROC trên tập Train và Validation không chênh lệch quá 5%.
    - Đảm bảo xem xét tính công bằng (Fairness) khi có/không có biến Gender.

*Ghi chú: Việc đánh giá cuối cùng sẽ được thực hiện trên tập Test ĐÚNG MỘT LẦN sau khi đóng băng hoàn toàn mô hình và ngưỡng quyết định.*
