# Khuyến Nghị Chính Sách Quản Trị Rủi Ro Tín Dụng

Tài liệu này trình bày các khuyến nghị ứng dụng mô hình LightGBM (ngưỡng 0.3763) vào thực tiễn quản trị rủi ro của ngân hàng. Dựa trên phân tích, dự báo và mô phỏng chính sách (Validation set).

## 1. Kết Quả Mô Phỏng Chính Sách

Bảng dưới đây trình bày 3 kịch bản đánh đổi giữa chi phí bỏ sót khách hàng vỡ nợ (False Negative) và chi phí cảnh báo nhầm khách hàng tốt (False Positive).

| Kịch Bản | Giả Định Chi Phí (FN : FP) | Ngưỡng Tối Ưu | Chi Phí Kỳ Vọng | Tỷ lệ bắt vỡ nợ (Recall) | Precision |
|----------|---------------------------|---------------|-----------------|--------------------------|-----------|
| **Cơ sở** (Khuyến nghị)| 5 : 1                     | **0.3763**    | **2475**        | **79.30%**               | 35.32%    |
| Mở rộng (Ưu tiên Sales)| 3 : 1                     | 0.5643        | 1824            | 56.28%                   | 51.90%    |
| Thắt chặt | 10 : 1                    | 0.2575        | 3089            | 92.86%                   | 27.97%    |

**Khuyến nghị:**
Ngân hàng nên áp dụng kịch bản **Cơ sở (Ngưỡng 0.3763)**, giúp bắt được ~79% các trường hợp vỡ nợ thực tế. Mặc dù Precision chỉ ở mức 35% (tức là 100 ca cảnh báo thì có 35 ca thực sự vỡ nợ, 65 ca nhầm), nhưng do chi phí bỏ sót một khoản nợ xấu lớn gấp 5 lần so với việc chăm sóc/giám sát dư thừa, đây là điểm tối ưu hóa tổng chi phí rủi ro cho ngân hàng.

---

## 2. Các Nhóm Khách Hàng Cảnh Báo Sớm 

Thông qua phân tích dữ liệu lịch sử, chúng tôi nhận diện được 3 nhóm rủi ro cao, có thể phát hiện ngay từ hành vi thanh toán, mà chưa cần chạy toàn bộ mô hình:

| Nhóm | Dấu Hiệu Nhận Diện | Tỷ Lệ Vỡ Nợ | Hành Động Khuyến Nghị |
|:---:|:---|:---:|:---|
| **EW-1** | **Bắt đầu trễ hạn (Tháng 9)**<br>Chỉ báo `pay_sep = 1` | **34.10%** | Nhắc nhở tự động, tư vấn trả nợ. Tránh để chuyển sang nợ nhóm 2. |
| **EW-2** | **Xu hướng trễ hạn tăng liên tiếp**<br>`delay_trend > 0.143` | **37.36%** | Hạn chế tạm thời việc tăng hạn mức. Chủ động gọi điện tư vấn tái cơ cấu nợ. |
| **EW-3** | **Hạn mức cấp cực thấp**<br>`limit_bal <= 50,000` | **31.73%** | Theo dõi sát sao hàng tuần (Review ưu tiên). Hỗ trợ thanh toán tối thiểu. |

*(Tỷ lệ vỡ nợ cơ sở chung của tập dữ liệu là khoảng 22.12%)*

---

## 3. Quy Trình Ứng Dụng Mô Hình 

Thay vì đối xử công bằng với mọi khách hàng, mô hình phân chia khách hàng thành 4 dải rủi ro để tối ưu chi phí vận hành:

1. **Rủi ro Thấp (Xác suất < 20%):** Không cần can thiệp. Duy trì hạn mức hiện tại hoặc xem xét up-sell.
2. **Rủi ro Trung Bình (20% - 37.63%):** Giám sát thường kỳ. Gửi email/SMS nhắc nhở sát ngày đến hạn thanh toán.
3. **Rủi ro Cao (37.63% - 60%):** Kích hoạt cảnh báo sớm. Giảm nhẹ hoặc đóng băng không cho tăng hạn mức. Nhân viên CSKH gọi điện nhắc nhở.
4. **Rủi ro Rất Cao (Xác suất > 60%):** Hành động can thiệp ngay lập tức. Cắt giảm hạn mức tạm thời để tránh nợ phình to. Tư vấn chuyển đổi dư nợ sang dạng vay trả góp với lãi suất phù hợp để hỗ trợ khách hàng.

---

## 4. Rủi Ro Tính Công Bằng   
Mô hình đã được **loại bỏ hoàn toàn biến Giới Tính (`SEX`)** trong quá trình huấn luyện nhằm tránh rủi ro phân biệt đối xử.   
Ngân hàng có thể yên tâm sử dụng kết quả dự đoán mà không vi phạm các tiêu chuẩn đạo đức hoặc quy định về công bằng trong tín dụng. Tuy nhiên, vẫn nên định kỳ đánh giá lại sự chênh lệch cảnh báo giữa các nhóm tuổi.   
