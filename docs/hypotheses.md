# Danh sách giả thuyết phân tích

Dựa trên kết quả EDA và các biến đặc trưng mới được tạo ra, dưới đây là danh sách các giả thuyết cần được kiểm chứng trên tập `train.csv`.

| Mã | Giả thuyết (Hypothesis) | Biến liên quan | Hướng dự kiến | Kiểm định đề xuất |
|---|---|---|---|---|
| **H1** | Mức độ trễ hạn càng nghiêm trọng (số tháng trễ và mức trễ tối đa) thì tỷ lệ vỡ nợ càng cao. | `num_delayed_months`, `max_delay` | Đồng biến | Mann-Whitney U |
| **H2** | Khách hàng thuộc nhóm hạn mức tín dụng càng thấp thì tỷ lệ vỡ nợ càng cao. | `limit_group` | Nghịch biến | Chi-square test & Cramér's V |
| **H3** | Tỷ lệ vỡ nợ có sự khác biệt đáng kể giữa các nhóm tuổi. | `age_group` | Có sự khác biệt | Chi-square test & Cramér's V |
| **H4** | Tỷ lệ vỡ nợ có sự khác biệt đáng kể giữa các trình độ học vấn. | `education` | Có sự khác biệt | Chi-square test & Cramér's V |
| **H5** | Tỷ lệ vỡ nợ có sự khác biệt đáng kể giữa các tình trạng hôn nhân. | `marriage` | Có sự khác biệt | Chi-square test & Cramér's V |
| **H6** | Trạng thái thanh toán của tháng gần nhất có sức mạnh liên hệ với vỡ nợ mạnh hơn so với các tháng trong quá khứ. | `pay_sep` đến `pay_apr` | Cramér's V của tháng gần nhất sẽ cao nhất | Chi-square test & Cramér's V so sánh chéo |
| **H7** | Tỷ lệ sử dụng hạn mức tín dụng càng cao thì rủi ro vỡ nợ càng lớn. | `bill_ratio_sep` | Đồng biến | Mann-Whitney U |
| **H8** | Tỷ lệ thực trả trên hóa đơn dư nợ càng thấp thì rủi ro vỡ nợ càng cao. | `pay_ratio_sep` | Nghịch biến | Mann-Whitney U |
| **H9** | Khách hàng có xu hướng trễ hạn ngày càng tăng (dấu hiệu xấu đi) sẽ có tỷ lệ vỡ nợ cao hơn. | `delay_trend` | Đồng biến | Mann-Whitney U |
| **H10** | Số tháng không có giao dịch thanh toán càng nhiều thì rủi ro vỡ nợ càng cao. | `num_zero_pay` | Đồng biến | Chi-square test & Cramér's V |

*Lưu ý: Do chạy nhiều kiểm định cùng lúc, các p-value sẽ được hiệu chỉnh bằng phương pháp FDR (Benjamini-Hochberg) để hạn chế sai lầm loại I (False Discovery Rate).*
