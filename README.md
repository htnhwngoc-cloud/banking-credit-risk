# Banking Credit Risk Prediction

Dự án phân tích và dự báo rủi ro vỡ nợ thẻ tín dụng dựa trên bộ dữ liệu [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## Tổng Quan Dự Án
Mục tiêu của dự án là xây dựng một hệ thống từ đầu đến cuối (End-to-End) hỗ trợ quản trị rủi ro tín dụng, bao gồm:
1. **Phân tích dữ liệu (EDA):** Nhận diện hành vi rủi ro từ dữ liệu lịch sử.
2. **Dự báo vỡ nợ (Machine Learning):** Xây dựng mô hình phân loại để dự đoán khách hàng có nguy cơ vỡ nợ tháng tiếp theo.
3. **Mô phỏng chính sách (Policy Simulation):** Tối ưu hóa điểm ngưỡng (threshold) quyết định dựa trên hàm chi phí rủi ro thực tế của ngân hàng.
4. **Ứng dụng (Dashboard):** Triển khai mô hình dưới dạng web app (Streamlit) để nhân viên tín dụng sử dụng và diễn giải kết quả với SHAP.

## Cấu Trúc Thư Mục
```
banking-credit-risk/
│
├── data/
│   ├── raw/                 # Dữ liệu gốc từ UCI
│   └── processed/           # Dữ liệu sau tiền xử lý (train, val, test)
│
├── notebooks/               # Jupyter notebooks (EDA, Khám phá dữ liệu)
├── src/                     # Mã nguồn xử lý dữ liệu (load_data, preprocess)
├── app/                     # Ứng dụng Streamlit Dashboard (streamlit_app.py, features.py)
├── models/                  # Lưu trữ mô hình đã huấn luyện (LightGBM)
├── artifacts/               # Metadata, Scaler, Encoder, Winsorize config
├── reports/                 # Báo cáo phân tích, hình ảnh, kết quả mô phỏng
├── tests/                   # Unit tests cho hệ thống (đặc biệt là Feature Engineering)
└── docs/                    # Tài liệu quyết định thiết kế, tiêu chí thành công
```

## Cài Đặt & Chạy Ứng Dụng

**1. Cài đặt môi trường:**
```bash
python -m venv .venv
.venv\Scripts\activate   # Trên Windows
pip install -r requirements.txt
```

*(Lưu ý: Nếu chưa có lightgbm, streamlit, shap, pytest, vui lòng cài đặt bổ sung qua pip).*

**2. Chạy ứng dụng Dashboard:**
```bash
streamlit run app/streamlit_app.py
```
Giao diện Web sẽ tự động mở lên. Hướng dẫn sử dụng:
- **Tab 1 (Dự Đoán Rủi Ro):** Nhập thông tin khách hàng (hạn mức, độ tuổi, lịch sử thanh toán 6 tháng gần nhất). Bấm **DỰ ĐOÁN** để nhận xác suất vỡ nợ, dải phân loại (Xanh/Vàng/Cam/Đỏ) và đề xuất hành động.
- **Tab 2 (Giải Thích SHAP):** Xem biểu đồ phân tích để hiểu rõ lý do (yếu tố nào làm tăng/giảm rủi ro vỡ nợ của hồ sơ vừa nhập).
- **Tab 3 (Khách Hàng Mẫu):** Cung cấp sẵn các bộ thông số (từ Rủi ro thấp đến Rủi ro rất cao) để bạn thử nghiệm nhanh.

**3. Chạy Unit Tests:**
```bash
pytest tests/ -v
```

## Kết Quả Đạt Được
- **Hiệu suất Mô hình:** Mô hình LightGBM đạt **AUC-ROC 0.7862** và **AUC-PR 0.5602** trên tập Test, vượt tiêu chí thành công đề ra.
- **Tối ưu chi phí:** Tại ngưỡng 0.3763 (được chọn để giảm thiểu Expected Cost với tỷ lệ False Negative Cost : False Positive Cost = 5:1), mô hình nhận diện được ~79% các trường hợp vỡ nợ thực tế.
- **Tính công bằng (Fairness):** Biến Giới tính (`SEX`) đã được loại bỏ hoàn toàn để tuân thủ đạo đức tín dụng.
- **Tài liệu chi tiết:** Vui lòng xem `reports/final_report.md` và `reports/recommendations.md`.

## Tác Giả
- Hồ Thị Như Ngọc
