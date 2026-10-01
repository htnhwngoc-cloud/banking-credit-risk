# Banking Credit Risk

## Mục tiêu

Xây dựng quy trình khám phá dữ liệu và mô hình phân loại để dự đoán khách hàng thẻ tín dụng có vỡ nợ trong tháng tiếp theo hay không. Dự án sử dụng dữ liệu khách hàng thẻ tín dụng tại Đài Loan năm 2005.

## Nguồn dữ liệu

- **Tên bộ dữ liệu:** Default of Credit Card Clients
- **Nguồn:** [UCI Machine Learning Repository, id 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Giấy phép:** CC BY 4.0
- **Ngày tải:** 01/10/2026

File dữ liệu thô được đặt tại `data/raw/default_credit_card_clients.csv`. Thư mục `data/raw/` được bỏ qua bởi Git để không đưa dữ liệu nguồn vào repository.

## Cài đặt và chạy

```bash
python -m pip install -r requirements.txt
python src/download_data.py  # chỉ cần khi cần tải lại dữ liệu thô
python src/load_data.py
```

Lệnh `load_data.py` chuẩn hoá tên cột, kiểm tra cấu trúc dữ liệu, in tóm tắt và lưu dữ liệu đã chuẩn hoá tại `data/processed/credit_card_clients_clean.csv`.

Để chạy kiểm tra khám phá ban đầu, mở và chạy `notebooks/00_data_check.ipynb` từ Jupyter.
