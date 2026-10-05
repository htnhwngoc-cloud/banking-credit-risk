import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

def main():
    # Bước 1: Chuẩn hóa cơ bản
    print("Loading data...")
    df = pd.read_csv('data/raw/default_credit_card_clients.csv')
    
    col_mapping = {
        'X1': 'limit_bal', 'X2': 'gender', 'X3': 'education', 'X4': 'marriage',
        'X5': 'age', 'X6': 'pay_sep', 'X7': 'pay_aug', 'X8': 'pay_jul',
        'X9': 'pay_jun', 'X10': 'pay_may', 'X11': 'pay_apr',
        'X12': 'bill_amt_sep', 'X13': 'bill_amt_aug', 'X14': 'bill_amt_jul',
        'X15': 'bill_amt_jun', 'X16': 'bill_amt_may', 'X17': 'bill_amt_apr',
        'X18': 'pay_amt_sep', 'X19': 'pay_amt_aug', 'X20': 'pay_amt_jul',
        'X21': 'pay_amt_jun', 'X22': 'pay_amt_may', 'X23': 'pay_amt_apr',
        'Y': 'default'
    }
    
    if 'X1' in df.columns:
        df = df.rename(columns=col_mapping)
    
    # Loại bỏ ID nếu có
    for id_col in ['id', 'ID']:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
            
    # Ép kiểu cơ bản
    df['default'] = df['default'].astype(int)
    
    # Bước 2: Loại dòng trùng lặp (trước khi chia dữ liệu)
    initial_len = len(df)
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} duplicate rows.")
    
    # Bước 3: Xử lý mã ngoài mô tả
    df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4})
    df['marriage'] = df['marriage'].replace({0: 3})
    # pay_* mã -2, 0 giữ nguyên, theo EDA
    
    # Bước 4: Kiểm tra giá trị thiếu
    assert df.isnull().sum().sum() == 0, "Dữ liệu có giá trị thiếu!"
    
    # Bước 5: Giá trị âm
    months = ['sep', 'aug', 'jul', 'jun', 'may', 'apr']
    bill_cols = [f'bill_amt_{m}' for m in months]
    df['has_negative_bill'] = (df[bill_cols] < 0).any(axis=1).astype(int)
    
    # Bước 6: Chia dữ liệu (70/15/15)
    print("Splitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['default'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['default'], random_state=42)
    
    def add_features(data, train_quantiles=None):
        d = data.copy()
        
        # 1. Tỷ lệ sử dụng hạn mức
        for m in months:
            d[f'bill_ratio_{m}'] = d[f'bill_amt_{m}'] / d['limit_bal']
            
        # 2. Tỷ lệ trả nợ
        for m in months:
            # Quy tắc nhất quán: nếu sao kê <= 0, tỷ lệ = 1 (tránh chia 0, không nợ => coi như trả đủ)
            d[f'pay_ratio_{m}'] = np.where(
                d[f'bill_amt_{m}'] <= 0, 
                1.0, 
                d[f'pay_amt_{m}'] / d[f'bill_amt_{m}']
            )
            
        # 3. Trễ hạn (chỉ tính > 0 là trễ)
        pay_cols = [f'pay_{m}' for m in months]
        d_pay_pos = d[pay_cols].clip(lower=0)
        
        d['num_delayed_months'] = (d_pay_pos > 0).sum(axis=1)
        d['max_delay'] = d_pay_pos.max(axis=1)
        d['avg_delay'] = d_pay_pos.mean(axis=1)
        
        # 4. Xu hướng trễ hạn (slope = cov(t, y) / var(t))
        # Thời gian: apr=1, may=2, jun=3, jul=4, aug=5, sep=6. Mean_t = 3.5
        # Trọng số: (t - mean_t) => sep: +2.5, ..., apr: -2.5
        t_diff = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5])
        d['delay_trend'] = d[pay_cols].dot(t_diff) / 17.5
        
        # 5. Trung bình
        d['avg_bill_amt'] = d[bill_cols].mean(axis=1)
        pay_amt_cols = [f'pay_amt_{m}' for m in months]
        d['avg_pay_amt'] = d[pay_amt_cols].mean(axis=1)
        
        # 6. Nhóm tuổi (<30, 30-40, 40-50, 50-60, 60+)
        d['age_group'] = pd.cut(d['age'], bins=[0, 29, 39, 49, 59, 100], labels=False)
        
        # 7. Nhóm hạn mức
        if train_quantiles is None:
            train_quantiles = [-np.inf] + list(d['limit_bal'].quantile([0.25, 0.5, 0.75])) + [np.inf]
            train_quantiles = sorted(list(set(train_quantiles)))
        d['limit_group'] = pd.cut(d['limit_bal'], bins=train_quantiles, labels=False)
        
        # 8. Số tháng pay_amt = 0
        d['num_zero_pay'] = (d[pay_amt_cols] == 0).sum(axis=1)
        
        return d, train_quantiles
        
    print("Feature engineering...")
    train_df, limit_quantiles = add_features(train_df)
    val_df, _ = add_features(val_df, limit_quantiles)
    test_df, _ = add_features(test_df, limit_quantiles)
    
    # Winsorize tiền tệ dựa trên tập train (cắt ở p1, p99)
    monetary_cols = ['limit_bal'] + bill_cols + [f'pay_amt_{m}' for m in months] + ['avg_bill_amt', 'avg_pay_amt']
    winsorize_thresholds = {}
    
    for col in monetary_cols:
        p1 = train_df[col].quantile(0.01)
        p99 = train_df[col].quantile(0.99)
        winsorize_thresholds[col] = {'p1': float(p1), 'p99': float(p99)}
        
        train_df[col] = train_df[col].clip(lower=p1, upper=p99)
        val_df[col] = val_df[col].clip(lower=p1, upper=p99)
        test_df[col] = test_df[col].clip(lower=p1, upper=p99)
        
    # Bước 8: Mã hóa, chuẩn hóa (Lưu artifact)
    # Tách gender ra meta (để riêng thành cột meta_gender)
    for data in [train_df, val_df, test_df]:
        data['meta_gender'] = data.pop('gender')
        
    cat_cols = ['education', 'marriage', 'age_group', 'limit_group']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[cat_cols])
    
    non_scale = ['default', 'meta_gender', 'has_negative_bill'] + cat_cols + [f'pay_{m}' for m in months]
    num_cols = [c for c in train_df.columns if c not in non_scale]
    
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols])
    
    # Lưu artifacts
    print("Saving artifacts...")
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(encoder, 'artifacts/encoder.pkl')
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    with open('artifacts/winsorize_thresholds.json', 'w') as f:
        json.dump(winsorize_thresholds, f, indent=4)
        
    artifact_metadata = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'limit_quantiles': limit_quantiles
    }
    with open('artifacts/metadata.json', 'w') as f:
        json.dump(artifact_metadata, f, indent=4)
        
    # Bước 9: Trọng số lớp
    classes = np.unique(train_df['default'])
    weights = compute_class_weight('balanced', classes=classes, y=train_df['default'])
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    with open('artifacts/class_weight.json', 'w') as f:
        json.dump(class_weights, f, indent=4)
        
    # Bước 10: Lưu dữ liệu
    print("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
