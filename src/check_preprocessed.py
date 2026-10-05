import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import hashlib
import subprocess
import shutil
import datetime
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def clean_old_files():
    deleted = []
    files_to_remove = ['reports/check_preprocessed.csv', 'reports/check_max_delay_detail.csv']
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            deleted.append(f)
    
    val_dir = 'reports/validation/'
    if os.path.exists(val_dir):
        for root, dirs, files in os.walk(val_dir, topdown=False):
            for name in files:
                f = os.path.join(root, name)
                os.remove(f)
                deleted.append(f)
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(val_dir)
        deleted.append(val_dir)
        
    print("CÁC FILE/THƯ MỤC ĐÃ DỌN DẸP:")
    for d in deleted:
        print(f" - {d}")
    print("-" * 40)

def get_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_pipeline():
    subprocess.run([sys.executable, 'src/preprocess.py'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def eval_condition(val, cond, thresh):
    if cond == '':
        return 'INFO'
    if isinstance(val, bool):
        val = int(val)
        thresh = int(thresh == 'True' or thresh is True)
        
    if cond == '==': return 'PASS' if val == thresh else 'FAIL'
    if cond == '<': return 'PASS' if val < thresh else 'FAIL'
    if cond == '>': return 'PASS' if val > thresh else 'FAIL'
    if cond == '<=': return 'PASS' if val <= thresh else 'FAIL'
    if cond == '>=': return 'PASS' if val >= thresh else 'FAIL'
    return 'FAIL'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    clean_old_files()
    
    raw_path = 'data/raw/default_credit_card_clients.csv'
    sha_raw_before = get_sha256(raw_path)
    
    run_pipeline()
    sha_raw_after = get_sha256(raw_path)
    sha_train_1 = get_sha256('data/processed/train.csv')
    sha_val_1 = get_sha256('data/processed/val.csv')
    sha_test_1 = get_sha256('data/processed/test.csv')
    
    run_pipeline()
    sha_train_2 = get_sha256('data/processed/train.csv')
    sha_val_2 = get_sha256('data/processed/val.csv')
    sha_test_2 = get_sha256('data/processed/test.csv')

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    target = 'default'
    if target not in train_df.columns and 'default_b' in train_df.columns:
        target = 'default_b'

    raw_df = pd.read_csv(raw_path)
    col_map = {'X1':'limit_bal', 'X2':'gender', 'X3':'education', 'X4':'marriage', 'X5':'age',
               'X6':'pay_sep', 'X7':'pay_aug', 'X8':'pay_jul', 'X9':'pay_jun', 'X10':'pay_may', 'X11':'pay_apr',
               'X12':'bill_amt_sep', 'X13':'bill_amt_aug', 'X14':'bill_amt_jul', 'X15':'bill_amt_jun', 'X16':'bill_amt_may', 'X17':'bill_amt_apr',
               'X18':'pay_amt_sep', 'X19':'pay_amt_aug', 'X20':'pay_amt_jul', 'X21':'pay_amt_jun', 'X22':'pay_amt_may', 'X23':'pay_amt_apr',
               'Y': 'default'}
    if 'X1' in raw_df.columns: raw_df = raw_df.rename(columns=col_map)
    for id_col in ['id', 'ID']:
        if id_col in raw_df.columns: raw_df = raw_df.drop(columns=[id_col])
    raw_df['default'] = raw_df['default'].astype(int)
    raw_df = raw_df.drop_duplicates(keep='first').reset_index(drop=True)
    raw_df['education'] = raw_df['education'].replace({0:4, 5:4, 6:4})
    raw_df['marriage'] = raw_df['marriage'].replace({0:3})
    
    raw_train, _ = train_test_split(raw_df, test_size=0.3, stratify=raw_df['default'], random_state=42)
    
    months = ['sep', 'aug', 'jul', 'jun', 'may', 'apr']
    bill_cols = [f'bill_amt_{m}' for m in months]
    pay_amt_cols = [f'pay_amt_{m}' for m in months]
    
    raw_train['avg_bill_amt'] = raw_train[bill_cols].mean(axis=1)
    raw_train['avg_pay_amt'] = raw_train[pay_amt_cols].mean(axis=1)
    
    with open('artifacts/metadata.json', 'r') as f:
        meta = json.load(f)
    win_thresh = {}
    if os.path.exists('artifacts/winsorize_thresholds.json'):
        with open('artifacts/winsorize_thresholds.json', 'r') as f:
            win_thresh = json.load(f)
    scaler = joblib.load('artifacts/scaler.pkl')
    
    max_win_err = 0.0
    for col, th in win_thresh.items():
        if col in raw_train.columns:
            p1 = raw_train[col].quantile(0.01)
            p99 = raw_train[col].quantile(0.99)
            err = max(abs(p1 - th['p1']), abs(p99 - th['p99']))
            if err > max_win_err: max_win_err = err
            
    limit_err = 0.0
    if 'limit_quantiles' in meta:
        computed_l = raw_train['limit_bal'].quantile([0.25, 0.5, 0.75]).values
        # the saved limit_quantiles contains [0, p25, p50, p75, inf], so we take [1:4]
        saved_l = np.array(meta['limit_quantiles'][1:4])
        limit_err = np.max(np.abs(computed_l - saved_l))
    
    num_cols = meta.get('num_cols', [])
    re_scaler = StandardScaler()
    re_scaler.fit(train_df[num_cols])
    max_scaler_err = max(np.max(np.abs(scaler.mean_ - re_scaler.mean_)), np.max(np.abs(scaler.scale_ - re_scaler.scale_)))
    
    C = []
    def add_c(nhom, chi_so, val, cond, thresh, ghi_chu=""):
        st = eval_condition(val, cond, thresh)
        C.append({
            'nhom': nhom, 'chi_so': chi_so, 'value': val, 
            'dieu_kien': cond, 'nguong': thresh, 'status': st, 'ghi_chu': ghi_chu
        })

    # 1
    add_c("Toàn vẹn dữ liệu", "Hash data/raw/ trước = sau khi chạy", (sha_raw_before == sha_raw_after), '==', 'True')
    
    # Tính hợp lý biến mới
    rate_overall = train_df[target].mean()
    add_c("Tính hợp lý biến mới", "Tỷ lệ vỡ nợ chung", rate_overall, '', '')
    
    rate_0 = train_df[train_df['max_delay'] == 0][target].mean() if 'max_delay' in train_df.columns else 0
    add_c("Tính hợp lý biến mới", "Tỷ lệ vỡ nợ nhóm max_delay = 0", rate_0, '<', rate_overall)
    
    rate_2p = train_df[train_df['max_delay'] >= 2][target].mean() if 'max_delay' in train_df.columns else 0
    add_c("Tính hợp lý biến mới", "Tỷ lệ vỡ nợ nhóm max_delay >= 2", rate_2p, '>', rate_overall)
    
    feats = [c for c in train_df.columns if c != target and pd.api.types.is_numeric_dtype(train_df[c])]
    corrs = {f: spearmanr(train_df[f], train_df[target])[0] for f in feats}
    ranks = {k: i+1 for i, (k, v) in enumerate(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))}
    r_md = corrs.get('max_delay', 0)
    add_c("Tính hợp lý biến mới", "Spearman max_delay với nhãn", r_md, '>', 0)
    add_c("Tính hợp lý biến mới", "Hạng của max_delay trong top |r|", ranks.get('max_delay', 99), '<=', 5)
    
    r_ps = corrs.get('pay_sep', 0)
    add_c("Tính hợp lý biến mới", "Spearman pay_sep với nhãn", r_ps, '>', 0)
    add_c("Tính hợp lý biến mới", "Hạng của pay_sep trong top |r|", ranks.get('pay_sep', 99), '<=', 5)
    
    if 'limit_group' in train_df.columns:
        lr = train_df.groupby('limit_group')[target].mean()
        min_lg, max_lg = lr.index.min(), lr.index.max()
        rate_min_lg, rate_max_lg = lr[min_lg], lr[max_lg]
    else:
        rate_min_lg, rate_max_lg = 0, 0
    add_c("Tính hợp lý biến mới", "Tỷ lệ vỡ nợ limit_group thấp nhất", rate_min_lg, '', '')
    add_c("Tính hợp lý biến mới", "Tỷ lệ vỡ nợ limit_group cao nhất", rate_max_lg, '<', rate_min_lg)
    
    rates = train_df.groupby('max_delay').agg(s=(target, 'size'), r=(target, 'mean'))
    r100 = rates[rates['s'] >= 100].sort_index()
    dec_pairs = sum((r100['r'].iloc[i] > r100['r'].iloc[i+1]) for i in range(len(r100)-1))
    add_c("Tính hợp lý biến mới", "Số cặp max_delay liền kề bị giảm tỷ lệ", dec_pairs, '', '', "nhóm >= 100 dòng")
    
    # Rò rỉ
    def full_set(df): return set(map(tuple, df.values))
    f_tr, f_va, f_te = full_set(train_df), full_set(val_df), full_set(test_df)
    full_dups = len(f_tr.intersection(f_va)) + len(f_tr.intersection(f_te)) + len(f_va.intersection(f_te))
    add_c("Rò rỉ giữa các tập", "Số dòng giống hệt toàn bộ cột (đặc trưng + nhãn) giữa train, val, test", full_dups, '==', 0)
    
    in_cols = [c for c in train_df.columns if c not in [target, 'meta_gender']]
    def get_fset(df): return set(map(tuple, df[in_cols].values))
    st, sv, stt = get_fset(train_df), get_fset(val_df), get_fset(test_df)
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng giữa train và val", len(st.intersection(sv)), '', '')
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng giữa train và test", len(st.intersection(stt)), '', '')
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng giữa val và test", len(sv.intersection(stt)), '', '')
    
    def get_lset(df): return set(map(tuple, df[in_cols + [target]].values))
    lst, lsv, lstt = get_lset(train_df), get_lset(val_df), get_lset(test_df)
    ov_label = len(lst.intersection(lsv)) + len(lst.intersection(lstt)) + len(lsv.intersection(lstt))
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng cùng nhãn giữa các tập", ov_label, '==', 0)
    
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng nội bộ train", len(train_df) - len(st), '', '')
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng nội bộ val", len(val_df) - len(sv), '', '')
    add_c("Rò rỉ giữa các tập", "Số cặp trùng đặc trưng nội bộ test", len(test_df) - len(stt), '', '')
    
    # Fit chỉ trên train
    add_c("Fit chỉ trên train", "Sai số lớn nhất của scaler so với tính lại từ train", float(max_scaler_err), '<', 1e-4)
    add_c("Fit chỉ trên train", "Sai số lớn nhất của mốc winsorize", float(max_win_err), '<', 1e-4)
    add_c("Fit chỉ trên train", "Sai số lớn nhất của mốc phân vị limit_group", float(limit_err), '<', 1e-4)
    
    enc_match = True
    if os.path.exists('artifacts/encoder.pkl'):
        encoder = joblib.load('artifacts/encoder.pkl')
        cat_c = meta.get('cat_cols', [])
        for i, col in enumerate(cat_c):
            if col in train_df.columns:
                train_cats = set(train_df[col].dropna().unique())
                enc_cats = set(encoder.categories_[i])
                if train_cats != enc_cats:
                    enc_match = False
                    break
            else:
                enc_match = False
                break
    add_c("Fit chỉ trên train", "Encoder: categories_ khớp với train", enc_match, '==', 'True')
    
    # Cột đầu vào
    model_in_cols = meta.get('num_cols', []) + meta.get('cat_cols', [])
    has_mg = 'meta_gender' in model_in_cols
    has_df = 'default_b' in model_in_cols or target in model_in_cols
    add_c("Cột đầu vào", "meta_gender có trong cột đầu vào mô hình", has_mg, '==', 'False')
    add_c("Cột đầu vào", "default_b có trong cột đầu vào mô hình", has_df, '==', 'False')
    add_c("Cột đầu vào", "Số dòng val sau pipeline", len(val_df), '==', 4495)
    add_c("Cột đầu vào", "Số dòng test sau pipeline", len(test_df), '==', 4495)
    
    # Tái lập
    add_c("Tái lập", "Hash file train đầu ra khớp giữa 2 lần chạy", (sha_train_1 == sha_train_2), '==', 'True')
    add_c("Tái lập", "Hash file val đầu ra khớp giữa 2 lần chạy", (sha_val_1 == sha_val_2), '==', 'True')
    add_c("Tái lập", "Hash file test đầu ra khớp giữa 2 lần chạy", (sha_test_1 == sha_test_2), '==', 'True')
    
    # Markdown formatting
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rs = 42
    
    n_pass = sum(1 for c in C if c['status'] == 'PASS')
    n_fail = sum(1 for c in C if c['status'] == 'FAIL')
    n_info = sum(1 for c in C if c['status'] == 'INFO')
    
    fail_ids = [str(i+1) for i, c in enumerate(C) if c['status'] == 'FAIL']
    fail_str = f"Liệt kê id các dòng FAIL: {', '.join(fail_ids)}" if fail_ids else "Không có FAIL."
    
    lines = [
        "# Kết quả kiểm tra dữ liệu sau tiền xử lý\n",
        f"Thời điểm chạy: {now_str} | random_state: {rs}\n",
        "## Tóm tắt",
        f"- Tổng PASS: {n_pass}",
        f"- Tổng FAIL: {n_fail}",
        f"- Tổng INFO: {n_info}",
        f"- {fail_str}\n",
        "## Bảng kiểm tra\n",
        "| id | nhom | chi_so | value | dieu_kien | nguong | status | ghi_chu |",
        "|---|---|---|---|---|---|---|---|"
    ]
    
    for i, c in enumerate(C):
        v = c['value']
        if isinstance(v, float):
            v_str = f"{v:.4f}"
        else:
            v_str = str(v)
            
        ghi = c['ghi_chu'] if c['ghi_chu'] else ""
        dk = c['dieu_kien'] if c['dieu_kien'] else ""
        ng = c['nguong'] if str(c['nguong']) != "" else ""
        if isinstance(ng, float):
            ng = f"{ng:.4f}"
        
        row_str = f"| {i+1} | {c['nhom']} | {c['chi_so']} | {v_str} | {dk} | {ng} | {c['status']} | {ghi} |"
        lines.append(row_str)
        
    lines.append("\n## Chi tiết max_delay\n")
    lines.append("| max_delay | so_dong | so_vo_no | ty_le_vo_no |")
    lines.append("|---|---|---|---|")
    
    rates_all = train_df.groupby('max_delay').agg(so_dong=(target, 'size'), so_vo_no=(target, 'sum'), ty_le_vo_no=(target, 'mean')).sort_index()
    for maxd, row in rates_all.iterrows():
        lines.append(f"| {maxd} | {int(row['so_dong'])} | {int(row['so_vo_no'])} | {row['ty_le_vo_no']:.4f} |")
        
    md_content = "\n".join(lines) + "\n"
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/check_preprocessed.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(md_content)
    
    if n_fail > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
