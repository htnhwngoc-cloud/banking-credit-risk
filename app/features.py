"""
app/features.py
===============
Feature engineering cho dashboard — mirror chính xác src/preprocess.py.

QUY TẮC QUAN TRỌNG
-------------------
- Mọi bước FE (1–11) phải giống hệt src/preprocess.py.
- Không tự ý thay đổi mà không cập nhật tests/test_features.py.
- Thứ tự cột đầu ra được lấy từ val.csv để khớp với mô hình đã huấn luyện.

Tác giả: Banking Credit Risk Project
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ─────────────────────────── Hằng số ────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR      = PROJECT_ROOT / "data" / "processed"

MONTHS       = ["sep", "aug", "jul", "jun", "may", "apr"]
BILL_COLS    = [f"bill_amt_{m}" for m in MONTHS]
PAY_COLS     = [f"pay_{m}" for m in MONTHS]
PAY_AMT_COLS = [f"pay_amt_{m}" for m in MONTHS]

MONTH_LABEL = {
    "sep": "Tháng 9 (gần nhất)",
    "aug": "Tháng 8",
    "jul": "Tháng 7",
    "jun": "Tháng 6",
    "may": "Tháng 5",
    "apr": "Tháng 4 (xa nhất)",
}

EDUCATION_LABEL = {1: "Sau đại học", 2: "Đại học", 3: "Phổ thông", 4: "Khác"}
MARRIAGE_LABEL  = {1: "Đã kết hôn", 2: "Độc thân", 3: "Khác"}

# Tên hiển thị tiếng Việt cho feature quan trọng (dùng trong SHAP chart)
FEATURE_DISPLAY: dict[str, str] = {
    "avg_delay":            "Mức trễ hạn TB (6 tháng)",
    "pay_sep":              "Lịch sử TT Tháng 9 (gần nhất)",
    "num_delayed_months":   "Số tháng trễ hạn",
    "max_delay":            "Mức trễ hạn tối đa",
    "bill_ratio_aug":       "Tỷ lệ SD hạn mức T8",
    "bill_ratio_sep":       "Tỷ lệ SD hạn mức T9",
    "bill_ratio_jul":       "Tỷ lệ SD hạn mức T7",
    "bill_ratio_jun":       "Tỷ lệ SD hạn mức T6",
    "bill_ratio_may":       "Tỷ lệ SD hạn mức T5",
    "bill_ratio_apr":       "Tỷ lệ SD hạn mức T4",
    "bill_amt_sep":         "Hóa đơn T9 (NTD)",
    "bill_amt_aug":         "Hóa đơn T8 (NTD)",
    "pay_amt_sep":          "Tiền đã trả T9 (NTD)",
    "pay_amt_aug":          "Tiền đã trả T8 (NTD)",
    "limit_bal":            "Hạn mức tín dụng (NTD)",
    "avg_bill_amt":         "Hóa đơn TB 6 tháng (NTD)",
    "avg_pay_amt":          "Tiền trả TB 6 tháng (NTD)",
    "num_zero_pay":         "Số tháng không thanh toán",
    "delay_trend":          "Xu hướng trễ hạn (slope)",
    "pay_ratio_sep":        "Tỷ lệ trả nợ T9",
    "pay_ratio_aug":        "Tỷ lệ trả nợ T8",
    "marriage":             "Hôn nhân",
    "education":            "Học vấn",
    "age":                  "Tuổi",
    "age_group":            "Nhóm tuổi",
    "limit_group":          "Nhóm hạn mức",
    "has_negative_bill":    "Có hóa đơn âm",
}

# ──────────────────────── Helper nội bộ ────────────────────────────
def _load_json_safe(path: Path) -> Any:
    """Load JSON có thể chứa Infinity theo cú pháp Python."""
    content = path.read_text(encoding="utf-8")
    # Thay thế -Infinity / Infinity (JSON không chuẩn) bằng sentinel 1e308
    content = re.sub(r"\b-Infinity\b", "-1e308", content)
    content = re.sub(r"\bInfinity\b",   "1e308",  content)
    data = json.loads(content)

    def _restore(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _restore(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_restore(v) for v in obj]
        if isinstance(obj, float):
            if obj >= 1e307:
                return np.inf
            if obj <= -1e307:
                return -np.inf
        return obj

    return _restore(data)


# Module-level cache để tránh đọc file nhiều lần
_METADATA:     dict | None       = None
_WINSORIZE:    dict | None       = None
_FEATURE_COLS: list[str] | None  = None


def _get_metadata() -> dict:
    global _METADATA
    if _METADATA is None:
        _METADATA = _load_json_safe(ARTIFACTS_DIR / "metadata.json")
    return _METADATA


def _get_winsorize() -> dict:
    global _WINSORIZE
    if _WINSORIZE is None:
        with open(ARTIFACTS_DIR / "winsorize_thresholds.json", encoding="utf-8") as f:
            _WINSORIZE = json.load(f)
    return _WINSORIZE


def get_feature_cols() -> list[str]:
    """
    Trả về danh sách tên cột features theo đúng thứ tự training.
    Lấy từ header val.csv để luôn đồng bộ với mô hình.
    """
    global _FEATURE_COLS
    if _FEATURE_COLS is None:
        val_cols = pd.read_csv(DATA_DIR / "val.csv", nrows=0).columns.tolist()
        _FEATURE_COLS = [c for c in val_cols if c not in ("default", "meta_gender")]
    return _FEATURE_COLS


# ────────────────────── Hàm chính build_features ───────────────────
def build_features(raw_input: dict) -> pd.DataFrame:
    """
    Chuyển đổi đầu vào thô từ người dùng thành DataFrame 1 hàng,
    sẵn sàng cho model.predict_proba().

    Tham số
    -------
    raw_input : dict
        Các khóa bắt buộc:
        - limit_bal (số dương)
        - age (18–100)
        - education (1|2|3|4)
        - marriage (1|2|3)
        - pay_{sep,aug,jul,jun,may,apr} (-2..8)
        - bill_amt_{sep..apr} (số thực)
        - pay_amt_{sep..apr} (số thực ≥ 0)

    Trả về
    ------
    pd.DataFrame
        1 hàng, cột theo đúng thứ tự huấn luyện.
    """
    metadata  = _get_metadata()
    winsorize = _get_winsorize()
    limit_quantiles = metadata["limit_quantiles"]

    d = pd.DataFrame([raw_input]).copy()

    # ── Bước 1: Làm sạch mã education / marriage ──
    d["education"] = d["education"].replace({0: 4, 5: 4, 6: 4})
    d["marriage"]  = d["marriage"].replace({0: 3})

    # ── Bước 2: Cờ hóa đơn âm ──
    d["has_negative_bill"] = (d[BILL_COLS] < 0).any(axis=1).astype(int)

    # ── Bước 3: Tỷ lệ sử dụng hạn mức ──
    for m in MONTHS:
        d[f"bill_ratio_{m}"] = d[f"bill_amt_{m}"] / d["limit_bal"]

    # ── Bước 4: Tỷ lệ trả nợ (= 1.0 khi hóa đơn ≤ 0) ──
    for m in MONTHS:
        d[f"pay_ratio_{m}"] = np.where(
            d[f"bill_amt_{m}"] <= 0,
            1.0,
            d[f"pay_amt_{m}"] / d[f"bill_amt_{m}"],
        )

    # ── Bước 5: Đặc trưng trễ hạn (chỉ pay > 0 mới tính là trễ) ──
    pay_pos = d[PAY_COLS].clip(lower=0)
    d["num_delayed_months"] = (pay_pos > 0).sum(axis=1)
    d["max_delay"]          = pay_pos.max(axis=1)
    d["avg_delay"]          = pay_pos.mean(axis=1)

    # ── Bước 6: Xu hướng trễ hạn (hệ số góc tuyến tính 6 tháng) ──
    # Chỉ số thời gian: apr=1, may=2, jun=3, jul=4, aug=5, sep=6  →  mean=3.5
    # Trọng số (t − mean_t): sep=+2.5, aug=+1.5, jul=+0.5, jun=−0.5, may=−1.5, apr=−2.5
    t_diff = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5])
    d["delay_trend"] = d[PAY_COLS].values.dot(t_diff) / 17.5

    # ── Bước 7: Trung bình 6 tháng ──
    d["avg_bill_amt"] = d[BILL_COLS].mean(axis=1)
    d["avg_pay_amt"]  = d[PAY_AMT_COLS].mean(axis=1)

    # ── Bước 8: Nhóm tuổi [0,29] [30,39] [40,49] [50,59] [60,100] ──
    d["age_group"] = pd.cut(
        d["age"], bins=[0, 29, 39, 49, 59, 100], labels=False
    ).astype(float)

    # ── Bước 9: Nhóm hạn mức (tứ phân vị từ train) ──
    d["limit_group"] = pd.cut(
        d["limit_bal"], bins=limit_quantiles, labels=False
    ).astype(float)

    # ── Bước 10: Số tháng không trả tiền ──
    d["num_zero_pay"] = (d[PAY_AMT_COLS] == 0).sum(axis=1)

    # ── Bước 11: Winsorize theo ngưỡng từ tập train ──
    monetary_cols = (
        ["limit_bal"] + BILL_COLS + PAY_AMT_COLS + ["avg_bill_amt", "avg_pay_amt"]
    )
    for col in monetary_cols:
        if col in winsorize:
            lo = winsorize[col]["p1"]
            hi = winsorize[col]["p99"]
            d[col] = d[col].clip(lower=lo, upper=hi)

    # ── Bước 12: Sắp xếp cột đúng thứ tự huấn luyện ──
    return d[get_feature_cols()]


# ───────────────────── Kiểm tra đầu vào ────────────────────────────
def validate_input(raw_input: dict) -> list[str]:
    """
    Kiểm tra tính hợp lệ của đầu vào người dùng.

    Trả về
    ------
    list[str]
        Danh sách thông báo lỗi. Rỗng = đầu vào hợp lệ.
    """
    errors: list[str] = []

    # Hạn mức tín dụng
    limit = raw_input.get("limit_bal")
    if limit is None or float(limit) <= 0:
        errors.append("Hạn mức tín dụng phải lớn hơn 0 NTD.")

    # Tuổi
    age = raw_input.get("age")
    if age is None or not (18 <= int(age) <= 100):
        errors.append("Tuổi phải nằm trong khoảng 18–100.")

    # Học vấn
    edu = raw_input.get("education")
    if edu not in (1, 2, 3, 4):
        errors.append("Học vấn phải là 1 (Sau ĐH), 2 (ĐH), 3 (PT), hoặc 4 (Khác).")

    # Hôn nhân
    mar = raw_input.get("marriage")
    if mar not in (1, 2, 3):
        errors.append("Hôn nhân phải là 1 (Kết hôn), 2 (Độc thân), hoặc 3 (Khác).")

    # Mã thanh toán và số tiền đã trả
    valid_pay_codes = set(range(-2, 9))  # −2..8
    for m in MONTHS:
        pay_val = raw_input.get(f"pay_{m}")
        if pay_val is None or int(pay_val) not in valid_pay_codes:
            errors.append(
                f"Mã thanh toán {MONTH_LABEL[m]} phải nằm trong khoảng -2 đến 8."
            )
        pay_amt = raw_input.get(f"pay_amt_{m}", 0)
        if float(pay_amt) < 0:
            errors.append(f"Số tiền đã trả {MONTH_LABEL[m]} không được âm.")

    return errors
