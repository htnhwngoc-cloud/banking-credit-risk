"""
tests/test_features.py
======================
Unit tests cho app/features.py — đảm bảo logic FE khớp với src/preprocess.py.

Nhóm test:
  1. test_build_vs_val    — đối chiếu với val.csv (hàng 0)
  2. test_formula_*       — kiểm tra từng công thức FE cụ thể
  3. test_edge_cases      — trường hợp biên (bill=0, pay=0, giá trị cực trị)
  4. test_validate_*      — kiểm tra hàm validate_input
  5. test_model_*         — tích hợp với mô hình đã lưu

Chạy: pytest tests/test_features.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Thêm project root vào sys.path để import app.features
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.features import (
    MONTHS,
    BILL_COLS,
    PAY_COLS,
    PAY_AMT_COLS,
    build_features,
    get_feature_cols,
    validate_input,
)

# ─────────────────────── Hồ sơ mẫu dùng chung ───────────────────────
SAMPLE_INPUT = {
    "limit_bal": 200_000,
    "age": 35,
    "education": 2,
    "marriage": 1,
    # Lịch sử thanh toán
    "pay_sep": -1, "pay_aug": -1, "pay_jul": 0,
    "pay_jun": 0,  "pay_may": -1, "pay_apr": -1,
    # Hóa đơn
    "bill_amt_sep": 50_000, "bill_amt_aug": 48_000, "bill_amt_jul": 45_000,
    "bill_amt_jun": 42_000, "bill_amt_may": 40_000, "bill_amt_apr": 38_000,
    # Tiền đã trả
    "pay_amt_sep": 50_000, "pay_amt_aug": 48_000, "pay_amt_jul": 45_000,
    "pay_amt_jun": 42_000, "pay_amt_may": 40_000, "pay_amt_apr": 38_000,
}


# ════════════════════════════════════════════════════════════════════
# 1. Đối chiếu với val.csv
# ════════════════════════════════════════════════════════════════════

class TestBuildVsValCsv:
    """Đối chiếu output của build_features với hàng thực trong val.csv."""

    @pytest.fixture(scope="class")
    def val_row(self):
        path = PROJECT_ROOT / "data" / "processed" / "val.csv"
        row = pd.read_csv(path, nrows=1).iloc[0]
        return row

    def _row_to_raw_input(self, row: pd.Series) -> dict:
        """Lấy các cột gốc từ val.csv row làm raw_input."""
        raw = {}
        raw["limit_bal"] = row["limit_bal"]
        raw["age"]       = row["age"]
        raw["education"] = row["education"]
        raw["marriage"]  = row["marriage"]
        for m in MONTHS:
            raw[f"pay_{m}"]      = row[f"pay_{m}"]
            raw[f"bill_amt_{m}"] = row[f"bill_amt_{m}"]
            raw[f"pay_amt_{m}"]  = row[f"pay_amt_{m}"]
        return raw

    def test_feature_columns_match(self, val_row):
        """Các cột trả về phải khớp với feature_cols."""
        raw = self._row_to_raw_input(val_row)
        df = build_features(raw)
        assert list(df.columns) == get_feature_cols()

    def test_bill_ratio_matches(self, val_row):
        """Tỷ lệ sử dụng hạn mức phải khớp với val.csv."""
        raw = self._row_to_raw_input(val_row)
        df = build_features(raw)
        for m in MONTHS:
            expected = float(val_row[f"bill_ratio_{m}"])
            actual   = float(df[f"bill_ratio_{m}"].iloc[0])
            assert abs(actual - expected) < 1e-4, \
                f"bill_ratio_{m}: got {actual}, expected {expected}"

    def test_delay_features_match(self, val_row):
        """num_delayed_months, max_delay, avg_delay phải khớp."""
        raw = self._row_to_raw_input(val_row)
        df = build_features(raw)
        for col in ("num_delayed_months", "max_delay", "avg_delay", "delay_trend"):
            expected = float(val_row[col])
            actual   = float(df[col].iloc[0])
            assert abs(actual - expected) < 1e-4, \
                f"{col}: got {actual}, expected {expected}"

    def test_avg_amounts_match(self, val_row):
        """avg_bill_amt và avg_pay_amt phải khớp."""
        raw = self._row_to_raw_input(val_row)
        df = build_features(raw)
        for col in ("avg_bill_amt", "avg_pay_amt"):
            expected = float(val_row[col])
            actual   = float(df[col].iloc[0])
            assert abs(actual - expected) < 1e-1, \
                f"{col}: got {actual}, expected {expected}"

    def test_num_zero_pay_matches(self, val_row):
        raw = self._row_to_raw_input(val_row)
        df  = build_features(raw)
        assert int(df["num_zero_pay"].iloc[0]) == int(val_row["num_zero_pay"])


# ════════════════════════════════════════════════════════════════════
# 2. Kiểm tra từng công thức FE
# ════════════════════════════════════════════════════════════════════

class TestFormulas:
    """Kiểm tra từng công thức FE bằng giá trị được tính tay."""

    def test_bill_ratio_positive_bill(self):
        inp = dict(SAMPLE_INPUT)
        inp["bill_amt_sep"] = 50_000
        inp["limit_bal"]    = 200_000
        df = build_features(inp)
        assert abs(df["bill_ratio_sep"].iloc[0] - 0.25) < 1e-6

    def test_pay_ratio_positive_bill(self):
        inp = dict(SAMPLE_INPUT)
        inp["bill_amt_sep"] = 100_000
        inp["pay_amt_sep"]  = 30_000
        df = build_features(inp)
        assert abs(df["pay_ratio_sep"].iloc[0] - 0.3) < 1e-6

    def test_pay_ratio_zero_bill_returns_1(self):
        """Khi hóa đơn ≤ 0, tỷ lệ trả nợ phải là 1.0."""
        inp = dict(SAMPLE_INPUT)
        inp["bill_amt_sep"] = 0
        inp["pay_amt_sep"]  = 5_000
        df = build_features(inp)
        assert df["pay_ratio_sep"].iloc[0] == 1.0

    def test_pay_ratio_negative_bill_returns_1(self):
        """Khi hóa đơn âm (khách trả dư), tỷ lệ trả nợ phải là 1.0."""
        inp = dict(SAMPLE_INPUT)
        inp["bill_amt_sep"] = -5_000
        inp["pay_amt_sep"]  = 0
        df = build_features(inp)
        assert df["pay_ratio_sep"].iloc[0] == 1.0

    def test_num_delayed_months_counts_positive_pay(self):
        """Chỉ giá trị pay_* > 0 mới tính là tháng trễ."""
        inp = dict(SAMPLE_INPUT)
        # sep=2, aug=0, jul=-1, jun=0, may=3, apr=-2
        inp.update({"pay_sep": 2, "pay_aug": 0, "pay_jul": -1,
                    "pay_jun": 0, "pay_may": 3, "pay_apr": -2})
        df = build_features(inp)
        assert int(df["num_delayed_months"].iloc[0]) == 2  # sep và may

    def test_max_delay(self):
        inp = dict(SAMPLE_INPUT)
        inp.update({"pay_sep": 0, "pay_aug": 2, "pay_jul": 3,
                    "pay_jun": 1, "pay_may": 0, "pay_apr": 0})
        df = build_features(inp)
        assert int(df["max_delay"].iloc[0]) == 3

    def test_delay_trend_no_delay_zero(self):
        """Khi tất cả pay_* = 0, delay_trend phải bằng 0."""
        inp = dict(SAMPLE_INPUT)
        for m in MONTHS:
            inp[f"pay_{m}"] = 0
        df = build_features(inp)
        assert abs(df["delay_trend"].iloc[0]) < 1e-9

    def test_delay_trend_increasing(self):
        """Trễ hạn tăng dần (tháng gần nhất nặng hơn) → delay_trend dương."""
        inp = dict(SAMPLE_INPUT)
        # pay: sep=3, aug=2, jul=1, jun=0, may=0, apr=0 → slope dương
        inp.update({"pay_sep": 3, "pay_aug": 2, "pay_jul": 1,
                    "pay_jun": 0, "pay_may": 0, "pay_apr": 0})
        df = build_features(inp)
        assert df["delay_trend"].iloc[0] > 0

    def test_delay_trend_formula(self):
        """Kiểm tra giá trị chính xác của delay_trend."""
        inp = dict(SAMPLE_INPUT)
        pays = [2, 1, 0, 0, 0, 0]  # sep..apr
        for m, v in zip(MONTHS, pays):
            inp[f"pay_{m}"] = v
        t_diff = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5])
        expected = np.dot(pays, t_diff) / 17.5
        df = build_features(inp)
        assert abs(df["delay_trend"].iloc[0] - expected) < 1e-9

    def test_has_negative_bill_flag(self):
        inp = dict(SAMPLE_INPUT)
        inp["bill_amt_sep"] = -1_000
        df = build_features(inp)
        assert int(df["has_negative_bill"].iloc[0]) == 1

    def test_has_negative_bill_flag_false(self):
        inp = dict(SAMPLE_INPUT)
        # tất cả bill đều dương
        for m in MONTHS:
            inp[f"bill_amt_{m}"] = 10_000
        df = build_features(inp)
        assert int(df["has_negative_bill"].iloc[0]) == 0

    def test_num_zero_pay(self):
        inp = dict(SAMPLE_INPUT)
        inp.update({
            "pay_amt_sep": 0, "pay_amt_aug": 5000, "pay_amt_jul": 0,
            "pay_amt_jun": 0, "pay_amt_may": 3000, "pay_amt_apr": 0,
        })
        df = build_features(inp)
        assert int(df["num_zero_pay"].iloc[0]) == 4


# ════════════════════════════════════════════════════════════════════
# 3. Trường hợp biên
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_winsorize_clips_extreme_bill(self):
        """Hóa đơn cực cao phải bị clip xuống p99."""
        inp = dict(SAMPLE_INPUT)
        inp["bill_amt_sep"] = 999_999_999   # cực lớn
        df = build_features(inp)
        # Sau winsorize không được quá lớn bất thường
        assert df["bill_amt_sep"].iloc[0] < 999_999_999

    def test_winsorize_clips_extreme_limit(self):
        """Hạn mức cực cao phải bị clip."""
        inp = dict(SAMPLE_INPUT)
        inp["limit_bal"] = 999_999_999
        df = build_features(inp)
        assert df["limit_bal"].iloc[0] < 999_999_999

    def test_age_group_young(self):
        """Tuổi 25 → nhóm 0 (< 30)."""
        inp = dict(SAMPLE_INPUT)
        inp["age"] = 25
        df = build_features(inp)
        assert int(df["age_group"].iloc[0]) == 0

    def test_age_group_middle(self):
        """Tuổi 45 → nhóm 2 (40–49)."""
        inp = dict(SAMPLE_INPUT)
        inp["age"] = 45
        df = build_features(inp)
        assert int(df["age_group"].iloc[0]) == 2

    def test_limit_group_q1(self):
        """Hạn mức 30.000 ≤ 50.000 → nhóm 0 (Q1)."""
        inp = dict(SAMPLE_INPUT)
        inp["limit_bal"] = 30_000
        df = build_features(inp)
        assert int(df["limit_group"].iloc[0]) == 0

    def test_limit_group_q4(self):
        """Hạn mức 300.000 > 240.000 → nhóm 3 (Q4)."""
        inp = dict(SAMPLE_INPUT)
        inp["limit_bal"] = 300_000
        df = build_features(inp)
        assert int(df["limit_group"].iloc[0]) == 3

    def test_output_has_no_inf_or_nan(self):
        """Output không được có Inf hoặc NaN."""
        df = build_features(SAMPLE_INPUT)
        assert not df.isnull().any().any(), "Có giá trị NaN trong output!"
        assert not np.isinf(df.values.astype(float)).any(), "Có giá trị Inf trong output!"


# ════════════════════════════════════════════════════════════════════
# 4. Kiểm tra validate_input
# ════════════════════════════════════════════════════════════════════

class TestValidateInput:
    def test_valid_input_no_errors(self):
        errors = validate_input(SAMPLE_INPUT)
        assert errors == [], f"Đầu vào hợp lệ nhưng có lỗi: {errors}"

    def test_invalid_limit_bal_zero(self):
        inp = dict(SAMPLE_INPUT, limit_bal=0)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_limit_bal_negative(self):
        inp = dict(SAMPLE_INPUT, limit_bal=-5000)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_age_too_young(self):
        inp = dict(SAMPLE_INPUT, age=15)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_age_too_old(self):
        inp = dict(SAMPLE_INPUT, age=110)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_education(self):
        inp = dict(SAMPLE_INPUT, education=5)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_marriage(self):
        inp = dict(SAMPLE_INPUT, marriage=0)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_pay_out_of_range(self):
        inp = dict(SAMPLE_INPUT, pay_sep=9)   # ngoài [-2, 8]
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_invalid_pay_amt_negative(self):
        inp = dict(SAMPLE_INPUT, pay_amt_sep=-1000)
        errors = validate_input(inp)
        assert len(errors) > 0

    def test_multiple_errors(self):
        """Nhiều lỗi phải được báo cáo đầy đủ."""
        inp = dict(SAMPLE_INPUT, limit_bal=0, age=10, education=9)
        errors = validate_input(inp)
        assert len(errors) >= 3


# ════════════════════════════════════════════════════════════════════
# 5. Tích hợp với mô hình
# ════════════════════════════════════════════════════════════════════

class TestModelIntegration:
    @pytest.fixture(scope="class")
    def model(self):
        return joblib.load(PROJECT_ROOT / "models" / "best_model.joblib")

    def test_model_accepts_features(self, model):
        """model.predict_proba() phải chạy không lỗi với output của build_features."""
        df = build_features(SAMPLE_INPUT)
        proba = model.predict_proba(df)
        assert proba.shape == (1, 2), "Output predict_proba phải có shape (1, 2)"
        assert 0.0 <= proba[0, 1] <= 1.0, "Xác suất phải nằm trong [0, 1]"

    def test_threshold_produces_correct_label(self, model):
        """
        Xác suất dự đoán trên ngưỡng 0.3763 → nhãn = 1 (dự báo vỡ nợ).
        Dùng khách hàng rủi ro rất cao để đảm bảo xác suất đủ cao.
        """
        high_risk = dict(SAMPLE_INPUT)
        high_risk.update({
            "limit_bal": 20_000, "age": 55, "education": 4, "marriage": 3,
            "pay_sep": 3, "pay_aug": 3, "pay_jul": 2,
            "pay_jun": 2, "pay_may": 1, "pay_apr": 0,
            "bill_amt_sep": 19_500, "bill_amt_aug": 19_000, "bill_amt_jul": 18_500,
            "bill_amt_jun": 18_000, "bill_amt_may": 17_000, "bill_amt_apr": 16_000,
            "pay_amt_sep": 0, "pay_amt_aug": 0, "pay_amt_jul": 500,
            "pay_amt_jun": 500, "pay_amt_may": 500, "pay_amt_apr": 500,
        })
        df    = build_features(high_risk)
        proba = model.predict_proba(df)[0, 1]
        THRESHOLD = 0.3763
        assert proba > THRESHOLD, \
            f"Hồ sơ rủi ro rất cao nhưng xác suất chỉ = {proba:.4f} < threshold {THRESHOLD}"

    def test_low_risk_below_threshold(self, model):
        """Hồ sơ trả nợ đầy đủ, hạn mức cao → xác suất dưới ngưỡng."""
        low_risk = {
            "limit_bal": 500_000, "age": 38, "education": 1, "marriage": 1,
            "pay_sep": -1, "pay_aug": -1, "pay_jul": -1,
            "pay_jun": -1, "pay_may": -1, "pay_apr": -1,
            "bill_amt_sep": 30_000, "bill_amt_aug": 28_000, "bill_amt_jul": 25_000,
            "bill_amt_jun": 22_000, "bill_amt_may": 20_000, "bill_amt_apr": 18_000,
            "pay_amt_sep": 30_000, "pay_amt_aug": 28_000, "pay_amt_jul": 25_000,
            "pay_amt_jun": 22_000, "pay_amt_may": 20_000, "pay_amt_apr": 18_000,
        }
        df    = build_features(low_risk)
        proba = model.predict_proba(df)[0, 1]
        THRESHOLD = 0.3763
        assert proba < THRESHOLD, \
            f"Hồ sơ rủi ro thấp nhưng xác suất = {proba:.4f} > threshold {THRESHOLD}"
