

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_clients_clean.csv"

TARGET = "default_next_month"
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]

# Danh sách cột chuẩn (23 đầu vào + 1 mục tiêu)
EXPECTED_COLUMNS = (
    ["limit_bal", "gender", "education", "marriage", "age"]
    + [f"pay_{m}" for m in MONTHS]
    + [f"bill_amt_{m}" for m in MONTHS]
    + [f"pay_amt_{m}" for m in MONTHS]
    + [TARGET]
)

# Bảng ánh xạ tên gốc -> tên chuẩn. Gồm cả 2 kiểu: mã X (ucimlrepo) và tên file xls gốc.
RENAME_MAP = {"id": None, "y": TARGET, "default payment next month": TARGET,
              "limit_bal": "limit_bal", "sex": "gender", "gender": "gender",
              "education": "education", "marriage": "marriage", "age": "age"}
for i, name in enumerate(["limit_bal", "gender", "education", "marriage", "age"], start=1):
    RENAME_MAP[f"x{i}"] = name
for i, m in enumerate(MONTHS):
    RENAME_MAP[f"x{6 + i}"] = f"pay_{m}"
    RENAME_MAP[f"x{12 + i}"] = f"bill_amt_{m}"
    RENAME_MAP[f"x{18 + i}"] = f"pay_amt_{m}"
    # tên trong file xls gốc: PAY_0, PAY_2..PAY_6 (không có PAY_1)
    RENAME_MAP["pay_0" if i == 0 else f"pay_{i + 1}"] = f"pay_{m}"
    RENAME_MAP[f"bill_amt{i + 1}"] = f"bill_amt_{m}"
    RENAME_MAP[f"pay_amt{i + 1}"] = f"pay_amt_{m}"

MIN_ROWS, MAX_ROWS = 25_000, 35_000


def find_raw_file(raw_dir: Path = RAW_DIR) -> Path:
    """Tìm file dữ liệu trong data/raw/ (csv, xls, xlsx)."""
    candidates = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in {".csv", ".xls", ".xlsx"})
    if not candidates:
        raise FileNotFoundError(f"Không thấy file .csv/.xls/.xlsx nào trong {raw_dir}")
    if len(candidates) > 1:
        logger.warning("Có nhiều file trong %s, dùng file đầu tiên: %s", raw_dir, candidates[0].name)
    return candidates[0]


def read_raw(path) -> pd.DataFrame:
    """Đọc file thô thành DataFrame. Hỗ trợ csv, xls, xlsx."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    suffix = path.suffix.lower()
    logger.info("Đọc file: %s", path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        # File gốc của UCI có dòng 1 là tiêu đề phụ (X1, X2...), dòng 2 mới là tên cột
        return pd.read_excel(path, header=1)
    raise ValueError(f"Định dạng không hỗ trợ: {suffix}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột về snake_case chuẩn và bỏ cột ID nếu có."""
    df = df.copy()
    new_names = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key not in RENAME_MAP:
            raise ValueError(f"Cột lạ, chưa có trong bảng ánh xạ: {col!r}")
        new_names[col] = RENAME_MAP[key]

    to_drop = [c for c, n in new_names.items() if n is None]
    df = df.drop(columns=to_drop)
    df = df.rename(columns={c: n for c, n in new_names.items() if n is not None})
    if to_drop:
        logger.info("Đã bỏ cột chỉ số: %s", to_drop)
    return df[[c for c in EXPECTED_COLUMNS if c in df.columns]]


def validate(df: pd.DataFrame) -> None:
    """Kiểm tra dữ liệu. Báo lỗi rõ ràng (ValueError) nếu có vấn đề.

    Không fail vì mã lạ của education/marriage/pay_*: đó là việc của Phase 2.
    """
    errors = []

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Thiếu cột: {missing_cols}")

    if TARGET in df.columns:
        bad = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad:
            errors.append(f"{TARGET} có giá trị ngoài 0/1: {sorted(bad)}")

    if not MIN_ROWS <= len(df) <= MAX_ROWS:
        errors.append(f"Số dòng {len(df)} ngoài khoảng {MIN_ROWS}-{MAX_ROWS}")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Cột không phải kiểu số: {non_numeric}")

    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError("Validate thất bại:\n- " + "\n- ".join(errors))
    logger.info("Validate đạt.")


def print_summary(df: pd.DataFrame) -> None:
    """In tóm tắt: kích thước, giá trị thiếu, tỷ lệ vỡ nợ, phân bố các cột phân loại."""
    print("=" * 60)
    print(f"Số dòng: {df.shape[0]:,} | Số cột: {df.shape[1]}")
    print("\nKiểu dữ liệu:")
    print(df.dtypes.value_counts().to_string())

    print("\nSố giá trị thiếu mỗi cột (chỉ in cột có thiếu):")
    na = df.isna().sum()
    print(na[na > 0].to_string() if na.any() else "Không có giá trị thiếu.")

    print(f"\nPhân bố {TARGET}:")
    counts = df[TARGET].value_counts().sort_index()
    print(pd.DataFrame({"so_dong": counts, "ty_le_%": (counts / len(df) * 100).round(2)}).to_string())

    print("\nSố dòng trùng lặp (toàn bộ cột):", int(df.duplicated().sum()))

    for col in ["gender", "education", "marriage"] + [f"pay_{m}" for m in MONTHS]:
        print(f"\nvalue_counts {col}:")
        print(df[col].value_counts().sort_index().to_string())
    print("=" * 60)


def load_data(path=None) -> pd.DataFrame:
    """Đọc, chuẩn hóa tên cột và validate. Trả về DataFrame sạch tên cột."""
    path = Path(path) if path else find_raw_file()
    df = standardize_columns(read_raw(path))
    validate(df)
    return df


if __name__ == "__main__":
    # Windows terminals may default to a legacy code page that cannot render Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_data(arg)
    print_summary(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    logger.info("Đã lưu dữ liệu đã chuẩn hóa: %s", OUTPUT_PATH)
