"""
app/streamlit_app.py
====================
Dashboard Streamlit cho dự báo rủi ro tín dụng.
"""

import streamlit as st
import pandas as pd
import joblib
import yaml
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Thêm root vào sys.path để import features
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.features import (
    build_features, 
    validate_input, 
    FEATURE_DISPLAY,
    MONTHS,
    MONTH_LABEL
)

# -------------------------------------------------------------------------
# Khởi tạo và cấu hình
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Dự báo rủi ro tín dụng thẻ",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cấu hình font cho biểu đồ matplotlib (SHAP plot)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'sans-serif']

@st.cache_data
def load_config():
    with open(PROJECT_ROOT / "app" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

@st.cache_resource
def load_model():
    return joblib.load(PROJECT_ROOT / CONFIG["model"]["path"])

@st.cache_resource
def load_explainer(_model):
    # Trích xuất LightGBM từ Pipeline
    estimator = _model.named_steps["classifier"]
    return shap.TreeExplainer(estimator)

MODEL = load_model()
EXPLAINER = load_explainer(MODEL)
THRESHOLD = CONFIG["model"]["threshold"]

# -------------------------------------------------------------------------
# Hàm phụ trợ UI
# -------------------------------------------------------------------------
def get_risk_band(prob):
    bands = CONFIG["risk_bands"]
    if prob < bands["low"][1]:
        return "low"
    elif prob < bands["medium"][1]:
        return "medium"
    elif prob < bands["high"][1]:
        return "high"
    else:
        return "very_high"

def draw_gauge(prob, threshold):
    st.progress(min(prob, 1.0))
    st.write(f"**Xác suất vỡ nợ:** {prob:.1%} (Ngưỡng cảnh báo: {threshold:.1%})")

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
st.sidebar.title("Banking Credit Risk")
st.sidebar.markdown(CONFIG["disclaimer"])

# -------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Dự Đoán Rủi Ro", "Giải Thích SHAP", "Khách Hàng Mẫu & Hướng Dẫn"])

with tab1:
    st.header("Nhập Thông Tin Khách Hàng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin cơ bản")
        limit_bal = st.number_input("Hạn mức tín dụng (NTD)", min_value=1000, value=50000, step=5000)
        age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
        education = st.selectbox("Học vấn", options=[1, 2, 3, 4], format_func=lambda x: {1: "Sau đại học", 2: "Đại học", 3: "Phổ thông", 4: "Khác"}[x])
        marriage = st.selectbox("Hôn nhân", options=[1, 2, 3], format_func=lambda x: {1: "Đã kết hôn", 2: "Độc thân", 3: "Khác"}[x])
        
    with col2:
        st.subheader("Lịch sử thanh toán (Mã)")
        st.markdown("*-2: Không giao dịch, -1: Trả đủ, 0: Nợ quay vòng, 1-8: Số tháng trễ*")
        pay_inputs = {}
        pay_cols = st.columns(3)
        for i, m in enumerate(MONTHS):
            with pay_cols[i % 3]:
                pay_inputs[f"pay_{m}"] = st.number_input(MONTH_LABEL[m], min_value=-2, max_value=8, value=0, key=f"pay_{m}")
                
    st.subheader("Sao kê và Tiền đã trả (NTD)")
    bill_amt_inputs = {}
    pay_amt_inputs = {}
    
    b_cols = st.columns(6)
    for i, m in enumerate(MONTHS):
        with b_cols[i]:
            bill_amt_inputs[f"bill_amt_{m}"] = st.number_input(f"Hóa đơn T{9-i}", value=0, step=1000, key=f"bill_{m}")
            pay_amt_inputs[f"pay_amt_{m}"] = st.number_input(f"Đã trả T{9-i}", min_value=0, value=0, step=1000, key=f"pay_amt_{m}")
            
    if st.button("DỰ ĐOÁN", type="primary", use_container_width=True):
        raw_input = {
            "limit_bal": limit_bal,
            "age": age,
            "education": education,
            "marriage": marriage,
            **pay_inputs,
            **bill_amt_inputs,
            **pay_amt_inputs
        }
        
        errors = validate_input(raw_input)
        if errors:
            for err in errors:
                st.error(err)
        else:
            # Build features
            df_features = build_features(raw_input)
            
            # Predict
            prob = MODEL.predict_proba(df_features)[0, 1]
            band = get_risk_band(prob)
            
            st.markdown("---")
            st.subheader("Kết quả dự đoán")
            
            draw_gauge(prob, THRESHOLD)
            
            st.markdown(f"**Dải rủi ro:** {CONFIG['risk_band_labels'][band]}")
            st.markdown(f"**Hành động gợi ý:** {CONFIG['risk_band_actions'][band]}")
            
            # Lưu vào session_state để dùng bên tab 2
            st.session_state["last_features"] = df_features
            st.session_state["last_prob"] = prob

with tab2:
    st.header("Giải thích quyết định của mô hình (SHAP)")
    if "last_features" in st.session_state:
        st.write("Dưới đây là các yếu tố ảnh hưởng mạnh nhất đến xác suất vỡ nợ của hồ sơ vừa nhập.")
        
        df_f = st.session_state["last_features"]
        
        # Lấy tên cột hiển thị tiếng Việt
        display_names = [FEATURE_DISPLAY.get(col, col) for col in df_f.columns]
        
        # Tiền xử lý dữ liệu qua scaler trước khi đưa vào TreeExplainer
        X_transformed = MODEL.named_steps["scaler"].transform(df_f)
        X_transformed_df = pd.DataFrame(X_transformed, columns=display_names)
        
        # Calculate SHAP values
        shap_values = EXPLAINER(X_transformed_df)
        
        # Tạo waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("Màu đỏ (hướng sang phải) đẩy xác suất rủi ro TĂNG lên. Màu xanh (hướng sang trái) đẩy xác suất rủi ro GIẢM xuống.")
        
    else:
        st.warning("Vui lòng thực hiện dự đoán ở tab 'Dự Đoán Rủi Ro' trước.")

with tab3:
    st.header("Khách Hàng Mẫu")
    st.write("Sử dụng các số liệu sau để thử nghiệm (bạn có thể nhập vào form bên Tab 1):")
    
    st.markdown("""
    **1. Rủi ro thấp (Khách hàng tốt):**
    - Hạn mức: 300,000 NTD. Tuổi: 35.
    - Lịch sử TT: Tất cả đều -1 (Trả đủ).
    - Hóa đơn: Đều đặn ~50,000. Tiền trả: ~50,000 (Đều trả hết).
    
    **2. Rủi ro trung bình:**
    - Hạn mức: 100,000 NTD. Tuổi: 40.
    - Lịch sử TT: xen kẽ 0 và -1.
    - Hóa đơn: ~80,000. Tiền trả: ~20,000 (Trả chưa hết).
    
    **3. Rủi ro cao:**
    - Hạn mức: 50,000 NTD. Tuổi: 25.
    - Lịch sử TT: Tháng 9 trễ (1), các tháng trước là 0.
    - Hóa đơn: 49,000 (Kiệt hạn mức). Tiền trả: 2,000 (Chỉ trả tối thiểu).
    
    **4. Rủi ro rất cao:**
    - Hạn mức: 20,000 NTD.
    - Lịch sử TT: Các tháng gần đây là 2, 3 (Trễ hạn nhiều tháng).
    - Tiền trả: 0 (Nhiều tháng không thanh toán đồng nào).
    """)

st.markdown("---")
st.caption(CONFIG["disclaimer"])
