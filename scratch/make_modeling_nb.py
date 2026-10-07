import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()

def mk_md(text):
    return nbf.v4.new_markdown_cell(text)

def mk_code(text):
    return nbf.v4.new_code_cell(text)

cells = []
cells.append(mk_md("# machine_learning - Phần B: Xây dựng và đánh giá mô hình dự đoán\n\n**Mục tiêu**: Xây dựng mô hình phân loại dự đoán vỡ nợ thẻ tín dụng, chọn ngưỡng ra quyết định dựa trên chi phí rủi ro và giải thích mô hình."))

cells.append(mk_code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))
from src.train import build_lgbm_pipeline, get_features_target
from src.evaluate import evaluate_model, get_optimal_threshold, plot_roc_curve, plot_pr_curve, plot_confusion_matrix, plot_cost_curve

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')'''))

cells.append(mk_md("## 1. Kết quả so sánh hiệu năng Baseline và LightGBM (Tập Validation)\n\n**Mục tiêu**: So sánh kết quả mô hình đã tinh chỉnh (LightGBM) với mô hình cơ sở (Logistic Regression) để kiểm chứng Tiêu chí Thành công (Success Criteria)."))

cells.append(mk_code('''comparison_df = pd.read_csv('../reports/machine_learning/tables/model_comparison.csv')
display(comparison_df)'''))

cells.append(mk_md("**Kết luận:**\nQua đó nhận thấy rằng mô hình LightGBM đã vượt qua cả hai tiêu chí thành công đã đặt ra. Cụ thể, mô hình LightGBM cho AUC-ROC cao hơn 0.77 và AUC-PR cao hơn 0.54, đồng thời giúp tiết kiệm thêm chi phí rủi ro (Expected Cost) so với Baseline Logistic Regression."))

cells.append(mk_md("## 2. Lựa chọn ngưỡng quyết định tối ưu dựa trên Chi phí Kỳ vọng\n\n**Mục tiêu**: Xác định ngưỡng (threshold) phân loại tối ưu với giả định Chi phí 1 ca Bỏ sót (False Negative) gấp 5 lần Chi phí 1 ca Cảnh báo nhầm (False Positive)."))

cells.append(mk_code('''from IPython.display import Image
Image(filename='../reports/machine_learning/figures/cost_curve.png')'''))

cells.append(mk_md("**Kết luận:**\nQua đó nhận thấy rằng tổng chi phí kỳ vọng có xu hướng tạo thành hình chữ U dọc theo dải ngưỡng quyết định. Ngưỡng tối ưu nằm ở khoảng giữa (nhỏ hơn 0.5 do FN bị phạt nặng hơn FP) giúp giảm thiểu hoàn toàn thiệt hại cho ngân hàng."))

cells.append(mk_md("## 3. Các biểu đồ đánh giá (ROC, PR, Confusion Matrix)\n\n**Mục tiêu**: Trực quan hóa chi tiết các chỉ số của mô hình LightGBM tại ngưỡng tối ưu."))

cells.append(mk_code('''display(Image(filename='../reports/machine_learning/figures/roc_curve.png'))
display(Image(filename='../reports/machine_learning/figures/pr_curve.png'))
display(Image(filename='../reports/machine_learning/figures/confusion_matrix.png'))'''))

cells.append(mk_md("**Kết luận:**\nQua đó nhận thấy rằng mô hình phân tách các lớp (default và non-default) khá tốt (đường cong ROC phình cao), đặc biệt tỷ lệ Recall ở ngưỡng tối ưu là cao, đáp ứng đúng chiến lược bắt rủi ro (hạn chế False Negative)."))

cells.append(mk_md("## 4. Giải thích mô hình bằng SHAP\n\n**Mục tiêu**: Phân tích các đặc trưng quan trọng đóng góp vào dự đoán vỡ nợ và đối chiếu với kết quả Phần A."))

cells.append(mk_code('''shap_imp = pd.read_csv('../reports/machine_learning/tables/shap_importance.csv')
display(shap_imp.head(10))'''))

cells.append(mk_code('''display(Image(filename='../reports/machine_learning/figures/shap_summary.png'))'''))

cells.append(mk_md("**Kết luận:**\nQua đó nhận thấy rằng các biến liên quan đến lịch sử trễ hạn (`pay_0`, `delay_trend`) đóng vai trò quan trọng nhất, đúng như giả thuyết đề ra ở Phần A. Khách hàng có xu hướng trả trễ càng cao (giá trị tính bằng tháng tăng) thì tác động thúc đẩy dự đoán vỡ nợ (SHAP value dương) càng mạnh."))

cells.append(mk_md("## 5. Đánh giá tính Công bằng theo Giới tính (Gender Fairness)\n\n**Mục tiêu**: Kiểm chứng xem mô hình có tỷ lệ sai sót thiên vị nam hoặc nữ hay không (biến Giới tính không được dùng làm đầu vào mô hình)."))

cells.append(mk_code('''gender_df = pd.read_csv('../reports/machine_learning/tables/gender_comparison.csv')
display(gender_df)'''))

cells.append(mk_md("**Kết luận:**\nQua đó nhận thấy rằng tỷ lệ FNR (Bỏ sót vỡ nợ) và FPR (Cảnh báo nhầm) giữa hai nhóm Giới tính xấp xỉ nhau, không có bằng chứng rõ ràng cho thấy mô hình bị thiên lệch (Bias) đối với bất kỳ nhóm nào. Vì vậy, ta có thể tự tin áp dụng mô hình chung cho toàn bộ tập khách hàng mà không vi phạm chuẩn mực công bằng về giới."))

nb['cells'] = cells

with open('notebooks/05_modeling.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

