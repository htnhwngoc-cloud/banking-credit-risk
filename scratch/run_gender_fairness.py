import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate import get_optimal_threshold
from src.train import get_features_target

def main():
    val_df = pd.read_csv('data/processed/val.csv')
    features, target = get_features_target(val_df)
    
    X_val = val_df[features]
    y_val = val_df[target]
    meta_gender_val = val_df['meta_gender']
    
    pipeline = joblib.load('models/best_model.joblib')
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    
    opt_thresh, _, _, _ = get_optimal_threshold(y_val, y_val_prob, 5, 1)
    
    gender_results = []
    for g in [1, 2]:
        mask = (meta_gender_val == g)
        y_g = y_val[mask]
        y_g_prob = y_val_prob[mask]
        y_g_pred = (y_g_prob >= opt_thresh).astype(int)
        
        cm = confusion_matrix(y_g, y_g_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        gender_results.append({'Gender': g, 'FPR': fpr, 'FNR': fnr, 'Recall': tp/(tp+fn)})
        
    pd.DataFrame(gender_results).to_csv('reports/machine_learning/tables/gender_comparison.csv', index=False)

if __name__ == "__main__":
    main()

