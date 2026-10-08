import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os
import shap
from copy import deepcopy
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import build_lgbm_pipeline, tune_lgbm, get_features_target, build_baseline_pipeline
from src.evaluate import evaluate_model, get_optimal_threshold, plot_roc_curve, plot_pr_curve, plot_confusion_matrix, plot_cost_curve

# Create output dirs
os.makedirs('reports/machine_learning/tables', exist_ok=True)
os.makedirs('reports/machine_learning/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

def main():
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    features, target = get_features_target(train_df)
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_val = val_df[features]
    y_val = val_df[target]
    
    print("Tuning LightGBM (15 trials for speed)...")
    best_params = tune_lgbm(X_train, y_train, n_trials=15)
    
    print("Training best model...")
    pipeline = build_lgbm_pipeline(best_params)
    pipeline.fit(X_train, y_train)
    
    # Save model
    joblib.dump(pipeline, 'models/best_model.joblib')
        
    print("Evaluating on validation...")
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    
    # Cost analysis
    opt_thresh, min_cost, thresholds, costs = get_optimal_threshold(y_val, y_val_prob, cost_fn=5, cost_fp=1)
    
    # Threshold cost analysis CSV
    pd.DataFrame({'threshold': thresholds, 'expected_cost': costs}).to_csv('reports/machine_learning/tables/threshold_cost_analysis.csv', index=False)
    plot_cost_curve(thresholds, costs, opt_thresh, save_path='reports/machine_learning/figures/cost_curve.png')
    
    opt_metrics = evaluate_model(y_val, y_val_prob, threshold=opt_thresh)
    
    # Export Model Comparison (Baseline vs Best)
    baseline_model = joblib.load('models/baseline_logreg.joblib')
        
    base_val_prob = baseline_model.predict_proba(X_val)[:, 1]
    base_opt_thresh, base_min_cost, _, _ = get_optimal_threshold(y_val, base_val_prob)
    base_metrics = evaluate_model(y_val, base_val_prob, threshold=base_opt_thresh)
    
    comparison = pd.DataFrame([
        {'Model': 'Baseline (LogReg)', 'AUC-ROC': base_metrics['AUC-ROC'], 'AUC-PR': base_metrics['AUC-PR'], 'Min_Cost': base_min_cost, 'Threshold': base_opt_thresh},
        {'Model': 'Best (LightGBM)', 'AUC-ROC': opt_metrics['AUC-ROC'], 'AUC-PR': opt_metrics['AUC-PR'], 'Min_Cost': min_cost, 'Threshold': opt_thresh}
    ])
    comparison.to_csv('reports/machine_learning/tables/model_comparison.csv', index=False)
    
    # Plots for best model
    plot_roc_curve(y_val, y_val_prob, save_path='reports/machine_learning/figures/roc_curve.png')
    plot_pr_curve(y_val, y_val_prob, save_path='reports/machine_learning/figures/pr_curve.png')
    plot_confusion_matrix(opt_metrics['Confusion_Matrix'], save_path='reports/machine_learning/figures/confusion_matrix.png')
    
    print("SHAP explanation...")
    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']
    
    # Scale data for SHAP
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=features)
    
    # SHAP explainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_val_scaled)
    
    # Mean absolute SHAP value for importance
    shap_imp = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    shap_imp.to_csv('reports/machine_learning/tables/shap_importance.csv', index=False)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_val_scaled, show=False)
    plt.savefig('reports/machine_learning/figures/shap_summary.png', bbox_inches='tight')
    plt.close()
    
    # Gender Fairness test
    print("Testing gender fairness...")
    # Get meta_gender
    meta_gender_val = val_df['meta_gender']
    
    gender_results = []
    for g in [1, 2]: # Assuming 1 is male, 2 is female based on docs
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
    
    print("Done!")

if __name__ == "__main__":
    main()

