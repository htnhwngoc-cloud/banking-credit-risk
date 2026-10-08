import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.train import get_features_target
from src.evaluate import evaluate_model, plot_roc_curve, plot_pr_curve, plot_confusion_matrix, calculate_expected_costs

def main():
    print("Loading test data and model...")
    test_df = pd.read_csv('data/processed/test.csv')
    features, target = get_features_target(test_df)
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    pipeline = joblib.load('models/best_model.joblib')
    
    # Get the exact threshold found on validation set
    # It's saved in reports/machine_learning/tables/model_comparison.csv
    comp_df = pd.read_csv('reports/machine_learning/tables/model_comparison.csv')
    best_row = comp_df[comp_df['Model'] == 'Best (LightGBM)'].iloc[0]
    opt_thresh = best_row['Threshold']
    
    print(f"Using frozen optimal threshold: {opt_thresh:.4f}")
    
    # Predict
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = evaluate_model(y_test, y_test_prob, threshold=opt_thresh)
    
    # Calculate Cost
    cost_array = calculate_expected_costs(y_test, y_test_prob, [opt_thresh], cost_fn=5, cost_fp=1)
    metrics['Expected_Cost'] = cost_array[0]
    
    print("\nFINAL TEST RESULTS:")
    results_list = []
    for k, v in metrics.items():
        if k != 'Confusion_Matrix':
            print(f"{k}: {v:.4f}")
            results_list.append({'Metric': k, 'Value': v})
            
    # Export metrics to CSV
    pd.DataFrame(results_list).to_csv('reports/machine_learning/tables/test_results.csv', index=False)
    
    # Export final plots
    os.makedirs('reports/machine_learning/figures/test', exist_ok=True)
    plot_roc_curve(y_test, y_test_prob, save_path='reports/machine_learning/figures/test/roc_curve_test.png')
    plot_pr_curve(y_test, y_test_prob, save_path='reports/machine_learning/figures/test/pr_curve_test.png')
    plot_confusion_matrix(metrics['Confusion_Matrix'], save_path='reports/machine_learning/figures/test/confusion_matrix_test.png')
    
    print("Test evaluation finished and saved.")

if __name__ == "__main__":
    main()

