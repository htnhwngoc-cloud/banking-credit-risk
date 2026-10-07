import pandas as pd
import joblib
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import build_baseline_pipeline, get_features_target
from src.evaluate import evaluate_model, get_optimal_threshold

def main():
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    features, target = get_features_target(train_df)
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_val = val_df[features]
    y_val = val_df[target]
    
    print("Building baseline model...")
    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/baseline_logreg.joblib')
    
    # Predict on validation
    print("Evaluating on validation...")
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    
    # Get standard metrics at threshold 0.5
    metrics = evaluate_model(y_val, y_val_prob, threshold=0.5)
    print("\nMetrics at threshold 0.5:")
    for k, v in metrics.items():
        if k != 'Confusion_Matrix':
            print(f"{k}: {v:.4f}")
            
    # Calculate optimal threshold based on cost 5:1
    opt_thresh, min_cost, _, _ = get_optimal_threshold(y_val, y_val_prob, cost_fn=5, cost_fp=1)
    
    print(f"\nCost Analysis (Cost FN=5, Cost FP=1):")
    print(f"Optimal Threshold: {opt_thresh:.4f}")
    print(f"Minimum Expected Cost: {min_cost:.2f}")
    
    # Eval at optimal threshold
    opt_metrics = evaluate_model(y_val, y_val_prob, threshold=opt_thresh)
    print(f"\nMetrics at Optimal Threshold ({opt_thresh:.4f}):")
    for k, v in opt_metrics.items():
        if k != 'Confusion_Matrix':
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
