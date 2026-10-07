import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def get_features_target(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['default', 'meta_gender']
    
    features = [col for col in df.columns if col not in exclude_cols]
    target = 'default'
    return features, target

def build_baseline_pipeline():
    """Tạo pipeline Logistic Regression làm Baseline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    return pipeline

def build_lgbm_pipeline(params=None):
    """Tạo pipeline LightGBM."""
    if params is None:
        params = {}
    
    # LGBM xử lý tree-based nên scaler không bắt buộc, nhưng đưa vào pipeline để chuẩn form
    lgbm = LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=100,
        verbose=-1,
        **params
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Scaler có thể không cần thiết nhưng giữ lại cho nhất quán
        ('classifier', lgbm)
    ])
    return pipeline

def tune_lgbm(X, y, n_trials=20):
    """Dò tìm siêu tham số cho LightGBM bằng Optuna và StratifiedKFold."""
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        pipeline = build_lgbm_pipeline(params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Optimize for roc_auc
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    
    return study.best_trial.params
