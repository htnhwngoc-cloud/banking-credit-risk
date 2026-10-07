import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import os

def calculate_expected_costs(y_true, y_prob, thresholds, cost_fn=5, cost_fp=1):
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            total_cost = fn * cost_fn + fp * cost_fp
        else:
            total_cost = np.nan
        costs.append(total_cost)
    return np.array(costs)

def get_optimal_threshold(y_true, y_prob, cost_fn=5, cost_fp=1):
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = calculate_expected_costs(y_true, y_prob, thresholds, cost_fn, cost_fp)
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx], thresholds, costs

def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "AUC-ROC": auc,
        "AUC-PR": auc_pr,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion_Matrix": cm
    }

def plot_roc_curve(y_true, y_prob, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_curve(y_true, y_prob, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cost_curve(thresholds, costs, opt_thresh, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, costs, label='Expected Cost')
    plt.axvline(x=opt_thresh, color='r', linestyle='--', label=f'Opt Thresh = {opt_thresh:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Total Expected Cost')
    plt.title('Expected Cost vs Threshold')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
