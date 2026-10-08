"""
scratch/gen_policy_simulation.py
Tạo policy_simulation.csv và early_warning_groups.csv
Chỉ chạy trên tập VALIDATION - KHÔNG dùng test set.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)

COST_SCENARIOS = [
    {"name": "Mo rong (3:1)",    "cost_fn": 3,  "cost_fp": 1, "color": "#4CAF50"},
    {"name": "Co so (5:1)",      "cost_fn": 5,  "cost_fp": 1, "color": "#2196F3"},
    {"name": "That chat (10:1)", "cost_fn": 10, "cost_fp": 1, "color": "#F44336"},
]
N_THRESHOLDS = 100
OUT_TABLES = "reports/machine_learning/tables"
OUT_FIGURES = "reports/machine_learning/figures"


def compute_policy(y_true, y_prob, cost_fn, cost_fp):
    thresholds = np.linspace(0.01, 0.99, N_THRESHOLDS)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            costs.append(fn * cost_fn + fp * cost_fp)
        else:
            costs.append(np.nan)

    costs = np.array(costs)
    opt_idx = int(np.nanargmin(costs))
    opt_thresh = float(thresholds[opt_idx])
    opt_cost = float(costs[opt_idx])

    y_pred_opt = (y_prob >= opt_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()

    return {
        "optimal_threshold": round(opt_thresh, 4),
        "min_expected_cost": int(opt_cost),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(precision_score(y_true, y_pred_opt, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred_opt, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred_opt, zero_division=0), 4),
        "thresholds": thresholds.tolist(),
        "costs":      costs.tolist(),
    }


def make_policy_figure(all_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("So sanh 3 Kich ban Chinh sach Nguong (Tap Validation)",
                 fontsize=14, fontweight="bold")

    for ax, sc, res in zip(axes, COST_SCENARIOS, all_results):
        thresholds = res["thresholds"]
        costs      = res["costs"]
        ax.plot(thresholds, costs, linewidth=2, color=sc["color"])
        ax.axvline(res["optimal_threshold"], color="black", linestyle="--",
                   linewidth=1.5,
                   label=f"Nguong toi uu: {res['optimal_threshold']:.4f}")
        ax.scatter([res["optimal_threshold"]], [res["min_expected_cost"]],
                   color="black", zorder=5, s=60)
        ax.set_title(sc["name"], fontsize=12, color=sc["color"])
        ax.set_xlabel("Nguong quyet dinh")
        ax.set_ylabel("Chi phi ky vong (don vi)")
        ax.legend(fontsize=8)
        ax.annotate(
            f"Cost={res['min_expected_cost']}\nRecall={res['recall']:.3f}",
            xy=(res["optimal_threshold"], res["min_expected_cost"]),
            xytext=(res["optimal_threshold"] + 0.06,
                    res["min_expected_cost"] + 120),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {out_path}")


def make_early_warning_groups():
    pay_sep_df = pd.read_csv("reports/analyze/tables/default_rate_by_pay_sep.csv")
    eng_df     = pd.read_csv("reports/analyze/tables/default_rate_by_engineered.csv")
    limit_df   = pd.read_csv("reports/analyze/tables/default_rate_by_limit_group.csv")

    ew1 = pay_sep_df[pay_sep_df["pay_sep"] == 1].iloc[0]
    ew2_rows = eng_df[eng_df["feature"] == "delay_trend_bin"]
    ew2 = ew2_rows.iloc[-1]  # nhom co delay_trend cao nhat
    ew3 = limit_df[limit_df["limit_group"] == 0].iloc[0]

    rows = [
        {
            "group_id":           "EW-1",
            "description":        "Bat dau tre han (pay_sep = 1)",
            "signal":             "pay_sep == 1",
            "total_customers":    int(ew1["total"]),
            "default_count":      int(ew1["defaults"]),
            "default_rate":       round(float(ew1["default_rate"]), 4),
            "ci_lower_95":        round(float(ew1["ci_lower_95"]), 4),
            "ci_upper_95":        round(float(ew1["ci_upper_95"]), 4),
            "recommended_action": "Nhac nho tu dong, tu van tra no",
        },
        {
            "group_id":           "EW-2",
            "description":        "Xu huong tre han tang (delay_trend > 0.143)",
            "signal":             "delay_trend > 0.143",
            "total_customers":    int(ew2["total"]),
            "default_count":      int(ew2["defaults"]),
            "default_rate":       round(float(ew2["default_rate"]), 4),
            "ci_lower_95":        round(float(ew2["ci_lower_95"]), 4),
            "ci_upper_95":        round(float(ew2["ci_upper_95"]), 4),
            "recommended_action": "Han che han muc tam thoi, tu van tai co cau no",
        },
        {
            "group_id":           "EW-3",
            "description":        "Han muc cuc thap Q1 (<=50.000 NTD)",
            "signal":             "limit_group == 0",
            "total_customers":    int(ew3["total"]),
            "default_count":      int(ew3["defaults"]),
            "default_rate":       round(float(ew3["default_rate"]), 4),
            "ci_lower_95":        round(float(ew3["ci_lower_95"]), 4),
            "ci_upper_95":        round(float(ew3["ci_upper_95"]), 4),
            "recommended_action": "Review uu tien hang tuan, can thiep chu dong",
        },
    ]

    out = os.path.join(OUT_TABLES, "early_warning_groups.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved: {out}")
    return pd.DataFrame(rows)


def main():
    print("=== Tao policy_simulation.csv ===")
    val_df = pd.read_csv("data/processed/val.csv")
    model  = joblib.load("models/best_model.joblib")
    X_val  = val_df.drop(columns=["default", "meta_gender"])
    y_val  = val_df["default"]
    y_prob = model.predict_proba(X_val)[:, 1]

    rows = []
    all_results = []
    for sc in COST_SCENARIOS:
        print(f"  Kich ban: {sc['name']} ...")
        res = compute_policy(y_val, y_prob, sc["cost_fn"], sc["cost_fp"])
        all_results.append(res)
        rows.append({
            "scenario":          sc["name"],
            "cost_fn":           sc["cost_fn"],
            "cost_fp":           sc["cost_fp"],
            "optimal_threshold": res["optimal_threshold"],
            "min_expected_cost": res["min_expected_cost"],
            "tp": res["tp"], "fp": res["fp"], "fn": res["fn"], "tn": res["tn"],
            "precision": res["precision"],
            "recall":    res["recall"],
            "f1":        res["f1"],
        })

    policy_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_TABLES, "policy_simulation.csv")
    policy_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    print()
    print(policy_df[["scenario", "optimal_threshold", "min_expected_cost",
                      "precision", "recall", "f1"]].to_string(index=False))

    print("\nTao bieu do policy_simulation.png ...")
    fig_path = os.path.join(OUT_FIGURES, "policy_simulation.png")
    make_policy_figure(all_results, fig_path)

    print("\n=== Tao early_warning_groups.csv ===")
    ew_df = make_early_warning_groups()
    print(ew_df[["group_id", "default_rate", "recommended_action"]].to_string(index=False))
    print("\nHoan thanh!")


if __name__ == "__main__":
    main()
