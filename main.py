import os
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import clean_data, encode_data
from src.data_processor import split_and_scale
from src.models import train_logistic_regression, train_random_forest
from src.evaluation import full_evaluation

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================
# Paths
# =========================
DATA_PATH = "data/Telco.csv"
FIG_DIR = "results/figures"
METRIC_DIR = "results/metrics"


# =========================
# Pretty Printing Helpers
# =========================
def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_model_result(idx, m):
    name = m["model"].replace("_", " ").title()
    print(f"\n[{idx}] {name}")
    print("-" * 60)
    print(f"ROC–AUC      : {m['roc_auc']:.4f}")
    print(f"Accuracy     : {m['accuracy']:.4f}")
    print(f"Precision(1) : {m['precision_1']:.4f}")
    print(f"Recall(1)    : {m['recall_1']:.4f}")
    print(f"F1-score(1)  : {m['f1_1']:.4f}")


# =========================
# Main Pipeline
# =========================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    # -------- Load & preprocess --------
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    results = []

    # ============================================================
    # 1. Logistic Regression
    # ============================================================
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)

    log_model = train_logistic_regression(X_train_scaled, y_train)
    results.append(
        full_evaluation(
            log_model,
            X_test_scaled,
            y_test,
            "logistic_regression",
            FIG_DIR
        )
    )

    # ============================================================
    # 2. Random Forest (Default)
    # ============================================================
    rf_model = train_random_forest(X_train, y_train)
    results.append(
        full_evaluation(
            rf_model,
            X_test,
            y_test,
            "random_forest",
            FIG_DIR
        )
    )

    # ============================================================
    # 3. Random Forest (SMOTE + Class Weight)
    # ============================================================
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    X_train_s, X_test_s, _, _, y_train_s, y_test_s = split_and_scale(X_sm, y_sm)

    rf_balanced = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    rf_balanced.fit(X_train_s, y_train_s)

    results.append(
        full_evaluation(
            rf_balanced,
            X_test_s,
            y_test_s,
            "rf_smote_balanced",
            FIG_DIR
        )
    )

    # ============================================================
    # 4. XGBoost
    # ============================================================
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train_s, y_train_s)

    results.append(
        full_evaluation(
            xgb,
            X_test_s,
            y_test_s,
            "xgboost",
            FIG_DIR
        )
    )

    # ============================================================
    # Save Metric Reports
    # ============================================================
    for r in results:
        with open(f"{METRIC_DIR}/{r['model']}.txt", "w") as f:
            f.write(f"ROC-AUC: {r['roc_auc']:.4f}\n\n")
            f.write(r["report"])

    # ============================================================
    # Pretty Terminal Output
    # ============================================================
    print_header("TELCO CUSTOMER CHURN – MODEL EVALUATION")

    for i, res in enumerate(results, start=1):
        print_model_result(i, res)

    # ============================================================
    # Final Comparison Table
    # ============================================================
    summary_df = (
        pd.DataFrame(results)[
            ["model", "roc_auc", "recall_1", "f1_1"]
        ]
        .rename(columns={
            "model": "Model",
            "roc_auc": "ROC–AUC",
            "recall_1": "Recall(1)",
            "f1_1": "F1(1)"
        })
        .sort_values("ROC–AUC", ascending=False)
    )

    summary_df.to_csv(f"{METRIC_DIR}/summary.csv", index=False)

    print_header("FINAL MODEL COMPARISON (Sorted by ROC–AUC)")
    print(summary_df.to_string(index=False))


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()
