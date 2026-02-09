# 📊 Telco Customer Churn Prediction

## 📌 Project Overview

Customer churn refers to customers discontinuing a service. Accurately predicting churn allows businesses to take proactive retention measures and reduce revenue loss.

In this project, we build a **tabular machine learning pipeline** to predict whether a customer will churn (`Yes/No`) using the **Telco Customer Churn dataset** from Kaggle. Multiple models are trained and evaluated with proper handling of **class imbalance**, and results are compared using robust evaluation metrics.

---

## 📂 Dataset

* **Source:** Kaggle – Telco Customer Churn
* **Samples:** ~7,000 customers
* **Target Variable:** `Churn`

  * `0` → No churn
  * `1` → Churn
* **Features:**

  * Customer demographics (gender, senior citizen)
  * Account information (tenure, contract type)
  * Services used (internet, phone, streaming)
  * Billing information (MonthlyCharges, TotalCharges)

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

1. Replaced blank values (`" "`) with missing values
2. Converted `TotalCharges` to numeric
3. Imputed missing values:

   * Numerical features → median
   * Categorical features → mode
4. Dropped non-informative identifier (`customerID`)
5. Encoded categorical features using **Label Encoding**
6. Applied **feature scaling** (StandardScaler) for Logistic Regression

---

## 🔀 Train / Validation Split Method

* **Split Strategy:** Stratified Train–Test Split
* **Train Size:** 80%
* **Test Size:** 20%
* **Reason:**

  * Stratification preserves the original churn class distribution
  * Prevents biased evaluation due to class imbalance

```text
Train set : 80%
Test set  : 20%
Stratify  : Target variable (Churn)
```

Additionally, for imbalance handling:

* **SMOTE (Synthetic Minority Oversampling Technique)** was applied **only on the training data** for selected models.

---

## 🤖 Models Trained

The following models were trained and compared:

1. **Logistic Regression (Baseline)**
2. **Random Forest (Default)**
3. **Random Forest (SMOTE + Class Weight)**
4. **XGBoost Classifier**

---

## 📏 Evaluation Metrics

For each model, we report **multiple evaluation metrics** to ensure a fair and business-relevant comparison.

### Metrics Used

* **Accuracy**
* **Precision (Class = Churn)**
* **Recall (Class = Churn)** ⭐
* **F1-score (Class = Churn)**
* **ROC–AUC Score**
* **Confusion Matrix**
* **ROC Curve**

📌 **Why Recall Matters Most:**
In churn prediction, **missing a churner is more costly** than incorrectly flagging a loyal customer. Therefore, recall and F1-score for the churn class are prioritized over accuracy.

All confusion matrices and ROC curves are automatically saved to:

```
results/figures/
```

---

## 🏆 Best Result

### 🔥 Best Performing Model: **XGBoost**

| Metric           | Value      |
| ---------------- | ---------- |
| ROC–AUC          | **0.9261** |
| Recall (Churn)   | **~0.88**  |
| F1-score (Churn) | **~0.86**  |
| Accuracy         | ~0.85      |

### Why XGBoost Performed Best

* Handles non-linear relationships effectively
* Robust to feature interactions
* Performs well on tabular data
* Combined with SMOTE, it achieved excellent churn recall and class separation

---

## 📊 Model Comparison Summary (ROC–AUC)

| Model                            | ROC–AUC   |
| -------------------------------- | --------- |
| **XGBoost**                      | **0.926** |
| Random Forest (SMOTE + Balanced) | 0.920     |
| Logistic Regression              | 0.840     |
| Random Forest (Default)          | 0.826     |

---

## 📁 Project Structure

```
telco-churn-project/
│
├── data/
│   └── Telco.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── data_processor.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python main.py
```

All metrics, plots, and reports will be generated automatically.

---

## ✅ Key Takeaways

* Handling class imbalance is **critical** for churn prediction
* ROC–AUC provides a more reliable performance measure than accuracy alone
* XGBoost with SMOTE delivered **production-grade performance**
* The project follows clean, modular, and reproducible ML practices

