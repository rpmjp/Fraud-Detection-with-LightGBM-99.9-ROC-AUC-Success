# 🛡️ Fraud Detection in Financial Transactions

An end-to-end machine learning project to detect fraudulent transactions using real-world financial data. The pipeline includes advanced feature engineering, imbalance handling with SMOTE, explainability via SHAP, and rigorous evaluation using multiple classifiers.

---

## 📁 Dataset

**Source:** [Kaggle - Financial Transactions Dataset for Fraud Detection](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection)

- **Rows:** 6.3 million transactions  
- **Target:** `isFraud` (binary)
- **Key Features:** `amount`, `type`, `nameOrig`, `oldbalanceOrg`, `nameDest`, etc.

---

## 🧠 Project Pipeline

### 1. 📊 **Exploratory Data Analysis (EDA)**
- Visualized fraud distribution and rates by transaction type
- Detected key patterns in amount, time, and behavior

### 2. 🧼 **Data Cleaning**
- Removed leakage-prone features (`newbalance*`, `isFlaggedFraud`)
- Verified no missing or duplicated values

### 3. 🛠️ **Feature Engineering**
- Derived features like:
  - `delta_orig`, `delta_dest` (balance movement)
  - `is_weekend`, `is_night` (temporal fraud signals)
  - `sender_avg_amount`, `sender_txn_count` (historical behavior)
  - `balance_jump_flag`, `delta_orig_ratio`, `is_c_to_c`

### 4. ⚖️ **Class Imbalance Handling**
- Applied **SMOTE** to oversample fraud class (3.5%)
- Used `class_weight='balanced'` where applicable

### 5. 🤖 **Modeling**
| Model            | Accuracy | Recall (Fraud) | ROC-AUC |
|------------------|----------|----------------|---------|
| LogisticRegression | 42%     | 90.9%         | 0.82    |
| Random Forest      | 99.9%   | 99.6%         | 0.999   |
| **LightGBM** ✅     | 99.9%   | **99.9%**     | **0.9999** |
| XGBoost            | 99.9%   | 99.4%         | 0.9998  |

### 6. 🔍 **Explainability**
- Used **SHAP** TreeExplainer on LightGBM
- Visualized:
  - Summary plots (global feature importance)
  - Waterfall plots (local prediction breakdown)
- Top contributing features: `amount`, `sender_avg_amount`, `delta_orig`, `step`

### 7. ✅ **Final Evaluation**
- Performed on **untouched test set**
- Metrics:
  - Precision (Fraud): 0.9885
  - Recall (Fraud): 0.9988
  - F1-Score: 0.9936
  - ROC-AUC: 0.9994

---

## 📌 Key Takeaways

- SMOTE and behavioral features greatly improve fraud recall
- LightGBM provides the best trade-off between speed, accuracy, and interpretability
- SHAP ensures model transparency, critical for financial audits

---

## 💼 Use Cases

- Online banking fraud detection
- Payment gateway anomaly detection
- Real-time transaction monitoring systems

---

## 🧰 Tech Stack

- **Python**, **NumPy**, **Pandas**
- **scikit-learn**, **LightGBM**, **XGBoost**
- **SHAP**, **Seaborn**, **Matplotlib**
- Jupyter Notebook (local execution)

---

## 📂 Folder Structure

