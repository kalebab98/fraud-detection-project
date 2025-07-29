# Fraud-detection-project

# Task 1: Data Analysis and Preprocessing

## Quick Summary

### Data Cleaning
- Started with 151,112 rows; no duplicates or missing values were found, so the row count stayed the same.
- All time columns were converted to datetime format.
- IP addresses were safely converted from float to integer for mapping.
- No data was lost during cleaning.

### EDA Highlights
- The dataset is imbalanced:
  - Non-Fraudulent: 136,961 (90.6%)
  - Fraudulent: 14,151 (9.4%)
- Purchase Value (standardized):
  - Mean: ~0.00, Median: -0.11, Min: -1.52, Max: 6.39
- Time Since Signup (seconds):
  - Mean: 4,932,029, Median: 4,926,346, Min: 1, Max: 10,367,970

---

## 1. Data Loading

**Datasets Used:**
- `Fraud_Data.csv`: E-commerce transaction data for fraud detection.
- `IpAddress_to_Country.csv`: IP address range to country mapping.
- `creditcard.csv`: Bank transaction data for fraud detection.

**Example Code:**
```python
import pandas as pd
fraud_data = pd.read_csv('data/Fraud_Data.csv')
```

---

## 2. Data Cleaning

- Removed duplicate rows to ensure data quality.
- Converted time columns to datetime objects for accurate calculations.
- Checked for missing values in all columns and dropped rows with missing IP addresses.

---

## 3. Exploratory Data Analysis (EDA)

- Visualized class distribution and purchase value distribution.

**Example Code:**
```python
import seaborn as sns
sns.countplot(x='class', data=fraud_data)
```

Key Insights:
- The dataset is highly imbalanced, with far fewer fraudulent transactions.

---

## 4. Feature Engineering

- Created time-based features: `time_since_signup`, `hour_of_day`, and `day_of_week`.
- Added `user_transaction_count` (number of transactions per user).

---

## 5. Geolocation Merge

- Converted IP addresses to integer format and mapped to country using IP range data.

---

## 6. Data Transformation

- One-hot encoded categorical variables (`source`, `browser`, `sex`, `country`).
- Scaled numerical features (`purchase_value`, `time_since_signup`).
- Used SMOTE to address class imbalance in the training set.

---

## 7. Visualizations

- Plotted class distribution, purchase value distribution, and time since signup by class.

---

## 8. Summary of Preprocessing Steps

1. Loaded and cleaned data.
2. Engineered new features for time and transaction frequency.
3. Mapped IP addresses to countries.
4. Encoded categorical variables.
5. Scaled numerical features.
6. Addressed class imbalance with SMOTE.

---

## 9. Challenges and Solutions

- Class imbalance was addressed with SMOTE to improve model training.
- Potential data quality issues were checked for negative values and confirmed they were due to scaling, not data errors.

---

Here’s a **GitHub-ready `README.md`** specifically tailored for **Task 2 - Model Building and Evaluation** of your fraud detection project. It’s concise, professional, and clearly communicates your work for the reviewers or teammates:

---


# Task 2: Fraud Detection - Model Building and Evaluation

## 📌 Overview

This repository contains model training and evaluation code for detecting fraudulent transactions in both e-commerce (`Fraud_Data.csv`) and banking (`creditcard.csv`) datasets. This is part of an ongoing project at **Adey Innovations Inc.**, aimed at building robust fraud detection systems that are accurate, interpretable, and business-friendly.

We implemented two supervised classification models:
- **Logistic Regression** – as a simple, interpretable baseline
- **Random Forest** – as a powerful ensemble model

The focus was on handling imbalanced classes, selecting appropriate evaluation metrics, and visualizing model performance effectively.


````

````
---

## ⚙️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fraud-detection-project.git
   cd fraud-detection-project
````
````
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. Run model training:

   ```bash
   python -m scripts.model_training
   ```

---

## 🧠 Models Trained

### 1. Logistic Regression

* Used as a simple baseline.
* Scikit-learn's `LogisticRegression(max_iter=1000)` used with default parameters.
* ROC-AUC Score: `~0.73`
* Limitations: Struggled with recall on minority class.

### 2. Random Forest

* Scikit-learn's `RandomForestClassifier(n_estimators=100, random_state=42)`
* ROC-AUC Score: `~0.76`
* Strong performance on both precision and recall.
* Better suited for handling non-linearities in imbalanced data.

---

## 📊 Evaluation Metrics

Since the data is highly imbalanced, the following metrics were emphasized:

* **Confusion Matrix**: To visualize true positives vs false positives.
* **Classification Report**: Includes precision, recall, F1-score.
* **ROC-AUC Score**: Measures ability to distinguish between classes.
* **ROC Curve**: Visual comparison between models.

### 🖼️ Sample Results (E-Commerce Fraud Data)

#### Logistic Regression

* **Confusion Matrix**

  ```
  [[21569  5824]
   [1074  1756]]
  ```
* **Recall (fraud):** 0.62
* **ROC-AUC:** 0.733

#### Random Forest

* **Confusion Matrix**

  ```
  [[27235   158]
   [1330  1500]]
  ```
* **Recall (fraud):** 0.53
* **ROC-AUC:** 0.765

---

# Task 3: Model Explainability with SHAP

## 📌 Objective

The goal of Task 3 is to interpret the best-performing fraud detection model using **SHAP (SHapley Additive exPlanations)**. While predictive accuracy is important, model transparency is critical—especially in fraud detection, where false positives and false negatives have serious business consequences.

By applying SHAP, we aim to:
- Identify which features most influence fraud predictions.
- Explain why individual transactions are classified as fraudulent or legitimate.
- Provide actionable insights and communicate results to stakeholders.

---

## 🧠 What is SHAP?

SHAP is a game-theoretic approach to explain the output of any machine learning model. It assigns each feature an importance value for a particular prediction, based on Shapley values from cooperative game theory.

Key benefits:
- **Global explanations**: Understand which features the model relies on most overall.
- **Local explanations**: Understand why a specific transaction was predicted as fraud or not.

---

## 📊 Summary of SHAP Visualizations

### ✅ SHAP Summary Plot

- This plot visualizes feature importance across all samples.
- **Top impactful features** included:
  - `purchase_time`: Strong influence in both directions, indicating timing is critical for identifying fraud.
  - `signup_time` & `user_id`: Certain user patterns or signup times are strong indicators.
  - `age`, `purchase_value`, `ip_address`: Have moderate influence.

**Insight**: The model captures behavioral signals like the transaction time and signup behavior rather than demographic data alone.
![photo_2025-07-29_21-19-19](https://github.com/user-attachments/assets/b1e44cf8-c242-4016-9d5a-8b12a200df22)

![photo_2025-07-29_21-19-22](https://github.com/user-attachments/assets/70d07d5e-4f5c-4cda-84e6-889ed362be51)

![photo_2025-07-29_21-19-24](https://github.com/user-attachments/assets/845076fd-26ec-41a6-8025-fd4824309f44)


---

### ✅ SHAP Force Plot (Local Explanation)

- This plot shows how a single prediction was made by summing feature contributions:
  - **Feature 2**: Largest negative effect (↓ likelihood of fraud).
  - **Feature 1**: Largest positive effect (↑ likelihood of fraud).
  - Other features had smaller additive effects.

**Insight**: Helps audit and justify specific decisions to users, regulators, or business analysts.

---

### ✅ SHAP Bar Plot (Global Importance)

- Displays mean absolute SHAP values for all features.
- **Feature 2** had the highest contribution to model decisions across the dataset.
- Other key features included Feature 0, Feature 1, and Feature 4.

**Insight**: Enables strategic prioritization—e.g., focusing security enhancements on the most fraud-sensitive transaction features.

---

## 📁 Output Files

- `force_plot.html`: Interactive SHAP force plot explaining one prediction.
- `summary_plot.png` *(optional)*: Visual summary of feature importance.
- `shap_feature_importance.png` *(optional)*: Global mean SHAP values.

---

## ✅ Business Value

SHAP increases trust in machine learning models by making decisions interpretable. For fraud detection:
- It helps prevent unnecessary user friction from false alarms.
- Allows domain experts to refine fraud rules based on insights.
- Supports regulatory compliance by making decisions auditable.

---

## 🛠 Tools Used

- `shap`
- `RandomForestClassifier` (scikit-learn)
- `SMOTE` (imbalanced-learn)
- `matplotlib`, `pandas`, `numpy`

---

## 📝 How to Reproduce

```bash
pip install shap scikit-learn imbalanced-learn pandas matplotlib
