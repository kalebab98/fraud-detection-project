# fraud-detection-project
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
