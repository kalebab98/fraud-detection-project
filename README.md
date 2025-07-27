
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


## 🧪 Next Steps

* Task 3: Model Explainability using **SHAP** to understand the key drivers of fraud.
* Try more advanced models like **XGBoost** or **LightGBM**.
* Perform hyperparameter tuning and cross-validation.

---

## 🧠 Learning Outcomes Demonstrated

* Handled class imbalance using SMOTE.
* Built interpretable and complex models.
* Used ROC-AUC and F1-score for fair comparison.
* Automated and visualized model evaluation.

---

## 📅 Milestones

* ✅ Interim-1: Data cleaning and preprocessing
* ✅ Interim-2: Model building and evaluation
* 🔜 Final Submission: SHAP explainability and full report (by July 29, 2025)

---

## 📌 License

This project is for educational purposes under the guidance of the 10Academy program.

```
