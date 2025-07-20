# fraud-detection-project

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

## 10. Next Steps

- Proceed to model building and evaluation (Task 2).

--- 