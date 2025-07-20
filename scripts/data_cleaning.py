"""
data_cleaning.py

This script contains functions for cleaning and preprocessing data for analysis.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(fraud_path, ip_path, credit_path):
    """
    Load fraud, IP, and credit data from CSV files.
    """
    fraud_data = pd.read_csv(fraud_path)
    ip_data = pd.read_csv(ip_path)
    credit_data = pd.read_csv(credit_path)
    return fraud_data, ip_data, credit_data


def clean_fraud_data(fraud_data):
    """
    Remove duplicates and convert time columns to datetime.
    """
    fraud_data = fraud_data.drop_duplicates()
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data = fraud_data[fraud_data['ip_address'].notnull()]
    return fraud_data


def add_ip_features(fraud_data, ip_data):
    """
    Convert IP to integer and map to country using IP data.
    """
    fraud_data['ip_int'] = fraud_data['ip_address'].astype(float).astype(int)
    ip_data['lower'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper'] = ip_data['upper_bound_ip_address'].astype(int)
    def map_ip_to_country(ip):
        match = ip_data[(ip_data['lower'] <= ip) & (ip_data['upper'] >= ip)]
        return match['country'].values[0] if not match.empty else 'Unknown'
    fraud_data['country'] = fraud_data['ip_int'].apply(map_ip_to_country)
    return fraud_data


def feature_engineering(fraud_data):
    """
    Add time-based and transaction frequency features.
    """
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    tx_freq = fraud_data.groupby('user_id')['purchase_time'].count().rename('user_transaction_count')
    fraud_data = fraud_data.merge(tx_freq, on='user_id')
    return fraud_data


def encode_and_scale(fraud_data):
    """
    One-hot encode categorical columns and scale numerical features.
    """
    fraud_data = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex', 'country'], drop_first=True)
    scaler = StandardScaler()
    fraud_data['purchase_value'] = scaler.fit_transform(fraud_data[['purchase_value']])
    fraud_data['time_since_signup'] = scaler.fit_transform(fraud_data[['time_since_signup']])
    return fraud_data


def split_and_balance(fraud_data):
    """
    Split data into train/test and apply SMOTE to handle class imbalance.
    Returns balanced X_train, y_train, X_test, y_test.
    """
    X = fraud_data.drop(columns=['class', 'signup_time', 'purchase_time', 'ip_address', 'device_id', 'user_id', 'ip_int'])
    y = fraud_data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    return X_train_bal, y_train_bal, X_test, y_test


def main():
    # Example usage
    fraud_data, ip_data, credit_data = load_data(
        'data/Fraud_Data.csv',
        'data/IpAddress_to_Country.csv',
        'data/creditcard.csv'
    )
    fraud_data = clean_fraud_data(fraud_data)
    fraud_data = add_ip_features(fraud_data, ip_data)
    fraud_data = feature_engineering(fraud_data)
    fraud_data = encode_and_scale(fraud_data)
    X_train_bal, y_train_bal, X_test, y_test = split_and_balance(fraud_data)
    print("Data cleaning and preprocessing complete.")

if __name__ == "__main__":
    main() 