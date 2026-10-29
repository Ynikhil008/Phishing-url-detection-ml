# src/preprocessor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os


class Preprocessor:
    """
    Handles scaling, imputation, and class balancing.
    """

    def __init__(self, scaler_type='standard', balance_method='smote'):
        self.scaler_type = scaler_type
        self.balance_method = balance_method
        self.scaler = None
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()

        # Impute missing values
        X_imp = self.imputer.fit_transform(X)

        # Scale
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        X_scaled = self.scaler.fit_transform(X_imp)

        # Balance classes
        X_bal, y_bal = self._balance(X_scaled, y)

        print(f"After balancing — Class 0: {(y_bal==0).sum()}, "
              f"Class 1: {(y_bal==1).sum()}")
        return X_bal, y_bal

    def transform(self, X: pd.DataFrame):
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp)

    def _balance(self, X, y):
        if self.balance_method == 'smote':
            sm = SMOTE(random_state=42, k_neighbors=5)
        elif self.balance_method == 'smotetomek':
            sm = SMOTETomek(random_state=42)
        else:
            return X, y
        return sm.fit_resample(X, y)

    def save(self, path='models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }, path)
        print(f"Preprocessor saved to {path}")

    def load(self, path='models/preprocessor.pkl'):
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.feature_names = data['feature_names']
        print(f"Preprocessor loaded from {path}")
