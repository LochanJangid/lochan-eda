import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

from lochan_eda.utils import get_iqr_bounds, get_active_cols


class HandleNumerical:
    def __init__(self, df):
        self.num_df = df.select_dtypes(include=["number"]).copy()

        self.drop_cols_ = []
        self.impute_values_ = {}
        self.outlier_rules_ = {}
        self.scalers_ = {}
        self.transforms_ = {}

    def num_imputer(self, is_train=True, exclude=None):
        """impute missing values based on their data behaviour."""
        active_cols = get_active_cols(self.num_df.columns, exclude)

        # Runs on training and memorize drop_cols & impute values
        if is_train: 
            missing_prcnt = self.num_df[active_cols].isna().mean() * 100
            # drop columns with > 40% missing data
            self.drop_cols_ = missing_prcnt[missing_prcnt > 40].index.tolist()

            active_cols = [col for col in active_cols if col not in self.drop_cols_]

            if active_cols:
                uniquness = (self.num_df[active_cols].nunique() / self.num_df[active_cols].shape[0]) * 100
                like_cat = uniquness[uniquness < 1].index

                for col in like_cat:
                    if self.num_df[col].isna().sum() > 0:
                        self.impute_values_[col] = self.num_df[col].mode().iloc[0]

                simple_cols = [col for col in active_cols if col not in like_cat]

                for col in simple_cols:
                    if missing_prcnt[col] > 0:
                        skewness = self.num_df[col].skew()
                        fill_val = self.num_df[col].mean() if abs(skewness) < 0.5 else self.num_df[col].median()
                        self.impute_values_[col] = fill_val
        
        # Main Work apply memorized things 

        if self.drop_cols_:
            cols_to_drop = [c for c in self.drop_cols_ if c in self.num_df.columns]
            self.num_df.drop(columns=cols_to_drop, inplace=True)

        for col, fill_val in self.impute_values_.items():
            if col in self.num_df.columns:
                self.num_df[col] = self.num_df[col].fillna(fill_val)

        return self.num_df

    def outlier_manager(self, is_train=True, exclude=None):
        active_cols = get_active_cols(self.num_df.columns, exclude)

        if is_train:
            for col in active_cols:
                lower_bound, upper_bound = get_iqr_bounds(self.num_df[col])
                outliers_prcnt = ((self.num_df[col] > upper_bound) | (self.num_df[col] < lower_bound)).mean() * 100

                if 0 < outliers_prcnt <= 3.0:
                    self.outlier_rules_[col] = {'type': 'clip', 'lower': lower_bound, 'upper': upper_bound}
                elif outliers_prcnt > 3.0:
                    p90 = self.num_df[col].quantile(0.90)
                    p99 = self.num_df[col].quantile(0.99)
                    max_val = self.num_df[col].max()
                    gap = (p99 - p90) / (max_val - p99 + 1e-9)

                    is_negative = (self.num_df[col] < 0).any()
                    
                    if gap <= 1.0 or is_negative:
                        p5 = self.num_df[col].quantile(0.05)
                        p95 = self.num_df[col].quantile(0.95)
                        self.outlier_rules_[col] = {'type': 'clip', 'lower': p5, 'upper': p95}
                    else:
                        is_zero = (self.num_df[col] == 0).any()
                        self.outlier_rules_[col] = {'type': 'sqrt' if is_zero else 'log1p'}

        # Work (Train and Test)
        for col, rule in self.outlier_rules_.items():
            if col in self.num_df.columns:
                if rule['type'] == 'clip':
                    self.num_df[col] = self.num_df[col].clip(lower=rule['lower'], upper=rule['upper'])
                elif rule['type'] == 'sqrt':
                    self.num_df[col] = np.sqrt(self.num_df[col])
                elif rule['type'] == 'log1p':
                    self.num_df[col] = np.log1p(self.num_df[col])

        return self.num_df

    def scaler(self, is_train=True, exclude=None):
        active_cols = get_active_cols(self.num_df.columns, exclude)

        if is_train:
            for col in active_cols:
                sparsity = (self.num_df[col] == 0).mean()
                skewness = self.num_df[col].skew()
                lower_bound, upper_bound = get_iqr_bounds(self.num_df[col])
                outliers_ratio = ((self.num_df[col] > upper_bound) | (self.num_df[col] < lower_bound)).mean()

                if sparsity >= 0.5:
                    scaler_obj, fit_data = MaxAbsScaler(), self.num_df[[col]]
                elif skewness > 1.0 and self.num_df[col].min() >= 0:
                    self.transforms_[col] = "log1p"
                    scaler_obj, fit_data = StandardScaler(), np.log1p(self.num_df[[col]])
                elif outliers_ratio >= 0.05:
                    scaler_obj, fit_data = RobustScaler(), self.num_df[[col]]
                else:
                    scaler_obj, fit_data = StandardScaler(), self.num_df[[col]]

                scaler_obj.fit(fit_data)
                self.scalers_[col] = scaler_obj

        # Apply rules (Train & Test)
        for col, scaler_obj in self.scalers_.items():
            if col in self.num_df.columns:

                data = self.num_df[[col]]
                
                if self.transforms_.get(col) == "log1p":
                    data = np.log1p(data)
                
                self.num_df[col] = scaler_obj.transform(data).flatten()

        return self.num_df

    def full_handler(self, is_train=True):
        self.num_imputer(is_train=is_train)
        self.outlier_manager(is_train=is_train)
        self.scaler(is_train=is_train)
        return self.num_df