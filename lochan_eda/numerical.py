import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

from lochan_eda.utils import get_iqr_bounds, get_active_cols


class Numerical:
    def __init__(self):
        self.data = None
        self.missing_drop_threshold = 40
        self.outlier_threshold = 3.0
        self.drop_cols_ = []
        self.impute_values_ = {}
        self.outlier_rules_ = {}
        self.scalers_ = {}
        self.transforms_ = {}

    def imputer(self, exclude=None, learn=False):
        """impute missing values based on their data behaviour."""
        active_cols = get_active_cols(self.data.columns, exclude)

        # Runs on training and memorize drop_cols & impute values
        if learn: 
            missing_prcnt = self.data[active_cols].isna().mean() * 100
            # drop columns with > threshold missing data
            self.drop_cols_ = missing_prcnt[missing_prcnt > self.missing_drop_threshold].index.tolist()

            active_cols = [col for col in active_cols if col not in self.drop_cols_]

            if active_cols:
                uniquness = (self.data[active_cols].nunique() / self.data[active_cols].shape[0]) * 100
                like_cat = uniquness[uniquness < 1].index

                for col in like_cat:
                    if self.data[col].isna().sum() > 0:
                        self.impute_values_[col] = self.data[col].mode().iloc[0]

                simple_cols = [col for col in active_cols if col not in like_cat]

                for col in simple_cols:
                    if missing_prcnt[col] > 0:
                        skewness = self.data[col].skew()
                        fill_val = self.data[col].mean() if abs(skewness) < 0.5 else self.data[col].median()
                        self.impute_values_[col] = fill_val
        
        else:
            # now Transform data based on learnt things
            if self.drop_cols_:
                cols_to_drop = [c for c in self.drop_cols_ if c in self.data.columns]
                self.data.drop(columns=cols_to_drop, inplace=True)

            for col, fill_val in self.impute_values_.items():
                if col in self.data.columns:
                    self.data[col] = self.data[col].fillna(fill_val)

    def outlier_manager(self, exclude=None, learn=False):
        active_cols = get_active_cols(self.data.columns, exclude)

        if learn:
            for col in active_cols:
                lower_bound, upper_bound = get_iqr_bounds(self.data[col])
                outliers_prcnt = ((self.data[col] > upper_bound) | (self.data[col] < lower_bound)).mean() * 100

                if 0 < outliers_prcnt <= self.outlier_threshold:
                    self.outlier_rules_[col] = {'type': 'clip', 'lower': lower_bound, 'upper': upper_bound}
                elif outliers_prcnt > self.outlier_threshold:
                    p90 = self.data[col].quantile(0.90)
                    p99 = self.data[col].quantile(0.99)
                    max_val = self.data[col].max()
                    gap = (p99 - p90) / (max_val - p99 + 1e-9)

                    is_negative = (self.data[col] < 0).any()
                    
                    if gap <= 1.0 or is_negative:
                        p5 = self.data[col].quantile(0.05)
                        p95 = self.data[col].quantile(0.95)
                        self.outlier_rules_[col] = {'type': 'clip', 'lower': p5, 'upper': p95}
                    else:
                        is_zero = (self.data[col] == 0).any()
                        self.outlier_rules_[col] = {'type': 'sqrt' if is_zero else 'log1p'}
        else:
            # now Transform data based on learnt things
            for col, rule in self.outlier_rules_.items():
                if col in self.data.columns:
                    if rule['type'] == 'clip':
                        self.data[col] = self.data[col].clip(lower=rule['lower'], upper=rule['upper'])
                    elif rule['type'] == 'sqrt':
                        self.data[col] = np.sqrt(self.data[col])
                    elif rule['type'] == 'log1p':
                        self.data[col] = np.log1p(self.data[col])

    def scaler(self, exclude=None, learn=False):
        active_cols = get_active_cols(self.data.columns, exclude)

        if learn:
            for col in active_cols:
                sparsity = (self.data[col] == 0).mean()
                skewness = self.data[col].skew()
                lower_bound, upper_bound = get_iqr_bounds(self.data[col])
                outliers_ratio = ((self.data[col] > upper_bound) | (self.data[col] < lower_bound)).mean()

                if sparsity >= 0.5:
                    scaler_obj, fit_data = MaxAbsScaler(), self.data[[col]]
                elif skewness > 1.0 and self.data[col].min() >= 0:
                    self.transforms_[col] = "log1p"
                    scaler_obj, fit_data = StandardScaler(), np.log1p(self.data[[col]])
                elif outliers_ratio >= 0.05:
                    scaler_obj, fit_data = RobustScaler(), self.data[[col]]
                else:
                    scaler_obj, fit_data = StandardScaler(), self.data[[col]]

                scaler_obj.fit(fit_data)
                self.scalers_[col] = scaler_obj
        else:                
            # now Transform data based on learnt things
            for col, scaler_obj in self.scalers_.items():
                if col in self.data.columns:
                    data = self.data[[col]]
                    if self.transforms_.get(col) == "log1p":
                        data = np.log1p(data)
                    self.data[col] = scaler_obj.transform(data).flatten()

    def fit(self, data, exclude):
        self.data = data
        self.imputer(exclude=exclude, learn=True)
        self.outlier_manager(exclude=exclude, learn=True)
        self.scaler(exclude=exclude, learn=True)
        return None

    def transform(self, data, exclude):
        ## RAISE ERROR if transform run before fit
        if self.data is None:
            raise Exception("How can you transform data before fit.")

        self.data = data
        self.imputer(exclude=exclude, learn=False)
        self.outlier_manager(exclude=exclude, learn=False)
        self.scaler(exclude=exclude, learn=False)
        return self.data
