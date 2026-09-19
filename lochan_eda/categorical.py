import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from lochan_eda.utils import get_active_cols


class Categorical:
    def __init__(self):
        self.data = None
        self.is_fitted_ = False
        self.missing_drop_threshold = 40
        self.freq_threshold = 0.05
        self.drop_cols_ = []
        self.impute_modes_ = {}
        self.rare_cats_ = {}
        self.encode_types_ = {}
        self.binary_maps_ = {}
        self.ohe_columns_ = {}
        self.freq_maps_ = {}
        self.target_encoders_ = {}
        self.te_train_encoded_ = {}

    def imputer(self, exclude=None, learn=False):
        """Impute missing values based on missingness percentage."""
        active_cols = get_active_cols(self.data.columns, exclude)

        if learn:
            missing_prcnt = self.data[active_cols].isna().mean() * 100
            self.drop_cols_ = missing_prcnt[missing_prcnt > self.missing_drop_threshold].index.tolist()
            remaining_cols = [c for c in active_cols if c not in self.drop_cols_]

            for col in remaining_cols:
                if missing_prcnt[col] <= 10 and self.data[col].dropna().shape[0] > 0:
                    self.impute_modes_[col] = self.data[col].mode()[0]
        else:
            self.data = self.data.drop(columns=[c for c in self.drop_cols_ if c in self.data.columns], errors='ignore')
            active_cols = [col for col in active_cols if col not in self.drop_cols_]
            for col in active_cols:
                fill_val = 'Unknown' if col not in self.impute_modes_ else self.impute_modes_[col]
                self.data[col] = self.data[col].fillna(fill_val)


    def rare_manager(self, exclude=None, learn=False):
        """Group low-frequency categories into an 'Other' bin."""
        active_cols = get_active_cols(self.data.columns, exclude)

        if learn:
            for col in active_cols:
                freqs = self.data[col].value_counts(normalize=True)
                self.rare_cats_[col] = freqs[freqs < self.freq_threshold].index.tolist()
        else:
            for col, rare_list in self.rare_cats_.items():
                if col in active_cols and rare_list:
                    self.data[col] = self.data[col].apply(lambda x: 'Other' if x in rare_list else x)

    def encoder(self, exclude=None, learn=False, target=None):
        """Encode categories to numbers based on cardinality (number of unique values)."""
        active_cols = get_active_cols(self.data.columns, exclude)
        encoded_dfs = []
        if learn:
            for col in active_cols:
                unique_cnt = self.data[col].nunique()
                if unique_cnt <= 2:
                    self.encode_types_[col] = 'binary'
                    unique_vals = sorted(self.data[col].dropna().unique())
                    self.binary_maps_[col] = {val: i for i, val in enumerate(unique_vals)}
                elif unique_cnt <= 10:
                    self.encode_types_[col] = 'ohe'
                    temp_ohe = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                    self.ohe_columns_[col] = temp_ohe.columns.tolist()
                else:
                    self.encode_types_[col] = 'freq'
                    self.freq_maps_[col] = self.data[col].value_counts(normalize=True).to_dict()
        else:
            for col in active_cols:
                etype = self.encode_types_.get(col)
                if etype == 'binary':
                    encoded_series = self.data[col].map(self.binary_maps_[col]).fillna(-1).astype(int)
                    encoded_dfs.append(encoded_series.rename(col))
                elif etype == 'ohe':
                    ohe = pd.get_dummies(self.data[col], prefix=col)
                    ohe = ohe.reindex(columns=self.ohe_columns_.get(col, []), fill_value=0)
                    encoded_dfs.append(ohe)
                elif etype == 'freq':
                    encoded_series = self.data[col].map(self.freq_maps_[col]).fillna(0).rename(f"{col}_Freq")
                    encoded_dfs.append(encoded_series)
            if encoded_dfs:
                self.data = pd.concat(encoded_dfs, axis=1)

    def fit(self, data, exclude=None, target=None):
        self.data = data
        self.imputer(exclude=exclude, learn=True)
        self.rare_manager(exclude=exclude, learn=True)
        self.encoder(exclude=exclude, learn=True)
        self.is_fitted_ = True
        return self.data

    def transform(self, data, exclude=None):
        if not self.is_fitted_:
            raise ValueError("How can you transform self.data before fit.")

        self.data = data
        self.imputer(exclude=exclude, learn=False)
        self.rare_manager(exclude=exclude, learn=False)
        self.encoder(exclude=exclude, learn=False)
        return self.data
