import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder

class HandleCategorical:
    def __init__(self, df):
        self.cat_df = df.select_dtypes(include=["category", "object", "string"]).copy()

        self.drop_cols_ = []
        self.impute_modes_ = {}
        self.rare_cats_ = {}
        self.encode_types_ = {}    
        self.binary_maps_ = {}
        self.ohe_columns_ = {}    
        self.freq_maps_ = {}
        self.target_encoders_ = {}

    def cat_imputer(self, is_train=True):
        """Impute missing values based on missingness percentage."""
        
        self.cat_df = self.cat_df.replace('nan', np.nan)

        if is_train:
            missing_prcnt = self.cat_df.isna().mean() * 100
            self.drop_cols_ = missing_prcnt[missing_prcnt > 40].index.tolist()
            remaining_cols = [c for c in self.cat_df.columns if c not in self.drop_cols_]
    
            for col in remaining_cols:
                if missing_prcnt[col] <= 10 and self.cat_df[col].dropna().shape[0] > 0:
                    self.impute_modes_[col] = self.cat_df[col].mode()[0]

        # Work for both train and test df

        self.cat_df.drop(columns=[c for c in self.drop_cols_ if c in self.cat_df.columns], inplace=True)
        for col in self.cat_df.columns:
            fill_val = 'Unknown' if col not in self.impute_modes_ else self.impute_modes_[col]
            self.cat_df[col] = self.cat_df[col].fillna(fill_val)
    
        return self.cat_df

    def rare_manager(self, is_train=True, threshold=0.05):
        """Group low-frequency categories into an 'Other' bin."""

        if is_train:
            for col in self.cat_df.columns:
                freqs = self.cat_df[col].value_counts(normalize=True)
                self.rare_cats_[col] = freqs[freqs < threshold].index.tolist()

        # Work
        for col, rare_list in self.rare_cats_.items():
            if col in self.cat_df.columns and rare_list:
                self.cat_df[col] = self.cat_df[col].apply(lambda x: 'Other' if x in rare_list else x)

        return self.cat_df

    def encoder(self, is_train=True, target=None):
        """Encode categories to numbers based on cardinality (number of unique values)."""
        encoded_dfs = []
        
        for col in self.cat_df.columns: 
            if is_train:
                unique_cnt = self.cat_df[col].nunique()
            
                if unique_cnt <= 2:
                    self.encode_types_[col] = 'binary'
                    unique_vals = sorted(self.cat_df[col].dropna().unique())
                    self.binary_maps_[col] = {val: i for i, val in enumerate(unique_vals)}
                elif unique_cnt <= 10:
                    self.encode_types_[col] = 'ohe'
                    temp_ohe = pd.get_dummies(self.cat_df[col], prefix=col, drop_first=True)
                    self.ohe_columns_[col] = temp_ohe.columns.tolist()
                else:
                    if target is not None:
                        self.encode_types_[col] = 'te'
                        te = TargetEncoder(smooth="auto")
                        te.fit(self.cat_df[[col]], target)
                        self.target_encoders_[col] = te
                    else:
                        self.encode_types_[col] = 'freq'
                        self.freq_maps_[col] = self.cat_df[col].value_counts(normalize=True).to_dict()
                    
            # work (train and test)

            etype = self.encode_types_.get(col)

            if etype == 'binary':
                encoded_series = self.cat_df[col].map(self.binary_maps_[col]).fillna(-1).astype(int)
                encoded_dfs.append(encoded_series)
            elif etype == 'ohe':
                ohe = pd.get_dummies(self.cat_df[col], prefix=col, drop_first=True)
                ohe = ohe.reindex(columns=self.ohe_columns_.get(col, []), fill_value=0)
                encoded_dfs.append(ohe)
            elif etype == 'te':
                encoded = self.target_encoders_[col].transform(self.cat_df[[col]])
                encoded_dfs.append(pd.DataFrame(encoded, columns=[f"{col}_TE"], index=self.cat_df.index))

            elif etype == 'freq':
                encoded_series = self.cat_df[col].map(self.freq_maps_[col]).fillna(0).rename(f"{col}_Freq")
                encoded_dfs.append(encoded_series)

        self.cat_df = pd.concat(encoded_dfs, axis=1)
        return self.cat_df
    
    
    def full_handler(self, target=None, is_train=True):
        """Execute Imputer, rare values Manager, Encoder (all in one)."""
        self.cat_imputer(is_train=is_train)
        self.rare_manager(is_train=is_train)
        self.encoder(target=target, is_train=is_train)
        return self.cat_df