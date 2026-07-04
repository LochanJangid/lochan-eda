import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder

class HandleCategorical:
    def __init__(self, df):
        self.cat_df = df.select_dtypes(include=["category", "object", "string"]).copy()

    def cat_imputer(self):
        """Impute missing values based on missingness percentage."""
        missing_prcnt = self.cat_df.isna().mean() * 100
        
        drop_cols = missing_prcnt[missing_prcnt > 40].index
        self.cat_df.drop(columns=drop_cols, inplace=True)
        
        remaining_cols = missing_prcnt[(missing_prcnt > 0) & (missing_prcnt <= 40)].index
        for col in remaining_cols:
            if missing_prcnt[col] > 10:
                fill_val = 'Unknown'
            else:
                fill_val = self.cat_df[col].mode()[0]
            self.cat_df[col] = self.cat_df[col].fillna(fill_val)
        
        return self.cat_df

    def rare_manager(self, threshold=0.05):
        """Group low-frequency categories into an 'Other' bin."""
        for col in self.cat_df.columns:
            freqs = self.cat_df[col].value_counts(normalize=True)
            rare_cats = freqs[freqs < threshold].index
            
            if not rare_cats.empty:
                if isinstance(self.cat_df[col].dtype, pd.CategoricalDtype):
                    if 'Other' not in self.cat_df[col].cat.categories:
                        self.cat_df[col] = self.cat_df[col].cat.add_categories('Other')
                self.cat_df[col] = self.cat_df[col].replace(rare_cats, 'Other')
                
        return self.cat_df

    def encoder(self, target=None):
        """Encode categories to numbers based on cardinality (number of unique values)."""
        encoded_dfs = []
        
        for col in self.cat_df.columns:
            unique_cnt = self.cat_df[col].nunique()
            
            if unique_cnt <= 2:
                mapping = {val: i for i, val in enumerate(self.cat_df[col].dropna().unique())}
                encoded_series = self.cat_df[col].map(mapping)
                encoded_dfs.append(encoded_series)
            elif unique_cnt <= 10:
                ohe = pd.get_dummies(self.cat_df[col], prefix=col, drop_first=True, dtype=int)
                encoded_dfs.append(ohe)
            else:
                if target is not None:
                    te = TargetEncoder(smooth="auto")
                    encoded = te.fit_transform(self.cat_df[[col]], target)
                    encoded_dfs.append(pd.DataFrame(encoded, columns=[f"{col}_TE"], index=self.cat_df.index))
                else:
                    freq = self.cat_df[col].value_counts(normalize=True)
                    encoded_series = self.cat_df[col].map(freq).rename(f"{col}_Freq")
                    encoded_dfs.append(encoded_series)
                    
        self.cat_df = pd.concat(encoded_dfs, axis=1)
        return self.cat_df
    
    
    def full_handler(self, target):
        """Execute Imputer, rare values Manager, Encoder (all in one)."""
        self.cat_imputer()
        self.rare_manager()
        self.encoder(target=target)
        return self.cat_df