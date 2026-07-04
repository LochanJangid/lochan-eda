import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

from lochan_eda.utils import get_iqr_bounds, get_active_cols

class HandleNumerical:
    def __init__(self, df):
        self.num_df = df.select_dtypes(include=["number"]).copy()

    def num_imputer(self, exclude=None):
        """impute missing values based on their data behaviour."""
        active_cols = get_active_cols(self.num_df.columns, exclude)

        missing_prcnt = self.num_df[active_cols].isna().mean() * 100
        drop_to_cols = missing_prcnt[missing_prcnt > 40].index
        self.num_df.drop(columns=drop_to_cols, inplace=True)

        uniquness = (self.num_df.nunique() / self.num_df.shape[0]) * 100
        # if uniquness prcnt < 1 that means column fill with only some values (like: categorical)
        like_cat = uniquness[uniquness < 1].index 
        # mode imputation for categorical-like numerical columns
        if not like_cat.empty:
          most_frequent_vals = self.num_df[like_cat].mode().iloc[0]
          self.num_df[like_cat] = self.num_df[like_cat].fillna(most_frequent_vals)


        simple_cols = missing_prcnt[(missing_prcnt > 0) & (missing_prcnt <= 40)].index
        if not simple_cols.empty:
          skewness = self.num_df[simple_cols].skew().abs()
          means = self.num_df[simple_cols].mean()
          medians = self.num_df[simple_cols].median()
          fill_vals = means.where(skewness < 0.5, medians)
          self.num_df[simple_cols] = self.num_df[simple_cols].fillna(fill_vals)
        
        return self.num_df
   
    def outlier_manager(self, exclude=None):
      """handle outliers based on there percentage and the distribution of data."""
      active_cols = get_active_cols(self.num_df.columns, exclude)

      for col in active_cols:
        # finding outlier percentage by iqr method
        lower_bound, upper_bound = get_iqr_bounds(self.num_df[col])
        outliers_prcnt = ((self.num_df[col] > upper_bound) | (self.num_df[col] < lower_bound)).mean() * 100
        
        # when we have only some outliers they can be error ( < 3%). and trimming them will not damage our dataset
        if outliers_prcnt <= 3.0 and outliers_prcnt > 0:
          self.num_df = self.num_df[(self.num_df[col] <= upper_bound) & (self.num_df[col] >= lower_bound)]
          continue

        # now outliers are so much so we can't just trim them we need to handle them softly :)
        p90 = self.num_df[col].quantile(0.90)
        p99 = self.num_df[col].quantile(0.99)
        max_val = self.num_df[col].max()
        gap = (p99 - p90) / (max_val - p99 + 1e-9)

        if gap <= 1.0:
          # spikes -> heavy tail. method -> Winsorization
          p5 = self.num_df[col].quantile(0.05)
          p95 = self.num_df[col].quantile(0.95)
          self.num_df[col] = self.num_df[col].clip(lower=p5, upper=p95)
        else:
          # Softly -> Long tail. method -> Transformation
          is_negative_exist = self.num_df[col][self.num_df[col] < 0].count()
          if not is_negative_exist:
            is_zero_exist = self.num_df[col][self.num_df[col] == 0].count()
            if is_zero_exist:
              # Square root transformation
              self.num_df[col] = np.sqrt(self.num_df[col])
            else:
              # Log transformation
              self.num_df[col] = np.log1p(self.num_df[col])
      return self.num_df
    
    def scaler(self, exclude=None):
      """Scale data on the basis of there sparsity, skewness, outlier_ratio."""
      active_cols = get_active_cols(self.num_df.columns, exclude)

      for col in active_cols:
        # percentage of zero values
        sparsity = (self.num_df[col] == 0).mean()
        # measure the skewness
        skewness = self.num_df[col].skew()
         # finding outlier percentage by iqr method
        lower_bound, upper_bound = get_iqr_bounds(self.num_df[col])
        outliers_ratio = ((self.num_df[col] > upper_bound) | (self.num_df[col] < lower_bound)).mean()

        if sparsity >= 0.5:
          scaler_obj = MaxAbsScaler()
        elif skewness > 1.0 and self.num_df[col].min() >= 0:
          self.num_df[col] = np.log1p(self.num_df[col])
          scaler_obj = StandardScaler()
        elif outliers_ratio >= 0.05:
          scaler_obj = RobustScaler()
        else:
          scaler_obj = StandardScaler()
        
        self.num_df[col] = scaler_obj.fit_transform(self.num_df[[col]]).flatten()
        
      return self.num_df
    
    def full_handler(self):
      """Execute Imputer, Outlier Manager, Scaler (all in one)."""
      self.num_imputer()
      self.outlier_manager()
      self.scaler()
      return self.num_df