class HandleNumerical:
    def __init__(self, num_df):
        self.num_df = num_df.copy()

    def imputer(self):
        """impute missing values based on their data behaviour."""

        missing_prcnt = self.num_df.isna().mean() * 100
        drop_to_cols = missing_prcnt[missing_prcnt > 40].index
        self.num_df.drop(columns=drop_to_cols, inplace=True)

        uniquness = (self.num_df.nunique() / self.num_df.shape[0]) * 100
        # if uniquness prcnt < 1 that means column fill with only some values (like: categorical)
        like_cat = uniquness[uniquness < 1].index 
        # mode imputation for categorical-like numerical columns
        most_frequent_vals = self.num_df[like_cat].mode()
        self.num_df[like_cat] = self.num_df[like_cat].fillna(most_frequent_vals)


        simple_cols = missing_prcnt[(missing_prcnt > 0) & (missing_prcnt <= 40)].index
        if not simple_cols.empty:
          skewness = self.num_df[simple_cols].skew().abs()
          means = self.num_df[simple_cols].mean()
          medians = self.num_df[simple_cols].median()
          fill_vals = means.where(skewness < 0.5, medians)
          self.num_df[simple_cols] = self.num_df[simple_cols].fillna(fill_vals)
        
        return self.num_df