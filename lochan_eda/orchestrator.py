import pandas as pd

from lochan_eda.numerical import HandleNumerical
from lochan_eda.categorical import HandleCategorical

class AutomatedEDA():
    def __init__(self):

        # make hanlder that will stay on both train and test
        self.num_handler = None
        self.cat_handler = None

    def run_pipeline(self, df, target=None, is_train=True):

        if is_train:
            self.num_handler = HandleNumerical(df)
            self.cat_handler = HandleCategorical(df)
        else:
            # for test data just replace inner dataframes attributes
            self.num_handler.num_df = df.select_dtypes(include=["number"]).copy()
            self.cat_handler.cat_df = df.select_dtypes(include=["category", "object", "string"]).copy()

        # Execute pipeline
        cleaned_num_df = self.num_handler.full_handler(is_train=is_train)
        cleaned_cat_df = self.cat_handler.full_handler(target=target, is_train=is_train)

        return pd.concat([cleaned_num_df, cleaned_cat_df], axis=1, join="inner")