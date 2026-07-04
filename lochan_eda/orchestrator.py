from lochan_eda.numerical import HandleNumerical
from lochan_eda.categorical import HandleCategorical

class AutomatedEDA():
    def __init__(self, df):
        self.df = df

    def run_pipeline(self, target=None):
        num_handler = HandleNumerical(self.df)
        cleaned_num_df = num_handler.full_handler()
        cat_handler = HandleCategorical(self.df)
        cleaned_cat_df = cat_handler.full_handler(target=target)

        return pd.concat([cleaned_num_df, cleaned_cat_df], axis=1, join="inner")