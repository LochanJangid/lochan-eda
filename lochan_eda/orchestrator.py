import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lochan_eda.numerical import HandleNumerical
from lochan_eda.categorical import HandleCategorical

class AutomatedEDA(HandleNumerical, HandleCategorical):
    def __init__(self, df):
        self.df = df

    def run_pipeline(self):
        num_handler = HandleNumerical(self.df)
        cleaned_num_df = num_handler.full_handler()
        cat_handler = HandleCategorical(self.df)
        cleaned_cat_df = cat_handler.full_handler()

        return pd.concat([cleaned_num_df, cleaned_cat_df], axis=1, join="inner")