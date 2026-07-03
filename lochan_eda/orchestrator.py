import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lochan_eda.numerical import HandleNumerical

class AutomatedEDA:
    def __init__(self, df):
        self.df = df

    def run_pipeline(self):
        num_df = self.df.select_dtypes(include=["number"])

        num_handler = HandleNumerical(num_df)
        imputed_df = num_handler.imputer()

        return imputed_df