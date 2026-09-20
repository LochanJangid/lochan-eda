import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lochan_eda.utils import get_active_cols, is_matching


class Categorical:
    def __init__(self, profiler_df=None):
        self.profiler_df=profiler_df
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

    def imputer(self, learn=False):
        """Impute missing values based on missingness percentage."""
        active_cols = get_active_cols(self.fit_data.columns, exclude=None)

        if learn:
            missing_prcnt = self.fit_data[active_cols].isna().mean() * 100
            self.drop_cols_ = missing_prcnt[missing_prcnt > self.missing_drop_threshold].index.tolist()
            remaining_cols = [c for c in active_cols if c not in self.drop_cols_]

            for col in remaining_cols:
                if missing_prcnt[col] <= 10 and self.fit_data[col].dropna().shape[0] > 0:
                    self.impute_modes_[col] = self.fit_data[col].mode()[0]
        else:
            self.data = self.data.drop(columns=[c for c in self.drop_cols_ if c in self.data.columns], errors='ignore')
            active_cols = [col for col in active_cols if col not in self.drop_cols_]
            for col in active_cols:
                fill_val = 'Unknown' if col not in self.impute_modes_ else self.impute_modes_[col]
                self.data[col] = self.data[col].fillna(fill_val)


    def rare_manager(self, learn=False):
        """Group low-frequency categories into an 'Other' bin."""
        active_cols = get_active_cols(self.fit_data.columns, exclude=None)

        if learn:
            for col in active_cols:
                freqs = self.fit_data[col].value_counts(normalize=True)
                self.rare_cats_[col] = freqs[freqs < self.freq_threshold].index.tolist()
        else:
            for col, rare_list in self.rare_cats_.items():
                if col in active_cols and rare_list:
                    self.data[col] = self.data[col].apply(lambda x: 'Other' if x in rare_list else x)

    def encoder(self, learn=False):
        """Encode categories to numbers based on cardinality (number of unique values)."""
        active_cols = get_active_cols(self.fit_data.columns, exclude=None)
        encoded_dfs = []
        if learn:
            for col in active_cols:
                unique_cnt = self.fit_data[col].nunique()
                if unique_cnt <= 2:
                    self.encode_types_[col] = 'binary'
                    unique_vals = sorted(self.fit_data[col].dropna().unique())
                    self.binary_maps_[col] = {val: i for i, val in enumerate(unique_vals)}
                elif unique_cnt <= 10:
                    self.encode_types_[col] = 'ohe'
                    temp_ohe = pd.get_dummies(self.fit_data[col], prefix=col, drop_first=True)
                    self.ohe_columns_[col] = temp_ohe.columns.tolist()
                else:
                    self.encode_types_[col] = 'freq'
                    self.freq_maps_[col] = self.fit_data[col].value_counts(normalize=True).to_dict()
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

    def fit(self, data, exclude=None):
        self.fit_data = data
        if exclude is not None and isinstance(exclude, list):
            self.fit_data = self.fit_data.drop(columns=exclude, errors="ignore")
        self.imputer(learn=True)
        self.rare_manager(learn=True)
        self.encoder(learn=True)
        self.is_fitted_ = True
        return None
    
    def transform(self, data, exclude=None):
        if not self.is_fitted_:
            raise Exception("How can you transform data before fit.")
        if exclude is not None and isinstance(exclude, list):
            data = data.drop(columns=exclude, errors="ignore")
        if not is_matching(self.fit_data, data):
            raise Exception("Unmatched columns.")

        self.data = data

        self.imputer(learn=False)
        self.rare_manager(learn=False)
        self.encoder(learn=False)

        return self.data

    def summary(self):
        my_data = self.profiler_df.select_dtypes(include=["object", "category", "string"])
        if my_data.empty:
            return pd.DataFrame()

        summary = my_data.describe().T
        return summary.reindex()

    def plot(self, columns=None, top_n=10):
        my_data = self.profiler_df.select_dtypes(include=["object", "category", "string"])
        if my_data.empty:
                    return None
        
        if columns is not None:
            if not is_matching(my_data, pd.DataFrame(index=columns)):
                raise Exception("given columns are not categorical cols in profiler dataset")
            my_data = my_data[columns]

        fig, axes = plt.subplots(
            nrows=len(my_data.columns),
            ncols=1,
            figsize=(10, 4 * len(my_data.columns)),
            squeeze=False,
            constrained_layout=True
        )

        axes = axes.ravel()

        for ax, col in zip(axes, my_data.columns):
            counts = (my_data[col].value_counts(dropna=False))

            if top_n is not None:
                if top_n <= 0:
                    raise ValueError("top_n should be > 0")

                top = counts.head(top_n)
                others = counts[top_n:].sum()
                if others > 0:
                    top.loc["Others"] = others
                counts = top

            counts.index = counts.index.astype(str)

            counts.plot(kind="bar", ax=ax)

            ax.set_title(f"{col} - Category Distribution", fontsize=12, pad=12)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            
            ax.tick_params(axis="x", rotation=45)
        
        fig.suptitle("Categorical Analysis", fontsize=16, fontweight="bold")

        fig.savefig("categorical_plots.png", dpi=150, bbox_inches="tight")
        print("plots is saved into `categorical_plots.png` file")
        return fig