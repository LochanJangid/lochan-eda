import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

from lochan_eda.utils import get_iqr_bounds, get_active_cols, is_matching


class Numerical:
    def __init__(self, profiler_df=None):
        self.profiler_df = profiler_df
        self.is_fitted_ = False
        self.missing_drop_threshold = 40
        self.outlier_threshold = 3.0
        self.drop_cols_ = []
        self.impute_values_ = {}
        self.outlier_rules_ = {}
        self.scalers_ = {}
        self.transforms_ = {}

    def imputer(self, learn=False):
        """impute missing values based on their data behaviour."""
        active_cols = get_active_cols(self.fit_data.columns, exclude=None)

        # Runs on training and memorize drop_cols & impute values
        if learn: 
            missing_prcnt = self.fit_data[active_cols].isna().mean() * 100
            # drop columns with > threshold missing data
            self.drop_cols_ = missing_prcnt[missing_prcnt > self.missing_drop_threshold].index.tolist()

            active_cols = [col for col in active_cols if col not in self.drop_cols_]
            self.fit_data.drop(columns=self.drop_cols_, inplace=True, errors="ignore")

            if active_cols:
                uniquness = (self.fit_data[active_cols].nunique() / self.fit_data[active_cols].shape[0]) * 100
                like_cat = uniquness[uniquness < 1].index

                for col in like_cat:
                    if self.fit_data[col].isna().sum() > 0:
                        self.impute_values_[col] = self.fit_data[col].mode().iloc[0]

                simple_cols = [col for col in active_cols if col not in like_cat]

                for col in simple_cols:
                    if missing_prcnt[col] > 0:
                        skewness = self.fit_data[col].skew()
                        fill_val = self.fit_data[col].mean() if abs(skewness) < 0.5 else self.fit_data[col].median()
                        self.impute_values_[col] = fill_val
        
        else:
            # now Transform data based on learnt things
            if self.drop_cols_:
                cols_to_drop = [c for c in self.drop_cols_ if c in self.data.columns]
                self.data.drop(columns=cols_to_drop, inplace=True)

            for col, fill_val in self.impute_values_.items():
                if col in self.data.columns:
                    self.data[col] = self.data[col].fillna(fill_val)

    def outlier_manager(self, learn=False):
        active_cols = get_active_cols(self.fit_data.columns, exclude=None)

        if learn:
            for col in active_cols:
                lower_bound, upper_bound = get_iqr_bounds(self.fit_data[col])
                outliers_prcnt = ((self.fit_data[col] > upper_bound) | (self.fit_data[col] < lower_bound)).mean() * 100

                if 0 < outliers_prcnt <= self.outlier_threshold:
                    self.outlier_rules_[col] = {'type': 'clip', 'lower': lower_bound, 'upper': upper_bound}
                elif outliers_prcnt > self.outlier_threshold:
                    p90 = self.fit_data[col].quantile(0.90)
                    p99 = self.fit_data[col].quantile(0.99)
                    max_val = self.fit_data[col].max()
                    gap = (p99 - p90) / (max_val - p99 + 1e-9)

                    is_negative = (self.fit_data[col] < 0).any()
                    is_non_negative = (self.fit_data[col] >= 0).all()

                    if gap <= 1.0 or is_negative:
                        p5 = self.fit_data[col].quantile(0.05)
                        p95 = self.fit_data[col].quantile(0.95)
                        self.outlier_rules_[col] = {'type': 'clip', 'lower': p5, 'upper': p95}
                    else:
                        is_zero = (self.fit_data[col] == 0).any()
                        self.outlier_rules_[col] = {'type': 'sqrt' if is_zero and is_non_negative else 'log1p'}
        else:
            # now Transform data based on learnt things
            for col, rule in self.outlier_rules_.items():
                if col in self.data.columns:
                    if rule['type'] == 'clip':
                        self.data[col] = self.data[col].clip(lower=rule['lower'], upper=rule['upper'])
                    elif rule['type'] == 'sqrt':
                        self.data[col] = np.sqrt(self.data[col])
                    elif rule['type'] == 'log1p':
                        self.data[col] = np.log1p(self.data[col])

    def scaler(self, learn=False):
        active_cols = get_active_cols(self.fit_data.columns, exclude=None)

        if learn:
            for col in active_cols:
                sparsity = (self.fit_data[col] == 0).mean()
                skewness = self.fit_data[col].skew()
                lower_bound, upper_bound = get_iqr_bounds(self.fit_data[col])
                outliers_ratio = ((self.fit_data[col] > upper_bound) | (self.fit_data[col] < lower_bound)).mean()

                if sparsity >= 0.5:
                    scaler_obj, fit_data = MaxAbsScaler(), self.fit_data[[col]]
                elif skewness > 1.0 and self.fit_data[col].min() >= 0:
                    self.transforms_[col] = "log1p"
                    scaler_obj, fit_data = StandardScaler(), np.log1p(self.fit_data[[col]])
                elif outliers_ratio >= 0.05:
                    scaler_obj, fit_data = RobustScaler(), self.fit_data[[col]]
                else:
                    scaler_obj, fit_data = StandardScaler(), self.fit_data[[col]]

                scaler_obj.fit(fit_data)
                self.scalers_[col] = scaler_obj
        else:                
            # now Transform data based on learnt things
            for col, scaler_obj in self.scalers_.items():
                if col in self.data.columns:
                    data = self.data[[col]]
                    if self.transforms_.get(col) == "log1p":
                        data = np.log1p(data)
                    self.data[col] = scaler_obj.transform(data).flatten()

    def fit(self, data, exclude=None):
        self.fit_data = data
        if exclude is not None and isinstance(exclude, list):
            self.fit_data = self.fit_data.drop(columns=exclude, errors="ignore")
        self.imputer(learn=True)
        self.outlier_manager(learn=True)
        self.scaler(learn=True)
        self.is_fitted_ = True

        return None

    def transform(self, data, exclude=None):
        ## RAISE ERROR if transform run before fit
        if not self.is_fitted_:
            raise Exception("How can you transform data before fit.")
        if exclude is not None and isinstance(exclude, list):
            data = data.drop(columns=exclude, errors="ignore")
        if not is_matching(self.fit_data, data):
            raise Exception("Unmatched columns.")
        
        self.data = data
        self.imputer(learn=False)
        self.outlier_manager(learn=False)
        self.scaler(learn=False)
        return self.data

    def summary(self):
        my_data = self.profiler_df.select_dtypes(include=["number"])

        if my_data.empty:
            return pd.DataFrame()

        summary = my_data.describe().T

        summary["std"] = my_data.std()
        summary["missings %"] = my_data.isna().mean()
        summary["skew"] = my_data.skew()
        summary["zeros %"] = (my_data == 0).mean()
        iqr = summary["75%"] - summary["25%"]
        lower_bound = summary["25%"] - iqr*0.5
        upper_bound = summary["25%"] + iqr*0.5
        summary["outliers %"] = ((my_data < lower_bound) | (my_data > upper_bound)).mean()

        print("\nNumerical Summary\n")
        print(summary)
        return summary

    def plot(self, columns=None):
        my_data = self.profiler_df.select_dtypes(include=["number"])

        if columns is not None:
            if not is_matching(my_data, pd.DataFrame(index=columns)):
                raise Exception("given columns are not numerical cols in profiler dataset")
            my_data = my_data[columns]

        fig, axes = plt.subplots(
            nrows=len(my_data.columns),
            ncols=3,
            figsize=(16, 4*len(my_data.columns)),
            squeeze=False
        )

        for row, col in enumerate(my_data.columns):
            data = my_data[col].dropna() # Save our plots from null values

            # Distribution
            axes[row, 0].hist(data, bins="auto", density=True)            
            axes[row, 0].set_title(f"{col} - Distribution")

            # Boxplot
            axes[row, 1].boxplot(data, vert=False)
            axes[row, 1].set_title(f"{col} - Box Plot")

            # Q-Q
            stats.probplot(data, dist="norm", plot=axes[row, 2])
            axes[row, 2].set_title(f"{col} - Q - Q Plot")

        fig.suptitle("Numerical Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout()

        fig.savefig("numerical_plots.png", dpi=150, bbox_inches="tight")
        print("plots is saved into `numerical_plots.png` file")
        return fig