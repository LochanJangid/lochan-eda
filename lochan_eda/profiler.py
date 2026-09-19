import pandas as pd
from lochan_eda.numerical import Numerical
from lochan_eda.categorical import Categorical
from lochan_eda.missing import Missing
from lochan_eda.report import Report

class Profiler():

    def __init__(self, df, target=None):
        """
            Profiler(
                df: pd.DataFrame -> data that you want to continue with
                target: str/pd.Series -> label
        """
        self.data = df
        if isinstance(target, str):
            self.target = df[target]
            self.data = df.drop(columns=[target])
        self.numerical = Numerical(profiler_df=self.data)
        self.categorical = Categorical(profiler_df=self.data)
        self.missing = Missing(profiler_df=self.data)
        self.report = Report(profiler=self)
        
    def overview(self):
        """
            Display a high-level overview of the dataset.

            The overview includes the dataset dimensions, memory usage,
            duplicate rows, missing values, and the number of numerical,
            categorical, and datetime columns.

            Returns
            -------
            dict
                A dictionary containing the computed dataset-level statistics:

                - ``rows`` : int
                    Total number of rows.
                - ``columns`` : int
                    Total number of columns.
                - ``memory_usage`` : int
                    Total memory usage of the DataFrame in bytes.
                - ``duplicate_rows`` : int
                    Number of duplicated rows.
                - ``duplicate_percentage`` : float
                    Percentage of rows that are duplicated.
                - ``missing_cells`` : int
                    Total number of missing cells.
                - ``missing_percentage`` : float
                    Percentage of missing cells across the entire dataset.
                - ``numerical_columns`` : int
                    Number of numerical columns.
                - ``categorical_columns`` : int
                    Number of categorical columns.
                - ``datetime_columns`` : int
                    Number of datetime columns.

            Notes
            -----
            This method performs descriptive inspection only. It does not
            modify the input DataFrame or learn any transformations.
        """
        rows = self.data.shape[0]
        cols = self.data.shape[1]

        memory_bytes = self.data.memory_usage(deep=True).sum()

        if memory_bytes < 1024:
            memory = f"{memory_bytes:.0f} B"
        elif memory_bytes < 1024 ** 2:
            memory = f"{memory_bytes / 1024:.2f} KB"
        elif memory_bytes < 1024 ** 3:
            memory = f"{memory_bytes / 1024 ** 2:.2f} MB"
        else:
            memory = f"{memory_bytes / 1024 ** 3:.2f} GB"

        duplicate_rows = self.data.duplicated().sum()
        duplicate_pct = (
            duplicate_rows / rows * 100
            if rows > 0
            else 0
        )

        missing_cells = self.data.isna().sum().sum()
        total_cells = rows * cols

        missing_pct = (
            missing_cells / total_cells * 100
            if total_cells > 0
            else 0
        )

        numerical_cols = self.data.select_dtypes(
            include="number"
        ).shape[1]

        categorical_cols = self.data.select_dtypes(
            include=["object", "category", "bool"]
        ).shape[1]

        datetime_cols = self.data.select_dtypes(
            include=["datetime"]
        ).shape[1]

        print()
        print("╔══════════════════════════════════════════╗")
        print("║              DATASET OVERVIEW            ║")
        print("╠══════════════════════════════════════════╣")
        print(f"║ Rows              : {rows:>20,} ║")
        print(f"║ Columns           : {cols:>20,} ║")
        print(f"║ Memory Usage      : {memory:>20} ║")
        print(f"║ Duplicate Rows    : {duplicate_rows:>20,} ║")
        print(f"║ Duplicate %       : {duplicate_pct:>19.2f}% ║")
        print(f"║ Missing Cells     : {missing_cells:>20,} ║")
        print(f"║ Missing %         : {missing_pct:>19.2f}% ║")
        print("╠══════════════════════════════════════════╣")
        print("║ COLUMN TYPES                             ║")
        print(f"║ Numerical         : {numerical_cols:>20,} ║")
        print(f"║ Categorical       : {categorical_cols:>20,} ║")
        print(f"║ Datetime          : {datetime_cols:>20,} ║")
        print("╚══════════════════════════════════════════╝")
        print()

        return {
            "rows": rows,
            "columns": cols,
            "memory_usage": memory_bytes,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_pct,
            "missing_cells": missing_cells,
            "missing_percentage": missing_pct,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
        }