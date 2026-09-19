import numpy as np
import pandas as pd
import pytest

from lochan_eda.orchestrator import AutomatedEDA
from lochan_eda.numerical import Numerical
from lochan_eda.categorical import Categorical


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "age": [22, 30, 27, 40, 35, 50, 29, 45, 31, 38],
        "income": [20000, 22000, 26000, 50000, 48000, 70000, 28000, 60000, 35000, 45000],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def missing_df():
    return pd.DataFrame({
        "age": [22, 30, np.nan, 40, 35, 50, 29, 45, 31, 38],
        "income": [20000, 22000, 26000, np.nan, 48000, 70000, 28000, 60000, 35000, 45000],
        "city": ["A", "B", np.nan, "C", "B", "A", "C", "B", "A", "C"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def high_cardinality_df():
    return pd.DataFrame({
        "category": [f"cat_{i}" for i in range(20)],
        "value": list(range(20)),
    })


# ============================================================
# AutomatedEDA
# ============================================================

class TestAutomatedEDA:

    def test_prepare_with_target_and_split(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target"
        )

        assert isinstance(Xtr, pd.DataFrame)
        assert isinstance(Xte, pd.DataFrame)
        assert isinstance(ytr, pd.Series)
        assert isinstance(yte, pd.Series)

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

        assert "target" not in Xtr.columns
        assert "target" not in Xte.columns

    def test_prepare_without_target_and_split(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte = eda.prepare(basic_df.drop(columns="target"))

        assert isinstance(Xtr, pd.DataFrame)
        assert isinstance(Xte, pd.DataFrame)

        assert len(Xtr) + len(Xte) == len(basic_df)

    def test_prepare_with_target_without_split(self, basic_df):
        eda = AutomatedEDA()

        X, y = eda.prepare(
            basic_df,
            target="target",
            split=False
        )

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        assert len(X) == len(y)
        assert "target" not in X.columns

    def test_prepare_without_target_without_split(self, basic_df):
        eda = AutomatedEDA()

        result = eda.prepare(
            basic_df.drop(columns="target"),
            split=False
        )

        assert isinstance(result, pd.DataFrame)

        assert len(result) == len(basic_df)
        assert "age" in result.columns
        assert "income" in result.columns
        assert "city_B" in result.columns
        assert "city_C" in result.columns

    def test_target_series(self, basic_df):
        eda = AutomatedEDA()

        X = basic_df.drop(columns="target")
        y = basic_df["target"]

        Xtr, Xte, ytr, yte = eda.prepare(
            X,
            target=y
        )

        assert isinstance(ytr, pd.Series)
        assert isinstance(yte, pd.Series)

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_exclude_numeric_column(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["age"]
        )

        assert "age" not in Xtr.columns
        assert "age" not in Xte.columns

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_exclude_categorical_column(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["city"]
        )

        assert "city" not in Xtr.columns
        assert "city" not in Xte.columns

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_exclude_multiple_columns(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["age", "city"]
        )

        assert "age" not in Xtr.columns
        assert "city" not in Xtr.columns

        assert "age" not in Xte.columns
        assert "city" not in Xte.columns

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_exclude_does_not_change_row_count(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["age"]
        )

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

        assert len(Xtr) + len(Xte) == len(basic_df)

    def test_exclude_drops_column_completely(self, basic_df):
        eda = AutomatedEDA()

        X, y = eda.prepare(
            basic_df,
            target="target",
            exclude=["age", "city"],
            split=False
        )

        assert "age" not in X.columns
        assert "city" not in X.columns

        assert set(X.columns) == {"income"}

    def test_fit_creates_processors(self, basic_df):
        eda = AutomatedEDA()

        X = basic_df.drop(columns="target")

        eda.fit(X)

        assert hasattr(eda, "numerical")
        assert hasattr(eda, "categorical")

        assert eda.numerical.is_fitted_
        assert eda.categorical.is_fitted_

    def test_transform_after_fit(self, basic_df):
        eda = AutomatedEDA()

        X = basic_df.drop(columns="target")

        eda.fit(X)
        transformed = eda.transform(X)

        assert isinstance(transformed, tuple)
        assert isinstance(transformed[0], pd.DataFrame)

    def test_transform_before_fit_raises(self, basic_df):
        eda = AutomatedEDA()

        X = basic_df.drop(columns="target")

        with pytest.raises(Exception):
            eda.transform(X)


# ============================================================
# Numerical
# ============================================================

class TestNumerical:

    def test_fit(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]

        result = numerical.fit(X)

        assert result is None
        assert numerical.is_fitted_

    def test_transform_before_fit_raises(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]

        with pytest.raises(Exception):
            numerical.transform(X)

    def test_transform_returns_dataframe(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]

        numerical.fit(X)
        result = numerical.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)

    def test_numeric_columns_remain_numeric(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]

        numerical.fit(X)
        result = numerical.transform(X)

        assert pd.api.types.is_numeric_dtype(result["age"])
        assert pd.api.types.is_numeric_dtype(result["income"])

    def test_missing_values_are_processed(self, missing_df):
        numerical = Numerical()

        X = missing_df[["age", "income"]]

        numerical.fit(X)
        result = numerical.transform(X)

        assert not result.isna().any().any()

    def test_exclude_numeric_column(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]

        numerical.fit(X, exclude=["age"])
        result = numerical.transform(X, exclude=["age"])

        assert "age" not in result.columns
        assert "income" in result.columns

    def test_exclude_missing_column_does_not_crash(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]

        numerical.fit(X, exclude=["city"])
        result = numerical.transform(X, exclude=["city"])

        assert isinstance(result, pd.DataFrame)

    def test_unmatched_columns_raise(self, basic_df):
        numerical = Numerical()

        X = basic_df[["age", "income"]]
        numerical.fit(X)

        different_X = basic_df[["age"]]

        with pytest.raises(Exception):
            numerical.transform(different_X)


# ============================================================
# Categorical
# ============================================================

class TestCategorical:

    def test_fit(self, basic_df):
        categorical = Categorical()

        X = basic_df[["city"]]

        result = categorical.fit(X)

        assert categorical.is_fitted_
        assert result is None

    def test_transform_before_fit_raises(self, basic_df):
        categorical = Categorical()

        X = basic_df[["city"]]

        with pytest.raises(Exception):
            categorical.transform(X)

    def test_binary_encoding(self):
        categorical = Categorical()

        X = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M"]
        })

        categorical.fit(X)
        result = categorical.transform(X)

        assert "gender" in result.columns
        assert pd.api.types.is_numeric_dtype(result["gender"])

    def test_one_hot_encoding(self):
        categorical = Categorical()

        X = pd.DataFrame({
            "city": ["A", "B", "C", "A", "B"]
        })

        categorical.fit(X)
        result = categorical.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)

        assert all(
            column.startswith("city_")
            for column in result.columns
        )

    def test_frequency_encoding(self, high_cardinality_df):
        categorical = Categorical()

        X = high_cardinality_df[["category"]]

        categorical.fit(X)
        result = categorical.transform(X)

        assert "category_Freq" in result.columns
        assert pd.api.types.is_numeric_dtype(
            result["category_Freq"]
        )

    def test_missing_values_are_processed(self, missing_df):
        categorical = Categorical()

        X = missing_df[["city"]]

        categorical.fit(X)
        result = categorical.transform(X)

        assert not result.isna().any().any()

    def test_exclude_categorical_column(self, basic_df):
        categorical = Categorical()

        X = basic_df[["city"]]

        categorical.fit(X, exclude=["city"])
        result = categorical.transform(X, exclude=["city"])

        assert "city" not in result.columns

    def test_exclude_missing_column_does_not_crash(self, basic_df):
        categorical = Categorical()

        X = basic_df[["city"]]

        categorical.fit(X, exclude=["age"])
        result = categorical.transform(X, exclude=["age"])

        assert isinstance(result, pd.DataFrame)

    def test_unmatched_columns_raise(self, basic_df):
        categorical = Categorical()

        X = basic_df[["city"]]
        categorical.fit(X)

        different_X = pd.DataFrame({
            "other_city": ["A", "B", "C"]
        })

        with pytest.raises(Exception):
            categorical.transform(different_X)


# ============================================================
# Integration
# ============================================================

class TestIntegration:

    def test_full_pipeline(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target"
        )

        assert isinstance(Xtr, pd.DataFrame)
        assert isinstance(Xte, pd.DataFrame)

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

        assert not Xtr.isna().any().any()
        assert not Xte.isna().any().any()

    def test_pipeline_with_numeric_exclude(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["age"]
        )

        assert "age" not in Xtr.columns
        assert "age" not in Xte.columns

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_pipeline_with_categorical_exclude(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["city"]
        )

        assert "city" not in Xtr.columns
        assert "city" not in Xte.columns

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_pipeline_with_multiple_excludes(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["age", "city"]
        )

        assert set(Xtr.columns) == {"income"}
        assert set(Xte.columns) == {"income"}

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

    def test_pipeline_without_target(self, basic_df):
        eda = AutomatedEDA()

        X = basic_df.drop(columns="target")

        Xtr, Xte = eda.prepare(X)

        assert isinstance(Xtr, pd.DataFrame)
        assert isinstance(Xte, pd.DataFrame)

        assert len(Xtr) + len(Xte) == len(X)

    def test_pipeline_preserves_train_test_row_counts(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target"
        )

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

        assert len(Xtr) + len(Xte) == len(basic_df)

    def test_pipeline_with_missing_values(self, missing_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            missing_df,
            target="target"
        )

        assert not Xtr.isna().any().any()
        assert not Xte.isna().any().any()

    def test_excluded_numeric_and_categorical_columns(self, basic_df):
        eda = AutomatedEDA()

        Xtr, Xte, ytr, yte = eda.prepare(
            basic_df,
            target="target",
            exclude=["age", "city"]
        )

        assert "age" not in Xtr.columns
        assert "city" not in Xtr.columns

        assert "age" not in Xte.columns
        assert "city" not in Xte.columns

        assert len(Xtr) == len(ytr)
        assert len(Xte) == len(yte)

        assert not Xtr.isna().any().any()
        assert not Xte.isna().any().any()