import numpy as np
import pandas as pd
import pytest

from lochan_eda import AutomatedEDA
from lochan_eda.numerical import HandleNumerical
from lochan_eda.categorical import HandleCategorical


@pytest.fixture
def messy_df():
    rng = np.random.default_rng(42)
    n = 4000

    df = pd.DataFrame({
        "income": rng.exponential(scale=45000, size=n),
        "temp_change": rng.normal(0, 4, n) + rng.choice([0, 40, -40], size=n, p=[0.94, 0.03, 0.03]),
        "num_purchases": rng.poisson(0.3, n),
        "age": rng.normal(35, 10, n).clip(18, 90),
        "score_like_cat": rng.choice([1, 2, 3], size=n).astype(float),
        "gender": rng.choice(["M", "F"], size=n),
        "city": rng.choice(["Jaipur", "Delhi", "Mumbai", "Bangalore", "Chennai"], size=n),
        "occupation": rng.choice([f"occ_{i}" for i in range(15)], size=n),
    })

    for col in ["income", "age", "score_like_cat", "city"]:
        mask = rng.random(n) < 0.08
        df.loc[mask, col] = np.nan

    return df


@pytest.fixture
def noise_target(messy_df):
    rng = np.random.default_rng(7)
    return pd.Series(rng.integers(0, 2, size=len(messy_df)), name="target")


@pytest.fixture
def informative_target(messy_df):
    rng = np.random.default_rng(11)
    occ_effect = {occ: rng.uniform(-2, 2) for occ in messy_df["occupation"].unique()}
    logits = messy_df["occupation"].map(occ_effect).fillna(0) + rng.normal(0, 0.5, len(messy_df))
    prob = 1 / (1 + np.exp(-logits))
    return pd.Series((rng.random(len(messy_df)) < prob).astype(int), name="target")

class TestNoCrashes:

    def test_full_pipeline_runs_with_target(self, messy_df, noise_target):
        out = AutomatedEDA().run_pipeline(messy_df, target=noise_target, is_train=True)
        assert len(out) == len(messy_df)
        assert out.shape[1] > 0

    def test_full_pipeline_runs_without_target(self, messy_df):
        out = AutomatedEDA().run_pipeline(messy_df, target=None, is_train=True)
        assert len(out) == len(messy_df)

    def test_scaler_does_not_crash_on_skewed_column(self, messy_df):
        handler = HandleNumerical(messy_df)
        handler.num_imputer(is_train=True)
        handler.outlier_manager(is_train=True)
        result = handler.scaler(is_train=True)  
        assert result["income"].notna().all()


class TestSchemaConsistency:

    def test_test_output_matches_train_columns(self, messy_df, noise_target):
        pipeline = AutomatedEDA()
        train_out = pipeline.run_pipeline(messy_df, target=noise_target, is_train=True)

        test_out = pipeline.run_pipeline(messy_df.iloc[:300].copy(), target=None, is_train=False)
        assert list(test_out.columns) == list(train_out.columns)

    def test_missing_column_in_test_is_filled_with_zero(self, messy_df, noise_target):
        pipeline = AutomatedEDA()
        train_out = pipeline.run_pipeline(messy_df, target=noise_target, is_train=True)

        test_df = messy_df.iloc[:300].drop(columns=["gender"]).copy()
        test_out = pipeline.run_pipeline(test_df, target=None, is_train=False)

        assert list(test_out.columns) == list(train_out.columns)
        assert (test_out["gender"] == 0).all()

    def test_extra_column_in_test_is_dropped(self, messy_df, noise_target):
        pipeline = AutomatedEDA()
        train_out = pipeline.run_pipeline(messy_df, target=noise_target, is_train=True)

        test_df = messy_df.iloc[:300].copy()
        test_df["brand_new_col"] = 1
        test_out = pipeline.run_pipeline(test_df, target=None, is_train=False)

        assert list(test_out.columns) == list(train_out.columns)
        assert "brand_new_col" not in test_out.columns

    def test_calling_test_before_train_fails_loudly(self, messy_df):
        pipeline = AutomatedEDA()
        with pytest.raises(AttributeError):
            pipeline.run_pipeline(messy_df, target=None, is_train=False)


class TestTargetEncodingLeakage:

    def test_no_leakage_with_noise_target(self, messy_df, noise_target):

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(1)
        n, n_cats = 150, 15
        small_df = pd.DataFrame({"grp": rng.choice([f"grp_{i}" for i in range(n_cats)], size=n)})
        small_target = pd.Series(rng.integers(0, 2, size=n))

        cat_handler = HandleCategorical(small_df)
        encoded = cat_handler.encoder(target=small_target, is_train=True)

        assert "grp_TE" in encoded.columns
        corr = np.corrcoef(encoded["grp_TE"], small_target)[0, 1]
        assert abs(corr) < 0.2, f"grp_TE correlates with a noise target (corr={corr:.3f}) -- leakage regression"

        X = encoded[["grp_TE"]].values
        auc = roc_auc_score(small_target, LogisticRegression().fit(X, small_target).predict_proba(X)[:, 1])
        assert auc < 0.65, f"train AUC from grp_TE alone is {auc:.3f} -- leakage regression"

    def test_real_signal_still_captured(self, messy_df, informative_target):

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        cat_handler = HandleCategorical(messy_df)
        encoded = cat_handler.full_handler(target=informative_target, is_train=True)

        X = encoded[["occupation_TE"]].values
        model = LogisticRegression().fit(X, informative_target)
        auc = roc_auc_score(informative_target, model.predict_proba(X)[:, 1])
        assert auc > 0.65, f"train AUC ({auc:.3f}) too low -- target encoding may have lost real signal"


class TestOutlierHandling:

    def test_negative_skewed_column_gets_a_rule(self, messy_df):
        handler = HandleNumerical(messy_df)
        handler.num_imputer(is_train=True)
        handler.outlier_manager(is_train=True)
        assert "temp_change" in handler.outlier_rules_, "negative-valued outlier column got no rule at all"

    def test_outliers_are_actually_bounded_after_treatment(self, messy_df):
        handler = HandleNumerical(messy_df)
        handler.num_imputer(is_train=True)
        result = handler.outlier_manager(is_train=True)
        rule = handler.outlier_rules_.get("temp_change")
        if rule and rule["type"] == "clip":
            assert result["temp_change"].max() <= rule["upper"] + 1e-9
            assert result["temp_change"].min() >= rule["lower"] - 1e-9


class TestOutputSanity:

    def test_no_nans_in_final_output(self, messy_df, noise_target):
        out = AutomatedEDA().run_pipeline(messy_df, target=noise_target, is_train=True)
        assert not out.isna().any().any(), "final pipeline output contains NaNs"

    def test_all_columns_numeric(self, messy_df, noise_target):
        out = AutomatedEDA().run_pipeline(messy_df, target=noise_target, is_train=True)
        non_numeric = out.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        assert not non_numeric, f"non-numeric columns leaked into model-ready output: {non_numeric}"

    def test_exclude_param_respected_in_numerical_handler(self, messy_df):
        handler = HandleNumerical(messy_df)
        handler.num_imputer(is_train=True, exclude=["age"])
        assert "age" not in handler.impute_values_


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))