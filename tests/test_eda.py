# test_eda.py
import pytest
import pandas as pd
import numpy as np
from lochan_eda import HandleNumerical
from lochan_eda import HandleCategorical
from lochan_eda import AutomatedEDA

@pytest.fixture
def sample_df():
    """Loads the test dataset before each test function runs."""
    return pd.read_csv("tests/test_dataset.csv")

# ==========================================
# TEST: HandleNumerical
# ==========================================
def test_num_imputer(sample_df):
    handler = HandleNumerical(sample_df)
    df_clean = handler.num_imputer()
    
    # Check drop > 40%
    assert "num_drop" not in df_clean.columns
    # Check cat-like imputation (should not have NaNs)
    assert df_clean["num_cat_like"].isna().sum() == 0
    # Check continuous imputation
    assert df_clean["num_skewed"].isna().sum() == 0
    assert df_clean["num_normal"].isna().sum() == 0

def test_outlier_manager(sample_df):
    handler = HandleNumerical(sample_df)
    handler.num_imputer() # Clean NaNs first
    df_clean = handler.outlier_manager()
    
    # Check trimming (< 3% outliers). The 20 injected extreme values should be gone.
    assert df_clean["num_outliers"].max() < 1000
    # Ensure rows were actually dropped, not just replaced with NaN
    assert df_clean["num_outliers"].isna().sum() == 0

def test_scaler(sample_df):
    handler = HandleNumerical(sample_df)
    handler.num_imputer()
    handler.outlier_manager()
    df_clean = handler.scaler()
    
    # Check standard scaler (mean ~ 0)
    assert np.isclose(df_clean["num_normal"].mean(), 0, atol=0.1)
    # Check MaxAbsScaler for sparse data (max should be 1)
    assert np.isclose(df_clean["num_sparse"].max(), 1.0, atol=0.1)

# ==========================================
# TEST: HandleCategorical
# ==========================================
def test_cat_imputer(sample_df):
    handler = HandleCategorical(sample_df)
    df_clean = handler.cat_imputer()
    
    # Check drop > 40%
    assert "cat_drop" not in df_clean.columns
    # Check <= 10% mode imputation
    assert df_clean["cat_mode"].isna().sum() == 0
    assert "Unknown" not in df_clean["cat_mode"].values
    # Check > 10% 'Unknown' imputation
    assert df_clean["cat_unknown"].isna().sum() == 0
    assert "Unknown" in df_clean["cat_unknown"].values

def test_rare_manager(sample_df):
    handler = HandleCategorical(sample_df)
    handler.cat_imputer()
    df_clean = handler.rare_manager(threshold=0.05)
    
    # 'Rare1' and 'Rare2' should be replaced by 'Other'
    assert "Rare1" not in df_clean["cat_rare"].values
    assert "Other" in df_clean["cat_rare"].values

def test_encoder(sample_df):
    handler = HandleCategorical(sample_df)
    handler.cat_imputer()
    df_clean = handler.encoder()
    
    # Binary should map to 0 and 1
    assert set(df_clean["cat_binary"].dropna().unique()).issubset({0, 1})
    # OHE should create new columns prefixed with the original column name
    assert any(col.startswith("cat_ohe_") for col in df_clean.columns)
    # Frequency encoding should create a numeric column
    assert "cat_freq_Freq" in df_clean.columns
    assert pd.api.types.is_numeric_dtype(df_clean["cat_freq_Freq"])

# ==========================================
# TEST: Orchestrator Pipeline
# ==========================================
def test_run_pipeline(sample_df):
    eda = AutomatedEDA(sample_df)
    final_df = eda.run_pipeline()
    
    # Check that output is a DataFrame and combines both types
    assert isinstance(final_df, pd.DataFrame)
    
    # Verify dropped columns from both handlers are gone
    assert "num_drop" not in final_df.columns
    assert "cat_drop" not in final_df.columns
    
    # Verify encoding happened (presence of OHE columns)
    assert any(col.startswith("cat_ohe_") for col in final_df.columns)
    
    # Verify scaling happened (no NaNs left in numeric, mean around 0 for normal)
    assert final_df["num_normal"].isna().sum() == 0