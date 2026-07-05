# test_eda.py
import pytest
import pandas as pd
import numpy as np
from lochan_eda.numerical import HandleNumerical
from lochan_eda.categorical import HandleCategorical
from lochan_eda.orchestrator import AutomatedEDA

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n_rows = 1000

    df = pd.DataFrame({
        # Numerical
        # Ensure the split sum equals n_rows exactly
        "num_drop": [np.nan] * 450 + list(np.random.randn(n_rows - 450)), 
        "num_cat_like": np.random.choice([0, 1, np.nan], size=n_rows),
        "num_skewed": np.random.exponential(scale=2, size=n_rows),
        "num_normal": np.random.normal(0, 1, size=n_rows),
        "num_sparse": np.random.choice([0, 1.5, 2.5], p=[0.6, 0.2, 0.2], size=n_rows),
        "num_outliers": np.random.normal(50, 10, size=n_rows),
        
        # Categorical
        "cat_drop": [np.nan] * 450 + ["A"] * (n_rows - 450), 
        "cat_mode": np.random.choice(["X", "Y", np.nan], p=[0.6, 0.3, 0.1], size=n_rows),
        "cat_unknown": np.random.choice(["M", "F", np.nan], p=[0.4, 0.4, 0.2], size=n_rows),
        "cat_rare": np.random.choice(["Common", "Rare1", "Rare2"], p=[0.92, 0.04, 0.04], size=n_rows),
        "cat_binary": np.random.choice(["Yes", "No"], size=n_rows),
        "cat_ohe": np.random.choice(["Red", "Blue", "Green"], size=n_rows),
        "cat_freq": np.random.choice([f"Type_{i}" for i in range(15)], size=n_rows)
    })
    
    # Inject specific anomalies
    df.loc[10:30, "num_skewed"] = np.nan
    df.loc[50:150, "num_normal"] = np.nan
    df.loc[0:15, "num_outliers"] = 5000  # Inject extreme outliers to trigger clipper
    
    return df


# ==========================================
# TEST: HandleNumerical
# ==========================================
def test_num_imputer(sample_df):
    handler = HandleNumerical(sample_df)
    df_clean = handler.num_imputer(is_train=True)
    
    # Check drop > 40%
    assert "num_drop" not in df_clean.columns
    # Check cat-like imputation (should not have NaNs)
    assert df_clean["num_cat_like"].isna().sum() == 0
    # Check continuous imputation
    assert df_clean["num_skewed"].isna().sum() == 0
    assert df_clean["num_normal"].isna().sum() == 0

def test_outlier_manager(sample_df):
    handler = HandleNumerical(sample_df)
    handler.num_imputer(is_train=True) 
    df_clean = handler.outlier_manager(is_train=True)
    
    # SHAPE CHECK: Ensure rows were CLIPPED, not dropped
    assert df_clean.shape[0] == sample_df.shape[0]
    
    # Check clipping: the injected extreme values (5000) should be pulled inward
    assert df_clean["num_outliers"].max() < 1000

def test_scaler(sample_df):
    handler = HandleNumerical(sample_df)
    handler.num_imputer(is_train=True)
    handler.outlier_manager(is_train=True)
    df_clean = handler.scaler(is_train=True)
    
    # Check standard scaler (mean should be extremely close to 0)
    assert np.isclose(df_clean["num_normal"].mean(), 0, atol=0.1)
    # Check MaxAbsScaler for sparse data (max should be 1)
    assert np.isclose(df_clean["num_sparse"].max(), 1.0, atol=0.1)


# ==========================================
# TEST: HandleCategorical
# ==========================================
def test_cat_imputer(sample_df):
    handler = HandleCategorical(sample_df)
    df_clean = handler.cat_imputer(is_train=True)
    
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
    handler.cat_imputer(is_train=True)
    df_clean = handler.rare_manager(is_train=True, threshold=0.05)
    
    # 'Rare1' and 'Rare2' should be replaced by 'Other'
    assert "Rare1" not in df_clean["cat_rare"].values
    assert "Other" in df_clean["cat_rare"].values

def test_encoder(sample_df):
    handler = HandleCategorical(sample_df)
    handler.cat_imputer(is_train=True)
    df_clean = handler.encoder(is_train=True)
    
    # Binary should map strictly to 0 and 1
    assert set(df_clean["cat_binary"].dropna().unique()).issubset({0, 1})
    # OHE should create new columns prefixed with the original column name
    assert any(col.startswith("cat_ohe_") for col in df_clean.columns)
    # Frequency encoding should create a numeric column ending in _Freq
    assert "cat_freq_Freq" in df_clean.columns
    assert pd.api.types.is_numeric_dtype(df_clean["cat_freq_Freq"])


# ==========================================
# TEST: Orchestrator Pipeline (Memory Test)
# ==========================================
def test_run_pipeline_memory(sample_df):
    """
    Critically tests the Stateful Memory architecture by simulating 
    a Train/Test split and ensuring test data matches train structure.
    """
    # 1. Simulate a Train/Test split
    train_df = sample_df.iloc[:800].copy()
    test_df = sample_df.iloc[800:].copy()

    # 2. Initialize the pipeline engine
    eda = AutomatedEDA()
    
    # 3. Process Train Data (Learns the rules)
    final_train_df = eda.run_pipeline(df=train_df, is_train=True)
    
    # 4. Process Test Data (Applies rules blindly)
    final_test_df = eda.run_pipeline(df=test_df, is_train=False)
    
    # ==================================
    # STRICT ASSERTIONS
    # ==================================
    assert isinstance(final_train_df, pd.DataFrame)
    
    # Verify shape integrity (no rows dropped during pipeline)
    assert final_train_df.shape[0] == 800
    assert final_test_df.shape[0] == 200

    # Verify structural alignment (Columns must match perfectly)
    assert list(final_train_df.columns) == list(final_test_df.columns)

    # Verify scaling didn't crash and left no NaNs
    assert final_test_df["num_normal"].isna().sum() == 0