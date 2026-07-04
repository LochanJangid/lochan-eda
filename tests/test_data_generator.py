# generate_test_data.py
import pandas as pd
import numpy as np

def create_test_csv(filename="test_dataset.csv"):
    np.random.seed(42)
    n = 1000  # 1000 rows for stable percentile/skew calculations

    df = pd.DataFrame({
        # --- NUMERICAL ---
        "num_drop": [np.nan if i < 450 else 1 for i in range(n)],          # 45% missing (>40%)
        "num_cat_like": np.random.choice([1, 2, 3], size=n).astype(float), # Low cardinality (<1% unique)
        "num_skewed": np.random.exponential(scale=2, size=n),              # Highly skewed
        "num_normal": np.random.normal(loc=0, scale=1, size=n),            # Normal distribution
        "num_sparse": [0 if i < 600 else np.random.rand() for i in range(n)], # 60% zeros (sparsity > 0.5)
        "num_outliers": np.random.normal(0, 1, n),                         # Base for outliers
        
        # --- CATEGORICAL ---
        "cat_drop": [np.nan if i < 450 else "A" for i in range(n)],        # 45% missing (>40%)
        "cat_mode": [np.nan if i < 50 else "X" for i in range(n)],         # 5% missing (<=10%)
        "cat_unknown": [np.nan if i < 200 else "M" for i in range(n)],     # 20% missing (>10%, <=40%)
        "cat_rare": ["Rare1"]*5 + ["Rare2"]*5 + ["Common"]*990,            # Rare categories (<5%)
        "cat_binary": np.random.choice(["Yes", "No"], size=n),             # <= 2 unique
        "cat_ohe": np.random.choice(["Red", "Green", "Blue", "Black"], n), # <= 10 unique
        "cat_freq": np.random.choice([f"Type_{i}" for i in range(15)], n)  # > 10 unique
    })

    # Inject NaNs for imputation tests
    df.loc[0:20, ["num_cat_like", "num_skewed", "num_normal"]] = np.nan
    
    # Inject 20 extreme outliers (~2%, triggers trim logic)
    df.loc[0:19, "num_outliers"] = 1000 

    df.to_csv(filename, index=False)
    print(f"Test data saved to {filename}")

if __name__ == "__main__":
    create_test_csv()