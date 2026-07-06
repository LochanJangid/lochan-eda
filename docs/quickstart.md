# Quickstart

## Installation

```bash
pip install lochan-eda
```

!!! note "Requirements"
    Python ≥ 3.8 and `scikit-learn >= 1.3.0` (needed for native `TargetEncoder` support). `pandas`, `numpy`, `matplotlib`, and `seaborn` install automatically as dependencies.

## Your first pipeline

Always split your data into train/test **before** passing it through the pipeline — that ordering is what makes the leakage-free guarantee possible.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda import AutomatedEDA

df = pd.read_csv("your_data.csv")

# 1. Split your data first
X = df.drop(columns="Target_Column")
y = df["Target_Column"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Initialize the pipeline engine
eda = AutomatedEDA()

# 3. TRAIN MODE — learns the rules, saves them, and cleans X_train
X_train_clean = eda.run_pipeline(df=X_train, target=y_train, is_train=True)

# 4. TEST MODE — blindly applies the saved rules to X_test (no re-fitting)
X_test_clean = eda.run_pipeline(df=X_test, is_train=False)
```

That's it — `X_train_clean` and `X_test_clean` are fully numeric, `NaN`-free, and share identical columns, ready to feed into any scikit-learn model.

## What just happened?

`run_pipeline` internally routes numeric columns through [`HandleNumerical`](api/handle-numerical.md) and categorical/object/string columns through [`HandleCategorical`](api/handle-categorical.md), then concatenates the results and aligns the schema. Read [Core Concepts](concepts.md) next to understand exactly how each decision is made.
