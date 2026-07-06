# lochan-eda

[![Tests](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml/badge.svg)](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://lochanjangid.github.io/lochan-eda/)
[![PyPI version](https://img.shields.io/pypi/v/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![Python versions](https://img.shields.io/pypi/pyversions/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![Downloads](https://static.pepy.tech/badge/lochan-eda)](https://pepy.tech/project/lochan-eda)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LochanJangid/lochan-eda/blob/main/LICENSE)

**An intelligent, stateful preprocessing pipeline that decides *how* to clean your data, prevents data leakage, and prepares raw datasets for production machine learning models.**

Most EDA helpers apply the same fixed rule to every column. `lochan-eda` acts as an automated data engineer — it inspects each column's statistical shape (missingness, skew, cardinality, sparsity, outlier ratio) and dynamically picks the right imputation, outlier-handling, and scaling strategy for it.

The core idea is **stateful memory**: the pipeline *learns* its rules on training data, saves them, and *blindly re-applies* those exact rules to test/inference data. No re-fitting, no leakage, no schema mismatches at deployment.

```bash
pip install lochan-eda
```

---

## Table of Contents

- [Why lochan-eda](#why-lochan-eda)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core Concept: `is_train`](#core-concept-is_train)
- [How It Decides](#how-it-decides)
- [API Reference](#api-reference)
  - [`AutomatedEDA`](#automatededa)
  - [`HandleNumerical`](#handlenumerical)
  - [`HandleCategorical`](#handlecategorical)
  - [Inspecting Learned Rules](#inspecting-learned-rules)
- [Full Example](#full-example-end-to-end)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why lochan-eda

A real preprocessing pass usually means writing the same boilerplate decision tree by hand, every time: *Is this column highly skewed? Does it have a long tail? Is this a high-cardinality string?*

`lochan-eda` automates that decision tree, and it is **not a black box** — every rule it applies is documented below and inspectable at runtime, so you always know why a column was scaled with `RobustScaler` instead of `StandardScaler`, or why a category got grouped into `"Other"`.

## Features

| Feature | Description |
|---|---|
| **Leakage-free architecture** | Strict `.fit()` / `.transform()` separation via an `is_train` flag — rules are learned once on training data and only ever applied (never re-learned) on test data. |
| **Adaptive imputation** | Chooses mean, median, or mode per column based on skewness and cardinality. |
| **Adaptive outlier handling** | Clips, winsorizes, or transforms tails depending on severity — never drops rows, so `X` and `y` shapes always stay aligned. |
| **Adaptive scaling** | Picks `StandardScaler`, `RobustScaler`, or `MaxAbsScaler` based on sparsity, skew, and outlier ratio. |
| **Smart categorical encoding** | Routes to binary mapping, one-hot encoding, frequency encoding, or `TargetEncoder` based on cardinality. |
| **Rare-category grouping** | Collapses low-frequency categories into `"Other"` to prevent high-dimensional noise. |
| **Schema-safe inference** | Test-time output is automatically reindexed to match the training schema — missing columns are filled with 0, unseen columns are dropped. |

## Installation

```bash
pip install lochan-eda
```

> **Requires:** Python ≥ 3.8 and `scikit-learn >= 1.3.0` (for native `TargetEncoder` support). `pandas`, `numpy`, `matplotlib`, and `seaborn` are installed automatically as dependencies.

## Quickstart

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

## Core Concept: `is_train`

Every method in the library takes an `is_train` flag. This is the entire leakage-prevention mechanism, so it's worth understanding once:

| `is_train=True` (fit + transform) | `is_train=False` (transform only) |
|---|---|
| Analyzes the dataframe's statistics (skew, missingness, cardinality, etc.) | Skips analysis entirely — uses whatever was learned during the last `is_train=True` call |
| Saves the resulting rules onto the handler instance | Applies the saved rules as-is, with no recalculation |
| Must be called **first**, on the training split | Must be called **after** a training pass on the *same* handler/pipeline instance |

> ⚠️ Calling a handler with `is_train=False` before it has ever been fit with `is_train=True` raises an `AttributeError` by design — there is nothing learned yet to apply.

## How It Decides

### Numerical columns (`HandleNumerical`)

| Step | Condition | Action |
|---|---|---|
| **Imputation** | Column has > 40% missing values | Column is dropped |
| | Uniqueness < 1% of rows | Treated as categorical-like → mode imputation |
| | Otherwise, `\|skew\| < 0.5` | Mean imputation |
| | Otherwise, `\|skew\| ≥ 0.5` | Median imputation |
| **Outliers** | Outlier share (IQR method) ≤ 3% | Clipped to IQR bounds |
| | Outlier share > 3%, moderate tail or negative values | Winsorized at the 5th / 95th percentile |
| | Outlier share > 3%, long tail, contains zeros | Square-root transform |
| | Outlier share > 3%, long tail, strictly positive | `log1p` transform |
| **Scaling** | Sparsity ≥ 50% zeros | `MaxAbsScaler` |
| | Skew > 1.0 and non-negative | `log1p` transform + `StandardScaler` |
| | Outlier ratio ≥ 5% | `RobustScaler` |
| | Otherwise | `StandardScaler` |

### Categorical columns (`HandleCategorical`)

| Step | Condition | Action |
|---|---|---|
| **Imputation** | Column has > 40% missing values | Column is dropped |
| | 10–40% missing | Filled with `"Unknown"` |
| | ≤ 10% missing | Mode imputation |
| **Rare grouping** | Category frequency < threshold (default 5%) | Grouped into `"Other"` |
| **Encoding** | ≤ 2 unique values | Binary integer mapping (0 / 1) |
| | 3–10 unique values | One-hot encoding (test columns aligned to train) |
| | > 10 unique values, `target` provided at fit time | `sklearn.preprocessing.TargetEncoder` |
| | > 10 unique values, no target provided | Frequency encoding |

## API Reference

### `AutomatedEDA`

The top-level orchestrator. Use this unless you specifically need to control the numerical or categorical pipeline in isolation.

```python
from lochan_eda import AutomatedEDA

eda = AutomatedEDA()
```

#### `AutomatedEDA.run_pipeline(df, target=None, is_train=True)`

Runs the full numerical + categorical pipeline end-to-end and returns a single, concatenated, fully numeric dataframe.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pandas.DataFrame` | required | The dataframe to process. |
| `target` | `pandas.Series` or `None` | `None` | Required if you want high-cardinality categorical columns to use `TargetEncoder` instead of frequency encoding. Only used when `is_train=True`. |
| `is_train` | `bool` | `True` | `True` to learn rules on this dataframe (training split). `False` to apply previously learned rules (test/inference split). |

**Returns:** `pandas.DataFrame` — fully numeric, `NaN`-free. On `is_train=False`, columns are automatically reindexed to match the training output (missing columns filled with `0`, extra columns dropped).

```python
X_train_clean = eda.run_pipeline(df=X_train, target=y_train, is_train=True)
X_test_clean  = eda.run_pipeline(df=X_test, is_train=False)
```

### `HandleNumerical`

Use this directly if you want to process only numeric columns, or need granular control (e.g. excluding an ID column from processing).

```python
from lochan_eda import HandleNumerical

num_handler = HandleNumerical(X_train)   # auto-selects numeric columns from X_train
```

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `num_imputer(is_train=True, exclude=None)` | `exclude`: column name or list of names to skip | `pandas.DataFrame` | Fills missing values using mean, median, or mode depending on column shape. |
| `outlier_manager(is_train=True, exclude=None)` | same | `pandas.DataFrame` | Clips, winsorizes, or transforms outliers per the rules above. |
| `scaler(is_train=True, exclude=None)` | same | `pandas.DataFrame` | Fits/applies the chosen `sklearn` scaler per column. |
| `full_handler(is_train=True)` | — | `pandas.DataFrame` | Runs `num_imputer` → `outlier_manager` → `scaler` in sequence. |

> **Note:** `full_handler()` does not currently forward an `exclude` argument — it always processes every numeric column. If you need to protect a column (e.g. `customer_id`) from processing, call the three granular methods yourself, as shown below.

**Granular usage (train → test):**

```python
from lochan_eda import HandleNumerical

# Instantiate once, on the raw training dataframe
num_handler = HandleNumerical(X_train)

# TRAIN: learn + apply rules, skipping an ID-like column
X_train_num = num_handler.num_imputer(is_train=True, exclude="customer_id")
X_train_num = num_handler.outlier_manager(is_train=True, exclude="customer_id")
X_train_num = num_handler.scaler(is_train=True, exclude="customer_id")

# TEST: swap in the test data, then re-apply the exact same learned rules
num_handler.num_df = X_test.select_dtypes(include=["number"]).copy()
X_test_num = num_handler.num_imputer(is_train=False, exclude="customer_id")
X_test_num = num_handler.outlier_manager(is_train=False, exclude="customer_id")
X_test_num = num_handler.scaler(is_train=False, exclude="customer_id")
```

**Simple usage (no exclusions needed):**

```python
X_train_num = num_handler.full_handler(is_train=True)

num_handler.num_df = X_test.select_dtypes(include=["number"]).copy()
X_test_num = num_handler.full_handler(is_train=False)
```

### `HandleCategorical`

Use this directly if you want to process only categorical/object/string columns.

```python
from lochan_eda import HandleCategorical

cat_handler = HandleCategorical(X_train)   # auto-selects category/object/string columns
```

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `cat_imputer(is_train=True, exclude=None)` | `exclude`: column name or list to skip | `pandas.DataFrame` | Fills missing values with mode or `"Unknown"` depending on missingness %. |
| `rare_manager(is_train=True, threshold=0.05, exclude=None)` | `threshold`: min frequency to avoid being grouped | `pandas.DataFrame` | Groups low-frequency categories into `"Other"`. |
| `encoder(is_train=True, target=None, exclude=None)` | `target`: series for `TargetEncoder` on high-cardinality columns | `pandas.DataFrame` | Encodes categories to numbers per the cardinality rules above. |
| `full_handler(target=None, is_train=True)` | `target`: passed through to `encoder` | `pandas.DataFrame` | Runs `cat_imputer` → `rare_manager` → `encoder` in sequence. |

```python
# TRAIN
X_train_cat = cat_handler.full_handler(target=y_train, is_train=True)

# TEST — swap in test data, keep the same handler instance
cat_handler.cat_df = X_test.select_dtypes(include=["category", "object", "string"]).copy()
X_test_cat = cat_handler.full_handler(target=None, is_train=False)
```

> Unseen categories at test time don't crash the pipeline — a binary-mapped column maps them to `-1`, a one-hot column gets all zeros, and a frequency-encoded column gets `0`.

### Inspecting Learned Rules

Because nothing here is a black box, every learned rule is stored as a plain attribute on the handler so you can audit *why* a decision was made:

```python
eda = AutomatedEDA()
eda.run_pipeline(df=X_train, target=y_train, is_train=True)

# Which columns were dropped, and why?
print(eda.num_handler.drop_cols_)      # numeric columns dropped (>40% missing)
print(eda.cat_handler.drop_cols_)      # categorical columns dropped (>40% missing)

# What imputation value did each column get?
print(eda.num_handler.impute_values_)  # {column: fill_value}

# What outlier rule and scaler did each numeric column get?
print(eda.num_handler.outlier_rules_)  # {column: {"type": "clip"/"winsorize"/"sqrt"/"log1p", ...}}
print(eda.num_handler.scalers_)        # {column: fitted sklearn scaler object}

# What encoding strategy did each categorical column get?
print(eda.cat_handler.encode_types_)   # {column: "binary"/"ohe"/"te"/"freq"}

# The exact column order the final model-ready output must match
print(eda.final_columns_)
```

| Attribute | Owner | Description |
|---|---|---|
| `drop_cols_` | `HandleNumerical`, `HandleCategorical` | Columns dropped for exceeding the 40% missing-value threshold. |
| `impute_values_` | `HandleNumerical` | `{column: fill_value}` used for imputation. |
| `outlier_rules_` | `HandleNumerical` | `{column: rule_dict}` describing the outlier treatment applied. |
| `scalers_` | `HandleNumerical` | `{column: fitted scaler}`. |
| `impute_modes_` | `HandleCategorical` | `{column: mode_value}` used for imputation. |
| `rare_cats_` | `HandleCategorical` | `{column: [rare category values]}` grouped into `"Other"`. |
| `encode_types_` | `HandleCategorical` | `{column: encoding strategy}`. |
| `final_columns_` | `AutomatedEDA` | Full training-time output column order, used to align test-time output. |

## Full Example (end-to-end)

This script is fully self-contained — copy it into a file and run it to see the pipeline work on synthetic, realistically messy data.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda import AutomatedEDA

# 1. Create a messy, realistic dataset
rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame({
    "income": rng.exponential(scale=45000, size=n),
    "age": rng.normal(35, 10, n).clip(18, 90),
    "city": rng.choice(["Jaipur", "Delhi", "Mumbai", "Bangalore", "Chennai"], size=n),
    "occupation": rng.choice([f"occ_{i}" for i in range(15)], size=n),
    "gender": rng.choice(["M", "F"], size=n),
})
target = pd.Series(rng.integers(0, 2, size=n), name="churn")

# Inject missing values, like real-world data
for col in ["income", "age", "city"]:
    df.loc[rng.random(n) < 0.08, col] = np.nan

# 2. Split BEFORE preprocessing — this is what prevents leakage
X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42
)

# 3. Fit on train, transform test using the exact same learned rules
eda = AutomatedEDA()
X_train_clean = eda.run_pipeline(X_train, target=y_train, is_train=True)
X_test_clean = eda.run_pipeline(X_test, is_train=False)

print("Train shape:", X_train_clean.shape)
print("Test shape :", X_test_clean.shape)
print("Columns match:", list(X_train_clean.columns) == list(X_test_clean.columns))
```

## FAQ / Troubleshooting

**I got `AttributeError` when calling `run_pipeline(..., is_train=False)`.**
You must call the pipeline once with `is_train=True` before ever calling it with `is_train=False`, on that same instance. There is nothing learned to apply until the first training pass runs.

**A column I expected is missing from my output.**
Check `eda.num_handler.drop_cols_` and `eda.cat_handler.drop_cols_` — columns with more than 40% missing values are dropped automatically during the training pass.

**How do I know why a column got a specific encoder or scaler?**
See [Inspecting Learned Rules](#inspecting-learned-rules) — every decision is stored as a readable attribute on the handler.

**My test set has a category value that never appeared in training. Will it crash?**
No. Unseen categories are handled gracefully: binary columns map them to `-1`, one-hot columns get all zeros, and frequency-encoded columns get `0`.

**Can I change the 40% missing-value or 5% rare-category thresholds?**
The rare-category threshold is configurable via `rare_manager(threshold=...)`. The missing-value drop threshold (40%) and outlier threshold (3%) are currently fixed constants — see [Roadmap](#roadmap).

## Testing

```bash
git clone https://github.com/LochanJangid/lochan-eda.git
cd lochan-eda
pip install -e .
pip install pytest
pytest -v
```

## Contributing

Issues and PRs are welcome. Please run `pytest -v` before submitting a pull request to verify structural integrity across train/test splits.

## License

MIT © [Lochan Jangid](https://github.com/LochanJangid)
