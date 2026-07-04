# lochan-eda

[![Tests](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml/badge.svg)](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![Python versions](https://img.shields.io/pypi/pyversions/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**An automated preprocessing pipeline that decides *how* to clean your data, not just that it should be cleaned.**

Most EDA helpers apply the same fixed rule to every column — fill all NaNs with the mean, scale everything with `StandardScaler`. `lochan-eda` looks at each column's actual statistical shape — missingness, skew, cardinality, sparsity, outlier ratio — and picks the appropriate imputation, outlier, and scaling strategy for that column specifically. One call in, a model-ready DataFrame out.

```python
from lochan_eda import AutomatedEDA

clean_df = AutomatedEDA(df).run_pipeline()
```

---

## Why

A real preprocessing pass usually means writing the same decision tree by hand, every time: *is this column mostly missing? is it skewed or symmetric? does it have a handful of extreme values or a long tail? is it high-cardinality categorical?* — before you can even decide how to impute, scale, or encode it.

`lochan-eda` encodes that decision tree in code. It's not a black box: every rule it applies is documented below, so you always know why a column was scaled with `RobustScaler` instead of `StandardScaler`, or why a category got grouped into `"Other"`.

## Features

- **Adaptive missing-value imputation** — different strategy for high-missingness, low-cardinality, and normally-distributed columns
- **Adaptive outlier handling** — trims, winsorizes, or transforms depending on how much of the column is actually affected
- **Adaptive scaling** — picks `StandardScaler`, `RobustScaler`, or `MaxAbsScaler` based on sparsity, skew, and outlier ratio
- **Adaptive categorical encoding** — binary mapping, one-hot, frequency, or target encoding based on cardinality
- **Rare-category grouping** — collapses low-frequency categories into `"Other"` before encoding
- **One-call pipeline** or **granular per-step control** — use `AutomatedEDA` end-to-end, or call `HandleNumerical` / `HandleCategorical` directly
- Built on `pandas` and `scikit-learn` — output is a plain DataFrame, drop it straight into any model

## Installation

```bash
pip install lochan-eda
```

## Quickstart

```python
import pandas as pd
from lochan_eda import AutomatedEDA

df = pd.read_csv("your_data.csv")

eda = AutomatedEDA(df)
clean_df = eda.run_pipeline()
```

`run_pipeline()` runs the full sequence — impute, handle outliers, scale numerical columns; impute, group rare categories, encode categorical columns — and returns a single combined DataFrame with no missing values, aligned on index.

### Verified example

Run against the 1,000-row test fixture shipped in this repo (`tests/test_dataset.csv`, 6 numerical + 7 categorical columns with injected missing values, outliers, and rare categories):

```python
>>> df.shape
(1000, 13)
>>> df.isna().sum().sort_values(ascending=False).head(4)
num_drop       450
cat_drop       450
cat_unknown    200
cat_mode        50

>>> clean_df = AutomatedEDA(df).run_pipeline()
>>> clean_df.shape
(938, 13)
>>> clean_df.isna().sum().sum()
0
>>> list(clean_df.columns)
['num_cat_like', 'num_skewed', 'num_normal', 'num_sparse', 'num_outliers',
 'cat_mode', 'cat_unknown', 'cat_rare', 'cat_binary',
 'cat_ohe_Blue', 'cat_ohe_Green', 'cat_ohe_Red', 'cat_freq_Freq']
```

The two >40%-missing columns (`num_drop`, `cat_drop`) were dropped, every remaining missing value was imputed, 62 outlier rows were trimmed from `num_outliers`, and `cat_ohe` was one-hot encoded — with zero manual decisions.

## How it decides

**Numerical columns** (`HandleNumerical`)

| Step | Condition | Action |
|---|---|---|
| Imputation | >40% missing | Column dropped |
| | Uniqueness < 1% of rows (categorical-like numeric) | Mode imputation |
| | 0–40% missing, \|skew\| < 0.5 | Mean imputation |
| | 0–40% missing, \|skew\| ≥ 0.5 | Median imputation |
| Outliers (IQR method) | Outlier share ≤ 3% | Rows trimmed |
| | Outlier share > 3%, spiked tail | Winsorized at 5th/95th percentile |
| | Outlier share > 3%, long tail, contains zeros | Square-root transform |
| | Outlier share > 3%, long tail, no zeros/negatives | Log1p transform |
| Scaling | Sparsity ≥ 50% zeros | `MaxAbsScaler` |
| | Skew > 1.0, non-negative | Log1p transform + `StandardScaler` |
| | Outlier ratio ≥ 5% | `RobustScaler` |
| | Otherwise | `StandardScaler` |

**Categorical columns** (`HandleCategorical`)

| Step | Condition | Action |
|---|---|---|
| Imputation | >40% missing | Column dropped |
| | >10–40% missing | Filled with `"Unknown"` |
| | ≤10% missing | Mode imputation |
| Rare grouping | Category frequency < threshold (default 5%) | Grouped into `"Other"` |
| Encoding | ≤2 unique values | Binary integer mapping |
| | 3–10 unique values | One-hot encoding |
| | >10 unique values, `target` provided | Target encoding |
| | >10 unique values, no target | Frequency encoding |

## API reference

| Class / method | Description |
|---|---|
| `AutomatedEDA(df)` | Orchestrates the full pipeline |
| `.run_pipeline()` | Runs numerical + categorical handling end-to-end, returns one combined DataFrame |
| `HandleNumerical(df)` | Isolates numeric columns for standalone use |
| `.num_imputer(exclude=None)` | Missing-value imputation only |
| `.outlier_manager(exclude=None)` | Outlier handling only |
| `.scaler(exclude=None)` | Scaling only |
| `.full_handler()` | Imputer → outlier manager → scaler, in order |
| `HandleCategorical(df)` | Isolates categorical columns for standalone use |
| `.cat_imputer()` | Missing-value imputation only |
| `.rare_manager(threshold=0.05)` | Rare-category grouping only |
| `.encoder(target=None)` | Encoding only; pass `target` to enable target encoding on high-cardinality columns |
| `.full_handler()` | Imputer → rare manager → encoder, in order |

Every numerical method takes an optional `exclude` (str or list) to skip specific columns — useful for keeping an ID column or a pre-engineered feature untouched.

### Using components individually

```python
from lochan_eda import HandleNumerical, HandleCategorical

num = HandleNumerical(df)
num.num_imputer(exclude="customer_id")
num.outlier_manager()
scaled_df = num.scaler()

cat = HandleCategorical(df)
cat.cat_imputer()
cat.rare_manager(threshold=0.02)
encoded_df = cat.encoder(target=df["churned"])
```

## Testing

```bash
git clone https://github.com/LochanJangid/lochan-eda.git
cd lochan-eda
pip install -e .
pip install pytest
pytest -v
```

## Contributing

Issues and PRs are welcome. Please run `pytest` before submitting a pull request.

## License

MIT © Lochan Jangid 

## Author

**Lochan Jangid**
GitHub: [@LochanJangid](https://github.com/LochanJangid) 