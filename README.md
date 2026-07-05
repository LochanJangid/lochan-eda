```markdown
# lochan-eda

[![Tests](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml/badge.svg)](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![Python versions](https://img.shields.io/pypi/pyversions/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**An intelligent, stateful preprocessing pipeline that decides *how* to clean your data, prevents data leakage, and prepares raw datasets for production machine learning models.**

Most EDA helpers apply the same fixed rule to every column. `lochan-eda` acts as an automated data engineer. It analyzes each column's statistical shape — missingness, skew, cardinality, sparsity, outlier ratio — and dynamically picks the appropriate imputation, outlier, and scaling strategy. 

Crucially, **v0.1.0 introduces Stateful Memory**. It learns the rules on your training data, saves them, and blindly applies those exact rules to your testing data. Zero data leakage. Zero deployment crashes.

---

## Why lochan-eda?

A real preprocessing pass usually means writing the same boilerplate decision tree by hand, every time. *Is this column highly skewed? Does it have a long tail? Is this a high-cardinality string?*

`lochan-eda` automates that decision tree. It is not a black box: every rule it applies is documented below. You always know why a column was scaled with `RobustScaler` instead of `StandardScaler`, or why a category got grouped into `"Other"`.

## Features

- **Leakage-Free Architecture (New in v0.1.0)** — strictly separates `.fit()` learning from `.transform()` application using an `is_train` flag to prevent train-test contamination.
- **Adaptive Imputation** — deploys mean, median, or mode strategies dynamically based on column skewness and data behavior.
- **Adaptive Outlier Handling** — softly clips, winsorizes, or mathematically transforms tails depending on severity, without ever dropping rows (ensuring `X` and `y` shapes never mismatch).
- **Adaptive Scaling** — selects `StandardScaler`, `RobustScaler`, or `MaxAbsScaler` based on sparsity and outlier ratios.
- **Smart Categorical Encoding** — routes to binary mapping, one-hot encoding, frequency, or `TargetEncoder` based on unique cardinality.
- **Rare-Category Grouping** — collapses low-frequency categories into `"Other"` to prevent high-dimensional noise.

## Installation

```bash
pip install lochan-eda

```

*Note: Requires `scikit-learn >= 1.3.0` for native `TargetEncoder` support.*

## Quickstart (Machine Learning Workflow)

To prevent data leakage, always split your data into Training and Testing sets *before* passing it through the pipeline.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda import AutomatedEDA

df = pd.read_csv("your_data.csv")

# 1. Split your data first
X = df.drop(columns="Target_Column")
y = df["Target_Column"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize the pipeline engine
eda = AutomatedEDA()

# 3. TRAIN MODE: Pipeline analyzes distributions, saves the rules, and cleans X_train
X_train_clean = eda.run_pipeline(df=X_train, target=y_train, is_train=True)

# 4. TEST MODE: Pipeline blindly applies the saved rules to X_test (No Leakage)
X_test_clean = eda.run_pipeline(df=X_test, is_train=False)

```

## How It Decides

### Numerical Columns (`HandleNumerical`)

| Step | Condition | Action |
| --- | --- | --- |
| **Imputation** | >40% missing | Column dropped |
|  | Uniqueness < 1% of rows | Mode imputation (treats as categorical) |
|  | 0–40% missing, |skew| < 0.5 | Mean imputation |
|  | 0–40% missing, |skew| ≥ 0.5 | Median imputation |
| **Outliers** | Outlier share ≤ 3% | Clipped strictly to IQR bounds |
|  | Outlier share > 3%, spiked tail | Winsorized at 5th/95th percentile |
|  | Outlier share > 3%, long tail, contains zeros | Square-root transform |
|  | Outlier share > 3%, long tail, strict positive | Log1p transform |
| **Scaling** | Sparsity ≥ 50% zeros | `MaxAbsScaler` |
|  | Skew > 1.0, non-negative | Log1p transform + `StandardScaler` |
|  | Outlier ratio ≥ 5% | `RobustScaler` |
|  | Otherwise | `StandardScaler` |

### Categorical Columns (`HandleCategorical`)

| Step | Condition | Action |
| --- | --- | --- |
| **Imputation** | >40% missing | Column dropped |
|  | >10–40% missing | Filled with `"Unknown"` |
|  | ≤10% missing | Mode imputation |
| **Rare Grouping** | Frequency < threshold (default 5%) | Grouped into `"Other"` |
| **Encoding** | ≤2 unique values | Binary integer mapping (0 and 1) |
|  | 3–10 unique values | One-hot encoding (aligns test columns perfectly) |
|  | >10 unique values, `target` provided | Scikit-Learn `TargetEncoder` |
|  | >10 unique values, no target | Frequency encoding |

## API Reference

### Orchestration

| Class / Method | Description |
| --- | --- |
| `AutomatedEDA()` | Initializes the persistent memory manager. |
| `.run_pipeline(df, target, is_train)` | Runs the end-to-end numerical and categorical pipeline. Set `is_train=True` for training data and `is_train=False` for testing/inference data. |

### Component Level

You can use the numerical or categorical engines individually. Ensure you instantiate the object once, then pass `is_train=True` and `is_train=False` to the processing methods.

```python
from lochan_eda import HandleNumerical, HandleCategorical

# Initialize
num_handler = HandleNumerical(X_train)

# Learn & Clean Train Data
X_train_num = num_handler.full_handler(is_train=True, exclude="customer_id")

# Inject & Clean Test Data
num_handler.num_df = X_test.select_dtypes(include=["number"]).copy()
X_test_num = num_handler.full_handler(is_train=False, exclude="customer_id")

```

## Testing

```bash
git clone [https://github.com/lochanjangid/lochan-eda.git](https://github.com/lochanjangid/lochan-eda.git)
cd lochan-eda
pip install -e .
pip install pytest
pytest -v

```

## Contributing

Issues and PRs are heavily welcomed. Please ensure you run `pytest` before submitting a pull request to verify structural integrity across Train/Test splits.

## License

MIT © Lochan Jangid

```

```