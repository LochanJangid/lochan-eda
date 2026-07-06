# `HandleNumerical`

Use this directly if you want to process only numeric columns, or need granular control (e.g. excluding an ID column from processing). See [Concepts → How it decides: numerical columns](../concepts.md#how-it-decides-numerical-columns) for the full decision logic.

```python
from lochan_eda import HandleNumerical

num_handler = HandleNumerical(X_train)   # auto-selects numeric columns from X_train
```

## Methods

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `num_imputer(is_train=True, exclude=None)` | `exclude`: column name or list of names to skip | `pandas.DataFrame` | Fills missing values using mean, median, or mode depending on column shape. |
| `outlier_manager(is_train=True, exclude=None)` | same | `pandas.DataFrame` | Clips, winsorizes, or transforms outliers. |
| `scaler(is_train=True, exclude=None)` | same | `pandas.DataFrame` | Fits/applies the chosen `sklearn` scaler per column. |
| `full_handler(is_train=True)` | — | `pandas.DataFrame` | Runs `num_imputer` → `outlier_manager` → `scaler` in sequence. |

!!! info "`exclude` is not available on `full_handler`"
    `full_handler()` does not currently forward an `exclude` argument — it always processes every numeric column. If you need to protect a column (e.g. `customer_id`) from processing, call the three granular methods yourself, as shown below.

## Granular usage (train → test)

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

## Simple usage (no exclusions needed)

```python
X_train_num = num_handler.full_handler(is_train=True)

num_handler.num_df = X_test.select_dtypes(include=["number"]).copy()
X_test_num = num_handler.full_handler(is_train=False)
```

## Learned attributes

| Attribute | Description |
|---|---|
| `drop_cols_` | Numeric columns dropped for exceeding the 40% missing-value threshold. |
| `impute_values_` | `{column: fill_value}` used for imputation. |
| `outlier_rules_` | `{column: rule_dict}` describing the outlier treatment applied. |
| `scalers_` | `{column: fitted scaler object}`. |
| `transforms_` | `{column: "log1p"}` for columns that received a log transform before scaling. |

See [Inspecting Learned Rules](inspecting-rules.md) for full usage examples.
