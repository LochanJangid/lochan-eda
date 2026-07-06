# Core Concepts

## The `is_train` flag

Every method in the library takes an `is_train` flag. This is the entire leakage-prevention mechanism, so it's worth understanding once:

| `is_train=True` (fit + transform) | `is_train=False` (transform only) |
|---|---|
| Analyzes the dataframe's statistics (skew, missingness, cardinality, etc.) | Skips analysis entirely — uses whatever was learned during the last `is_train=True` call |
| Saves the resulting rules onto the handler instance | Applies the saved rules as-is, with no recalculation |
| Must be called **first**, on the training split | Must be called **after** a training pass on the *same* handler/pipeline instance |

!!! warning "Fit before transform"
    Calling a handler with `is_train=False` before it has ever been fit with `is_train=True` raises an `AttributeError` by design — there is nothing learned yet to apply.

## How it decides: numerical columns

Handled by [`HandleNumerical`](api/handle-numerical.md).

| Step | Condition | Action |
|---|---|---|
| **Imputation** | Column has > 40% missing values | Column is dropped |
| | Uniqueness < 1% of rows | Treated as categorical-like → mode imputation |
| | Otherwise, `\|skew\| < 0.5` | Mean imputation |
| | Otherwise, `\|skew\| ≥ 0.5` | Median imputation |
| **Outliers** | Outlier share (IQR method) ≤ 3% | Clipped to IQR bounds |
| | Outlier share > 3%, moderate tail or negative values present | Winsorized at the 5th / 95th percentile |
| | Outlier share > 3%, long tail, contains zeros | Square-root transform |
| | Outlier share > 3%, long tail, strictly positive | `log1p` transform |
| **Scaling** | Sparsity ≥ 50% zeros | `MaxAbsScaler` |
| | Skew > 1.0 and non-negative | `log1p` transform + `StandardScaler` |
| | Outlier ratio ≥ 5% | `RobustScaler` |
| | Otherwise | `StandardScaler` |

## How it decides: categorical columns

Handled by [`HandleCategorical`](api/handle-categorical.md).

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

## Why this design prevents leakage

Because every threshold is computed **only** during an `is_train=True` call and then frozen onto the handler (`impute_values_`, `outlier_rules_`, `scalers_`, `encode_types_`, ...), the test/inference pass can never see or influence a decision. It can only replay decisions the training data already made. This is the same `.fit()` / `.transform()` contract scikit-learn transformers use — see [Inspecting Learned Rules](api/inspecting-rules.md) to view those frozen decisions directly.
