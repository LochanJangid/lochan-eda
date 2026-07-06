# `HandleCategorical`

Use this directly if you want to process only categorical/object/string columns. See [Concepts → How it decides: categorical columns](../concepts.md#how-it-decides-categorical-columns) for the full decision logic.

```python
from lochan_eda import HandleCategorical

cat_handler = HandleCategorical(X_train)   # auto-selects category/object/string columns
```

## Methods

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `cat_imputer(is_train=True, exclude=None)` | `exclude`: column name or list to skip | `pandas.DataFrame` | Fills missing values with mode or `"Unknown"` depending on missingness %. |
| `rare_manager(is_train=True, threshold=0.05, exclude=None)` | `threshold`: min frequency to avoid being grouped into `"Other"` | `pandas.DataFrame` | Groups low-frequency categories. |
| `encoder(is_train=True, target=None, exclude=None)` | `target`: series for `TargetEncoder` on high-cardinality columns | `pandas.DataFrame` | Encodes categories to numbers. |
| `full_handler(target=None, is_train=True)` | `target`: passed through to `encoder` | `pandas.DataFrame` | Runs `cat_imputer` → `rare_manager` → `encoder` in sequence. |

!!! info "Excluding a column drops it from `cat_df`"
    Unlike `HandleNumerical`, passing `exclude` to a `HandleCategorical` method removes that column from the handler's working dataframe entirely for that call, rather than passing it through untouched. Keep this in mind if you plan to exclude and later re-attach a categorical column manually.

## Train → test usage

```python
# TRAIN
X_train_cat = cat_handler.full_handler(target=y_train, is_train=True)

# TEST — swap in test data, keep the same handler instance
cat_handler.cat_df = X_test.select_dtypes(include=["category", "object", "string"]).copy()
X_test_cat = cat_handler.full_handler(target=None, is_train=False)
```

!!! tip "Unseen categories are handled gracefully"
    A binary-mapped column maps an unseen category to `-1`, a one-hot column gets all zeros, and a frequency-encoded column gets `0`. The pipeline will not crash on new categories at inference time.

## Learned attributes

| Attribute | Description |
|---|---|
| `drop_cols_` | Categorical columns dropped for exceeding the 40% missing-value threshold. |
| `impute_modes_` | `{column: mode_value}` used for imputation. |
| `rare_cats_` | `{column: [rare category values]}` grouped into `"Other"`. |
| `encode_types_` | `{column: encoding strategy}` — one of `"binary"`, `"ohe"`, `"te"`, `"freq"`. |
| `binary_maps_` | `{column: {value: 0/1}}` for binary-encoded columns. |
| `ohe_columns_` | `{column: [one-hot column names]}` learned at train time, used to align test output. |
| `freq_maps_` | `{column: {value: frequency}}` for frequency-encoded columns. |
| `target_encoders_` | `{column: fitted sklearn TargetEncoder}` for target-encoded columns. |

See [Inspecting Learned Rules](inspecting-rules.md) for full usage examples.
