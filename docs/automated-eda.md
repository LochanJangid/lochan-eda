# AutomatedEDA

`AutomatedEDA` is the high-level preprocessing interface. It separates numerical and categorical columns, learns preprocessing rules from training data, and applies those rules to transformed data.

```python
from lochan_eda import AutomatedEDA

eda = AutomatedEDA()
```

## `prepare()`

```python
eda.prepare(
    X,
    target=None,
    exclude=None,
    split=True,
    test_size=0.2,
    random_state=42,
    stratify=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `X` | `pd.DataFrame` | required | Feature DataFrame, or a DataFrame containing the target column. |
| `target` | `str \| pd.Series \| None` | `None` | Target column name or an independent target Series. |
| `exclude` | list | `None` | Columns to leave out of preprocessing. |
| `split` | `bool` | `True` | Whether to create train/test splits. |
| `test_size` | `float` | `0.2` | Fraction assigned to the test set when splitting. |
| `random_state` | `int` | `42` | Random state passed to `train_test_split`. |
| `stratify` | array-like | `None` | Stratification values passed to `train_test_split` when a target is present. |

### Return shapes

The return value depends on `target` and `split`:

| `target` | `split` | Returns |
|---|---|---|
| `None` | `False` | `X` |
| `None` | `True` | `X_train, X_test` |
| provided | `False` | `X, y` |
| provided | `True` | `X_train, X_test, y_train, y_test` |

### Typical use

```python
X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target",
    test_size=0.2,
    random_state=42,
)
```

## `fit()`

```python
eda.fit(X, y=None)
```

Learns preprocessing rules separately for numerical and categorical columns.

`fit()` returns `None`.

## `transform()`

```python
X_processed = eda.transform(X)
```

With a target:

```python
X_processed, y = eda.transform(X, y)
```

The method applies the rules learned by `fit()`.

## Preprocessing behavior

### Numerical data

`Numerical` can learn rules for:

- columns with high missingness
- missing-value imputation
- outlier treatment
- scaling
- logarithmic transformation for strongly right-skewed non-negative features

### Categorical data

`Categorical` can learn rules for:

- columns with high missingness
- missing-value imputation
- rare-category grouping
- binary encoding
- one-hot encoding
- frequency encoding

The exact rule is learned from the training data and reused during transformation.
