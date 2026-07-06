# `AutomatedEDA`

The top-level orchestrator. Use this unless you specifically need to control the numerical or categorical pipeline in isolation.

```python
from lochan_eda import AutomatedEDA

eda = AutomatedEDA()
```

## `run_pipeline(df, target=None, is_train=True)`

Runs the full numerical + categorical pipeline end-to-end and returns a single, concatenated, fully numeric dataframe.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pandas.DataFrame` | required | The dataframe to process. |
| `target` | `pandas.Series` or `None` | `None` | Required if you want high-cardinality categorical columns to use `TargetEncoder` instead of frequency encoding. Only used when `is_train=True`. |
| `is_train` | `bool` | `True` | `True` to learn rules on this dataframe (training split). `False` to apply previously learned rules (test/inference split). |

**Returns**

`pandas.DataFrame` — fully numeric, `NaN`-free. On `is_train=False`, columns are automatically reindexed to match the training output (missing columns filled with `0`, extra columns dropped).

**Example**

```python
X_train_clean = eda.run_pipeline(df=X_train, target=y_train, is_train=True)
X_test_clean  = eda.run_pipeline(df=X_test, is_train=False)
```

## Attributes

| Attribute | Description |
|---|---|
| `num_handler` | The internal [`HandleNumerical`](handle-numerical.md) instance, created on the first `is_train=True` call. |
| `cat_handler` | The internal [`HandleCategorical`](handle-categorical.md) instance, created on the first `is_train=True` call. |
| `final_columns_` | The full training-time output column order, used to align test-time output. |

!!! danger "Calling test mode before training"
    `eda.num_handler` and `eda.cat_handler` are `None` until the first `is_train=True` call. Calling `run_pipeline(..., is_train=False)` before that raises `AttributeError` by design.
