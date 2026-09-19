# lochan-eda

**A practical toolkit for exploring and preparing tabular data for machine learning.**

`lochan-eda` provides two simple workflows:

- **Explore** a dataset with summaries, plots, missing-value analysis, and a PDF report.
- **Prepare** tabular data with automatic numerical and categorical preprocessing.

---

# Ready to Use

## Installation

```bash
pip install lochan-eda
```

## 1. Automatically Prepare Data for Machine Learning

Use `AutomatedEDA` when you want to prepare a tabular dataset before training a machine-learning model.

```python
import pandas as pd
from lochan_eda import AutomatedEDA

# Load your data
df = pd.read_csv("data.csv")

eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)
```

By default, `prepare()`:

- separates the target column
- creates an 80/20 train-test split
- learns preprocessing from the training data
- applies the same learned preprocessing to the test data
- processes numerical and categorical columns separately

You can then pass the processed data directly to your model.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### Exclude columns

Keep columns such as IDs out of automatic preprocessing:

```python
eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target",
    exclude=["customer_id"]
)
```

### Use your own target Series

```python
X = df.drop(columns="target")
y = df["target"]

eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    X,
    target=y
)
```

### Prepare without a train-test split

```python
X, y = eda.prepare(
    df,
    target="target",
    split=False
)
```

If there is no target, `prepare()` can also process feature data only:

```python
X = eda.prepare(
    df,
    split=False
)
```

---

## 2. Profile a Dataset

Use `Profiler` when you want to understand a dataset before modeling.

```python
import pandas as pd
from lochan_eda import Profiler

df = pd.read_csv("data.csv")

profile = Profiler(df, target="target")
```

### Dataset overview

```python
profile.overview()
```

This returns information about:

- rows and columns
- memory usage
- duplicate rows
- duplicate percentage
- missing cells
- missing percentage
- numerical columns
- categorical columns
- datetime columns

### Numerical analysis

```python
profile.numerical.summary()
profile.numerical.plot()
```

### Categorical analysis

```python
profile.categorical.summary()
profile.categorical.plot()
```

### Missing-value analysis

```python
profile.missing.plot()
```

### Generate a PDF report

```python
profile.report.save("eda_report.pdf")
```

The report contains dataset overview information, numerical and categorical summaries, and generated analysis plots.

---

## 3. Analyze Only Selected Columns

Numerical and categorical plots can be limited to selected columns.

```python
profile.numerical.plot(
    columns=["age", "income"]
)
```

```python
profile.categorical.plot(
    columns=["city", "education"],
    top_n=10
)
```

`top_n` keeps the most frequent categories visible and groups the remaining categories as `Others`.

---

# API Overview

## `AutomatedEDA`

```python
from lochan_eda import AutomatedEDA
```

The main interface for automatic tabular-data preprocessing.

### `AutomatedEDA()`

```python
AutomatedEDA()
```

Creates an automatic preprocessing object.

### `prepare()`

```python
prepare(
    X,
    target=None,
    exclude=None,
    split=True,
    test_size=0.2,
    random_state=42,
    stratify=None
)
```

The high-level method for preparing data.

| Parameter | Description |
|---|---|
| `X` | Input `pandas.DataFrame`. |
| `target` | Target column name or `pandas.Series`. |
| `exclude` | Column name or list of columns to exclude from preprocessing. |
| `split` | Whether to create train/test data. Default: `True`. |
| `test_size` | Proportion used for the test set. Default: `0.2`. |
| `random_state` | Random state for reproducibility. Default: `42`. |
| `stratify` | Values used for stratified splitting. |

### Return values

The returned values depend on `target` and `split`:

| `target` | `split` | Returns |
|---|---|---|
| Not provided | `False` | `X` |
| Not provided | `True` | `X_train, X_test` |
| Provided | `False` | `X, y` |
| Provided | `True` | `X_train, X_test, y_train, y_test` |

### `fit()`

```python
fit(X, y=None)
```

Learns preprocessing rules from the supplied data.

`fit()` separates numerical and categorical columns and learns the required transformations for each type.

### `transform()`

```python
transform(X, y=None)
```

Applies preprocessing rules learned by `fit()`.

`transform()` must be called after `fit()`.

---

# `Profiler`

```python
from lochan_eda import Profiler
```

Provides dataset-level exploratory analysis.

### `Profiler()`

```python
Profiler(df, target=None)
```

| Parameter | Description |
|---|---|
| `df` | Input `pandas.DataFrame`. |
| `target` | Target column name. |

When `target` is a column name, the target is separated from the profiling data.

### `overview()`

```python
profile.overview()
```

Prints and returns a dictionary containing:

- `rows`
- `columns`
- `memory_usage`
- `duplicate_rows`
- `duplicate_percentage`
- `missing_cells`
- `missing_percentage`
- `numerical_columns`
- `categorical_columns`
- `datetime_columns`

---

# `Numerical`

```python
from lochan_eda import Numerical
```

Handles numerical-column profiling and preprocessing.

### `Numerical()`

```python
Numerical(profiler_df=None)
```

`profiler_df` is used by the profiling methods such as `summary()` and `plot()`.

### `fit()`

```python
fit(data, exclude=None)
```

Learns numerical preprocessing rules from the supplied data.

The numerical workflow can learn rules for:

- missing-value imputation
- column removal based on missingness
- outlier handling
- scaling

### `transform()`

```python
transform(data, exclude=None)
```

Applies the rules learned by `fit()`.

### `summary()`

```python
profile.numerical.summary()
```

Returns a DataFrame containing descriptive numerical statistics together with missing-value, skewness, zero, and outlier information.

### `plot()`

```python
profile.numerical.plot(columns=None)
```

Creates numerical analysis plots for each selected numerical column:

- distribution
- box plot
- Q-Q plot

The figure is also saved as `numerical_plots.png`.

### `imputer()`

```python
imputer(learn=False)
```

Internal numerical preprocessing step for learning or applying missing-value handling.

### `outlier_manager()`

```python
outlier_manager(learn=False)
```

Internal numerical preprocessing step for learning or applying outlier rules.

### `scaler()`

```python
scaler(learn=False)
```

Internal numerical preprocessing step for learning or applying feature scaling.

> `imputer()`, `outlier_manager()`, and `scaler()` are lower-level methods. For normal usage, prefer `AutomatedEDA` or `Numerical.fit()` / `Numerical.transform()`.

---

# `Categorical`

```python
from lochan_eda import Categorical
```

Handles categorical-column profiling and preprocessing.

### `Categorical()`

```python
Categorical(profiler_df=None)
```

`profiler_df` is used by the profiling methods.

### `fit()`

```python
fit(data, exclude=None)
```

Learns categorical preprocessing rules.

The categorical workflow can learn rules for:

- missing-value handling
- rare-category handling
- binary encoding
- one-hot encoding
- frequency encoding

### `transform()`

```python
transform(data, exclude=None)
```

Applies the categorical preprocessing rules learned by `fit()`.

### `summary()`

```python
profile.categorical.summary()
```

Returns descriptive statistics for categorical columns.

### `plot()`

```python
profile.categorical.plot(columns=None, top_n=10)
```

Creates category-distribution bar charts.

The figure is also saved as `categorical_plots.png`.

### `imputer()`

```python
imputer(learn=False)
```

Internal categorical preprocessing step for learning or applying missing-value handling.

### `rare_manager()`

```python
rare_manager(learn=False)
```

Internal preprocessing step that learns or applies rare-category grouping.

### `encoder()`

```python
encoder(learn=False)
```

Internal preprocessing step that selects and applies categorical encoding based on category cardinality.

> `imputer()`, `rare_manager()`, and `encoder()` are lower-level methods. For normal usage, prefer `AutomatedEDA` or `Categorical.fit()` / `Categorical.transform()`.

---

# `Missing`

```python
from lochan_eda.missing import Missing
```

Provides missing-value visualization.

### `Missing()`

```python
Missing(profiler_df=None)
```

Creates a missing-value analysis object.

### `plot()`

```python
profile.missing.plot()
```

Creates a horizontal bar chart showing missing-value percentages by column.

The figure is also saved as `missing_plot.png`.

---

# `Report`

```python
from lochan_eda.report import Report
```

Generates an EDA PDF report from a `Profiler` instance.

### `Report()`

```python
Report(profiler)
```

### `save()`

```python
profile.report.save("report.pdf")
```

Generates and saves the exploratory data analysis report.

Default path:

```python
profile.report.save()
```

which creates:

```text
report.pdf
```

---

# Utility Functions

The package also contains helper functions in `lochan_eda.utils`.

### `get_iqr_bounds()`

```python
from lochan_eda.utils import get_iqr_bounds

lower, upper = get_iqr_bounds(series)
```

Returns lower and upper bounds calculated using the IQR method.

### `get_active_cols()`

```python
from lochan_eda.utils import get_active_cols

columns = get_active_cols(all_cols, exclude=["id"])
```

Returns columns after removing excluded columns.

### `is_matching()`

```python
from lochan_eda.utils import is_matching

is_matching(fit_df, transform_df)
```

Checks whether the two DataFrames contain matching column sets.

---

# Typical Workflow

A common machine-learning workflow with `lochan-eda` looks like this:

```python
import pandas as pd
from lochan_eda import AutomatedEDA
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Prepare data
eda = AutomatedEDA()
X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)

# 3. Train your model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)
```

For exploration before modeling:

```python
from lochan_eda import Profiler

profile = Profiler(df, target="target")

profile.overview()
profile.numerical.summary()
profile.categorical.summary()
profile.missing.plot()
profile.report.save("eda_report.pdf")
```

---

# Package

- **Package:** `lochan-eda`
- **Author:** Lochan Jangid
- **Version:** `0.2.0`

