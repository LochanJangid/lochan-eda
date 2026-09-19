# Ready to use

This page is the shortest path from installation to useful output.

## Install

```bash
pip install lochan-eda
```

## 1. Prepare a dataset for machine learning

For a DataFrame with a target column:

```python
from lochan_eda import AutomatedEDA

eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)
```

By default, `prepare()` performs an 80/20 train-test split and learns preprocessing from the training data before transforming both splits.

### Keep a column untouched

Use `exclude` for columns that should not be processed:

```python
X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target",
    exclude=["customer_id"]
)
```

### Use a separate target Series

```python
y = df["target"]
X = df.drop(columns="target")

X_train, X_test, y_train, y_test = eda.prepare(
    X,
    target=y
)
```

### Do not split

```python
X_processed, y = eda.prepare(
    df,
    target="target",
    split=False
)
```

If there is no target:

```python
X_processed = eda.prepare(
    df,
    split=False
)
```

## 2. Profile a dataset

```python
from lochan_eda import Profiler

profile = Profiler(df, target="target")
```

Get a high-level overview:

```python
profile.overview()
```

## 3. Inspect numerical columns

```python
profile.numerical.summary()
profile.numerical.plot()
```

The numerical plot contains a distribution, box plot, and Q-Q plot for each numerical column.

## 4. Inspect categorical columns

```python
profile.categorical.summary()
profile.categorical.plot()
```

To limit the number of categories shown per column:

```python
profile.categorical.plot(top_n=10)
```

## 5. Inspect missing values

```python
profile.missing.plot()
```

The plot is saved as `missing_plot.png` in the current working directory.

## 6. Generate a PDF report

```python
profile.report.save("report.pdf")
```

The report contains a dataset overview and the report components implemented by the package.

## A practical first workflow

```python
import pandas as pd
from lochan_eda import Profiler, AutomatedEDA

# Load
 df = pd.read_csv("data.csv")

# Explore
profile = Profiler(df, target="target")
profile.overview()
profile.numerical.summary()
profile.categorical.summary()
profile.missing.plot()
profile.report.save("eda_report.pdf")

# Prepare
eda = AutomatedEDA()
X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)
```
