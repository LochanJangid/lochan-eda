# Numerical

`Numerical` provides numerical EDA and numerical preprocessing.

```python
from lochan_eda import Numerical

numerical = Numerical(profiler_df=df)
```

In normal exploratory workflows, use it through `Profiler`:

```python
profile.numerical
```

## EDA API

### `summary()`

```python
summary = profile.numerical.summary()
```

Returns a DataFrame from numerical `describe()` output with additional columns for:

- standard deviation
- missing-value percentage
- skew
- zero percentage
- outlier percentage

If there are no numerical columns, an empty DataFrame is returned.

### `plot(columns=None)`

```python
profile.numerical.plot()
```

Restrict the plot to selected numerical columns:

```python
profile.numerical.plot(columns=["age", "income"])
```

For each selected column the figure contains:

1. Distribution histogram
2. Box plot
3. Normal Q-Q plot

The figure is also saved as `numerical_plots.png`.

## Preprocessing API

### `fit(data, exclude=None)`

```python
numerical.fit(X_train)
```

Learns numerical preprocessing rules from the supplied training DataFrame.

### `transform(data, exclude=None)`

```python
X_processed = numerical.transform(X_test)
```

Applies the learned preprocessing rules. `fit()` must be called first.

### `imputer(learn=False)`

Internal preprocessing stage for missing numerical values. During learning it determines columns to drop and values used for imputation.

### `outlier_manager(learn=False)`

Internal preprocessing stage for outlier handling. Depending on learned data behavior, it can learn clipping, square-root, or `log1p` transformations.

### `scaler(learn=False)`

Internal preprocessing stage for feature scaling. The implementation can select `MaxAbsScaler`, `StandardScaler`, or `RobustScaler`, and can learn a `log1p` transformation for strongly right-skewed non-negative data.
