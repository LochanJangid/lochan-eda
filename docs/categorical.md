# Categorical

`Categorical` provides categorical EDA and categorical preprocessing.

```python
from lochan_eda import Categorical

categorical = Categorical(profiler_df=df)
```

In normal exploratory workflows, use it through `Profiler`:

```python
profile.categorical
```

## EDA API

### `summary()`

```python
summary = profile.categorical.summary()
```

Returns `DataFrame.describe()` output for object, category, and string columns.

If there are no categorical columns, an empty DataFrame is returned.

### `plot(columns=None, top_n=10)`

```python
profile.categorical.plot()
```

Select specific categorical columns:

```python
profile.categorical.plot(
    columns=["city", "segment"],
    top_n=8,
)
```

Categories beyond `top_n` are grouped as `Others` in the plot.

The figure is saved as `categorical_plots.png`.

## Preprocessing API

### `fit(data, exclude=None)`

```python
categorical.fit(X_train)
```

Learns missing-value, rare-category, and encoding rules from the training data.

### `transform(data, exclude=None)`

```python
X_processed = categorical.transform(X_test)
```

Applies the rules learned during `fit()`.

### `imputer(learn=False)`

Learns which highly-missing categorical columns should be dropped and which missing values should be filled.

### `rare_manager(learn=False)`

Learns low-frequency categories and groups them into `Other` during transformation.

### `encoder(learn=False)`

Chooses an encoding strategy from the number of unique values:

- **Binary** for up to 2 unique values
- **One-hot** for up to 10 unique values
- **Frequency encoding** above 10 unique values
