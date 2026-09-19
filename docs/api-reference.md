# API reference

This page summarizes the public surface of the current package implementation.

## Top-level imports

```python
from lochan_eda import AutomatedEDA, Profiler, Numerical, Categorical
```

## `AutomatedEDA`

- `prepare(X, target=None, exclude=None, split=True, test_size=0.2, random_state=42, stratify=None)`
- `fit(X, y=None)`
- `transform(X, y=None)`

See [AutomatedEDA](automated-eda.md).

## `Profiler`

- `overview()`
- `numerical` → `Numerical`
- `categorical` → `Categorical`
- `missing` → `Missing`
- `report` → `Report`

See [Profiler](profiler.md).

## `Numerical`

### Analysis

- `summary()`
- `plot(columns=None)`

### Preprocessing

- `fit(data, exclude=None)`
- `transform(data, exclude=None)`
- `imputer(learn=False)`
- `outlier_manager(learn=False)`
- `scaler(learn=False)`

See [Numerical](numerical.md).

## `Categorical`

### Analysis

- `summary()`
- `plot(columns=None, top_n=10)`

### Preprocessing

- `fit(data, exclude=None)`
- `transform(data, exclude=None)`
- `imputer(learn=False)`
- `rare_manager(learn=False)`
- `encoder(learn=False)`

See [Categorical](categorical.md).

## `Missing`

- `plot()`

See [Missing](missing.md).

## `Report`

- `save(filepath="report.pdf")`

See [Report](report.md).
