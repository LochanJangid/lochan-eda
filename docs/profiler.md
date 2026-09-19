# Profiler

`Profiler` is the main entry point for exploratory analysis.

```python
from lochan_eda import Profiler

profile = Profiler(df, target="target")
```

If `target` is a column name, that column is separated from the DataFrame used by the profiler.

The profiler exposes:

```python
profile.numerical
profile.categorical
profile.missing
profile.report
```

## `overview()`

```python
result = profile.overview()
```

Prints and returns dataset-level statistics.

The returned dictionary contains:

```text
rows
columns
memory_usage
duplicate_rows
duplicate_percentage
missing_cells
missing_percentage
numerical_columns
categorical_columns
datetime_columns
```

Example:

```python
overview = profile.overview()
print(overview["missing_percentage"])
```

## Numerical tools

```python
profile.numerical.summary()
profile.numerical.plot()
```

See [Numerical](numerical.md).

## Categorical tools

```python
profile.categorical.summary()
profile.categorical.plot()
```

See [Categorical](categorical.md).

## Missing-value tools

```python
profile.missing.plot()
```

See [Missing](missing.md).

## PDF report

```python
profile.report.save("report.pdf")
```

See [Report](report.md).
