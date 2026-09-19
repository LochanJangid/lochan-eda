# Missing

`Missing` provides a focused visualization of missing values across columns.

It is normally accessed through `Profiler`:

```python
from lochan_eda import Profiler

profile = Profiler(df)
profile.missing.plot()
```

## `plot()`

```python
fig = profile.missing.plot()
```

The plot shows each column's percentage of missing values, ordered from lower to higher missingness in the chart.

The figure is saved as:

```text
missing_plot.png
```

If the DataFrame is empty, the method returns `None`. If there are no missing values, the returned figure displays `No Missing Value Found`.
