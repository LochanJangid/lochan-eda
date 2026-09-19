# lochan-eda documentation

`lochan-eda` is a Python toolkit for two closely related jobs:

1. **Explore** a tabular dataset quickly with summaries, plots, missing-value analysis, and a PDF report.
2. **Prepare** numerical and categorical features for machine-learning workflows with learned preprocessing.

The documentation is organized around the way you actually use the package, not around its source-code file tree.

## Start here

- [Ready to use](ready-to-use.md) — copy-paste workflows for the common cases.
- [AutomatedEDA](automated-eda.md) — automatic train/test preparation and preprocessing.
- [Profiler](profiler.md) — inspect a dataset and access the analysis tools.
- [Numerical](numerical.md) — numerical summaries, plots, and numerical preprocessing.
- [Categorical](categorical.md) — categorical summaries, plots, and categorical preprocessing.
- [Missing](missing.md) — visualize missing values.
- [Report](report.md) — generate a PDF EDA report.

## Package API

The package currently exposes these classes from the top-level namespace:

```python
from lochan_eda import AutomatedEDA, Profiler, Numerical, Categorical
```

`Missing` and `Report` are used by `Profiler` but are not currently exported from `lochan_eda.__init__`.
