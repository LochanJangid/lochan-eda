# lochan-eda

[![Tests](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml/badge.svg)](https://github.com/LochanJangid/lochan-eda/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![Python versions](https://img.shields.io/pypi/pyversions/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LochanJangid/lochan-eda/blob/main/LICENSE)

**An intelligent, stateful preprocessing pipeline that decides *how* to clean your data, prevents data leakage, and prepares raw datasets for production machine learning models.**

Most EDA helpers apply the same fixed rule to every column. `lochan-eda` acts as an automated data engineer — it inspects each column's statistical shape (missingness, skew, cardinality, sparsity, outlier ratio) and dynamically picks the right imputation, outlier-handling, and scaling strategy for it.

The core idea is **stateful memory**: the pipeline *learns* its rules on training data, saves them, and *blindly re-applies* those exact rules to test/inference data. No re-fitting, no leakage, no schema mismatches at deployment.

```bash
pip install lochan-eda
```

## Why lochan-eda

A real preprocessing pass usually means writing the same boilerplate decision tree by hand, every time: *Is this column highly skewed? Does it have a long tail? Is this a high-cardinality string?*

`lochan-eda` automates that decision tree, and it is **not a black box** — every rule it applies is documented in this site and inspectable at runtime, so you always know why a column was scaled with `RobustScaler` instead of `StandardScaler`, or why a category got grouped into `"Other"`. See [Inspecting Learned Rules](api/inspecting-rules.md).

## Features

| Feature | Description |
|---|---|
| **Leakage-free architecture** | Strict `.fit()` / `.transform()` separation via an `is_train` flag — rules are learned once on training data and only ever applied (never re-learned) on test data. |
| **Adaptive imputation** | Chooses mean, median, or mode per column based on skewness and cardinality. |
| **Adaptive outlier handling** | Clips, winsorizes, or transforms tails depending on severity — never drops rows, so `X` and `y` shapes always stay aligned. |
| **Adaptive scaling** | Picks `StandardScaler`, `RobustScaler`, or `MaxAbsScaler` based on sparsity, skew, and outlier ratio. |
| **Smart categorical encoding** | Routes to binary mapping, one-hot encoding, frequency encoding, or `TargetEncoder` based on cardinality. |
| **Rare-category grouping** | Collapses low-frequency categories into `"Other"` to prevent high-dimensional noise. |
| **Schema-safe inference** | Test-time output is automatically reindexed to match the training schema — missing columns are filled with 0, unseen columns are dropped. |

## Requirements

- Python ≥ 3.8
- `scikit-learn >= 1.3.0` (for native `TargetEncoder` support)
- `pandas`, `numpy`, `matplotlib`, `seaborn` (installed automatically as dependencies)

## Next steps


- **[Quickstart](quickstart.md)** — install the package and run your first pipeline in under a minute
- **[Core Concepts](concepts.md)** — understand the `is_train` flag and how every decision is made
- **[API Reference](api/automated-eda.md)** — full parameter tables for every class and method
- **[Full Example](examples.md)** — a copy-pasteable, runnable end-to-end script
