# lochan-eda

**Exploratory data analysis and behaviour-driven preprocessing for tabular machine learning.**

[![PyPI version](https://img.shields.io/pypi/v/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![Python versions](https://img.shields.io/pypi/pyversions/lochan-eda.svg)](https://pypi.org/project/lochan-eda/)
[![License](https://img.shields.io/pypi/l/lochan-eda.svg)](https://github.com/LochanJangid/lochan-eda)
[![GitHub](https://img.shields.io/badge/source-GitHub-black.svg)](https://github.com/LochanJangid/lochan-eda)

`lochan-eda` is a Python package for exploratory data analysis and preprocessing of tabular datasets.

It provides reusable components for inspecting numerical and categorical features, analysing missing values, handling common preprocessing tasks, and constructing a consistent train/test preprocessing workflow.

The package is designed around a simple principle:

> **Understand the behaviour of the data before choosing how to preprocess it.**

---

## Installation

Install the latest release from PyPI:

```bash
pip install lochan-eda
```

For development:

```bash
git clone https://github.com/LochanJangid/lochan-eda.git
cd lochan-eda

pip install -e .
```

---

## Quick Start

### Automated workflow

```python
import pandas as pd

from lochan_eda import AutomatedEDA

df = pd.read_csv("data.csv")

eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)
```

`prepare()` provides a high-level interface for the common tabular preprocessing workflow.

---

## Dataset Profiling

For explicit dataset inspection, use `Profiler`:

```python
from lochan_eda import Profiler

profile = Profiler(df)

profile.overview()
```

The profiler provides access to dataset-level analysis and the numerical and categorical analysis components.

---

## Numerical Analysis

Numerical features are handled through the `Numerical` component.

```python
from lochan_eda import Numerical

numerical = Numerical(df)

numerical.summary()
numerical.plot()
```

The numerical interface includes operations for:

* Missing-value handling
* Outlier management
* Scaling
* Statistical summaries
* Visualization

Available methods include:

```text
imputer()
outlier_manager()
scaler()
summary()
plot()
```

---

## Categorical Analysis

Categorical features are handled separately through `Categorical`.

```python
from lochan_eda import Categorical

categorical = Categorical(df)

categorical.summary()
categorical.plot()
```

The categorical interface provides operations for:

* Missing-value handling
* Rare-category management
* Encoding
* Statistical summaries
* Visualization

Available methods include:

```text
imputer()
rare_manager()
encoder()
summary()
plot()
```

---

## Missing Values

Missing-value analysis is available independently:

```python
# Example interface

missing.plot()
```

This allows missing-value patterns to be inspected before deciding how they should be handled.

---

# Preprocessing Philosophy

A common preprocessing workflow can be written directly with scikit-learn:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_columns
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ]),
        categorical_columns
    )
])
```

This is a valid and useful approach.

The problem is not the pipeline itself.

The problem is deciding whether the selected transformations are appropriate for the dataset.

For example:

```text
median imputation
       ↓
Why?

standard scaling
       ↓
Why?

most-frequent imputation
       ↓
Why?

one-hot encoding
       ↓
Why?
```

`lochan-eda` provides a layer for analysing the dataset before those decisions are applied.

```text
                  Dataset
                     │
                     ▼
              ┌─────────────┐
              │   Profile   │
              └──────┬──────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Numerical             Categorical
     behaviour               behaviour
          │                     │
          └──────────┬──────────┘
                     ▼
             Preprocessing
               workflow
                     │
                     ▼
              Model-ready data
```

The package therefore complements preprocessing libraries rather than attempting to replace them.

---

# Train / Test Workflow

Preprocessing should be learned from training data and then reused when transforming other data.

`AutomatedEDA` supports this separation:

```python
eda.fit(X_train)

X_train = eda.transform(X_train)
X_test = eda.transform(X_test)
```

This keeps the fitting of preprocessing separate from its application.

For the common workflow, `prepare()` provides a single entry point:

```python
X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)
```

---

# API

## `AutomatedEDA`

High-level interface for the complete preprocessing workflow.

```text
prepare()
fit()
transform()
```

---

## `Profiler`

Dataset-level analysis interface.

```text
overview()
numerical
categorical
```

---

## `Numerical`

Numerical feature analysis and preprocessing.

```text
imputer()
outlier_manager()
scaler()
summary()
plot()
```

---

## `Categorical`

Categorical feature analysis and preprocessing.

```text
imputer()
rare_manager()
encoder()
summary()
plot()
```

---

## `Missing`

Missing-value visualization.

```text
plot()
```

---

## `Report`

Generate a shareable analysis report.

```text
save()
```

---

# Design

`lochan-eda` separates the workflow into two levels.

### High-level API

Use `AutomatedEDA` when the goal is to move efficiently from a DataFrame to model-ready data.

```python
eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)
```

### Component-level API

Use `Profiler`, `Numerical`, and `Categorical` when explicit control or inspection is required.

```python
profile = Profiler(df)

profile.overview()

profile.numerical
profile.categorical
```

This keeps the common workflow simple without removing access to the underlying analysis components.

---

# Package Structure

The project is organized around the major stages of tabular data analysis and preprocessing:

```text
lochan_eda/
│
├── automated_eda/
├── profiler/
├── numerical/
├── categorical/
├── missing/
└── report/
```

The internal implementation may evolve independently from the public API.

---

# Dependencies

`lochan-eda` is built around the Python data-science ecosystem and integrates with commonly used tools for tabular machine learning.

The package is intended to work alongside libraries such as:

* pandas
* NumPy
* scikit-learn
* matplotlib

Rather than replacing these libraries, `lochan-eda` provides a higher-level workflow for analysis and preprocessing.

---

# Example

A complete workflow can be as small as:

```python
import pandas as pd

from lochan_eda import Profiler
from lochan_eda import AutomatedEDA

df = pd.read_csv("data.csv")

# Inspect
profile = Profiler(df)
profile.overview()

# Prepare
eda = AutomatedEDA()

X_train, X_test, y_train, y_test = eda.prepare(
    df,
    target="target"
)

# Continue with model training
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Development

Clone the repository:

```bash
git clone https://github.com/LochanJangid/lochan-eda.git
cd lochan-eda
```

Install in editable mode:

```bash
pip install -e .
```

If you are developing new functionality, install the development dependencies defined by the project.

Run the test suite with the project's configured test command.

---

# Contributing

Contributions are welcome.

Before submitting a change:

1. Keep the public API consistent with the existing design.
2. Add or update tests for behavioural changes.
3. Keep preprocessing decisions explicit and reproducible.
4. Avoid introducing unnecessary dependencies.
5. Document new public functionality.

For larger changes, open an issue first to discuss the proposed design.

---

# License

See the repository's license file for the applicable license.

---

# Links

* **PyPI:** https://pypi.org/project/lochan-eda/
* **Source repository:** https://github.com/LochanJangid/lochan-eda/

---

## Status

`lochan-eda` is an actively developed project. The API may evolve as additional preprocessing and analysis capabilities are introduced.
