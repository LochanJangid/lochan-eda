# Contributing

Issues and pull requests are welcome.

## Running the test suite

```bash
git clone https://github.com/LochanJangid/lochan-eda.git
cd lochan-eda
pip install -e .
pip install pytest
pytest -v
```

Please run `pytest -v` before submitting a pull request to verify structural integrity across train/test splits — the suite covers leakage regressions, schema consistency between train and test output, and output sanity (no NaNs, all-numeric columns).

## Building the documentation locally

This site is built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

```bash
pip install mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000` to preview changes live as you edit files under `docs/`.

## Known gaps (good first issues)

- `exclude` is not forwarded through `full_handler()` on either handler
- Missing-value (40%) and outlier (3%) thresholds are hardcoded rather than configurable
- No verbose/logging mode that explains each column's decision at runtime
