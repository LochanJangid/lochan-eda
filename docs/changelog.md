# Changelog

This page tracks notable changes to `lochan-eda` across published PyPI releases.

## 0.1.1 (current)

- `AutomatedEDA`, `HandleNumerical`, and `HandleCategorical` public API as documented on this site.
- Leakage-free `TargetEncoder` handling for high-cardinality categorical columns (encoder is fit only during `is_train=True`).
- Schema-safe test-time output: columns reindexed to the training schema, missing columns filled with `0`.

## 0.0.1 – 0.1.0

Earlier releases; detailed per-version notes weren't tracked at the time. See the [GitHub tags](https://github.com/LochanJangid/lochan-eda/tags) and [commit history](https://github.com/LochanJangid/lochan-eda/commits/main) for the full history.

## Planned

See [Contributing → Known gaps](contributing.md#known-gaps-good-first-issues) for the current roadmap:

- Forward `exclude` through `full_handler()` on both handlers
- Configurable missing-value and outlier thresholds
- Optional verbose/logging mode
- Datetime column handling

---