# FAQ / Troubleshooting

**I got `AttributeError` when calling `run_pipeline(..., is_train=False)`.**

You must call the pipeline once with `is_train=True` before ever calling it with `is_train=False`, on that same instance. There is nothing learned to apply until the first training pass runs. See [Core Concepts](concepts.md#the-is_train-flag).

**A column I expected is missing from my output.**

Check `eda.num_handler.drop_cols_` and `eda.cat_handler.drop_cols_` — columns with more than 40% missing values are dropped automatically during the training pass.

**How do I know why a column got a specific encoder or scaler?**

See [Inspecting Learned Rules](api/inspecting-rules.md) — every decision is stored as a readable attribute on the handler.

**My test set has a category value that never appeared in training. Will it crash?**

No. Unseen categories are handled gracefully: binary columns map them to `-1`, one-hot columns get all zeros, and frequency-encoded columns get `0`.

**Can I change the 40% missing-value or 5% rare-category thresholds?**

The rare-category threshold is configurable via `rare_manager(threshold=...)`. The missing-value drop threshold (40%) and outlier threshold (3%) are currently fixed constants — see the [Changelog](changelog.md) for planned configurability.

**Can I exclude an ID column from processing?**

Yes for `HandleNumerical`, via the granular methods (`num_imputer`, `outlier_manager`, `scaler`) — see [HandleNumerical](api/handle-numerical.md). Note that `full_handler()` does not currently forward `exclude`.
