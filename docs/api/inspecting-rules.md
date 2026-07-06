# Inspecting Learned Rules

Because nothing in `lochan-eda` is a black box, every rule learned during an `is_train=True` pass is stored as a plain attribute on the handler, so you can audit *why* a decision was made.

```python
eda = AutomatedEDA()
eda.run_pipeline(df=X_train, target=y_train, is_train=True)

# Which columns were dropped, and why?
print(eda.num_handler.drop_cols_)      # numeric columns dropped (>40% missing)
print(eda.cat_handler.drop_cols_)      # categorical columns dropped (>40% missing)

# What imputation value did each column get?
print(eda.num_handler.impute_values_)  # {column: fill_value}

# What outlier rule and scaler did each numeric column get?
print(eda.num_handler.outlier_rules_)  # {column: {"type": "clip"/"sqrt"/"log1p", ...}}
print(eda.num_handler.scalers_)        # {column: fitted sklearn scaler object}

# What encoding strategy did each categorical column get?
print(eda.cat_handler.encode_types_)   # {column: "binary"/"ohe"/"te"/"freq"}

# The exact column order the final model-ready output must match
print(eda.final_columns_)
```

## Full attribute reference

| Attribute | Owner | Description |
|---|---|---|
| `drop_cols_` | `HandleNumerical`, `HandleCategorical` | Columns dropped for exceeding the 40% missing-value threshold. |
| `impute_values_` | `HandleNumerical` | `{column: fill_value}` used for imputation. |
| `outlier_rules_` | `HandleNumerical` | `{column: rule_dict}` describing the outlier treatment applied. |
| `scalers_` | `HandleNumerical` | `{column: fitted scaler}`. |
| `transforms_` | `HandleNumerical` | `{column: "log1p"}` for pre-scaling log transforms. |
| `impute_modes_` | `HandleCategorical` | `{column: mode_value}` used for imputation. |
| `rare_cats_` | `HandleCategorical` | `{column: [rare category values]}` grouped into `"Other"`. |
| `encode_types_` | `HandleCategorical` | `{column: encoding strategy}`. |
| `binary_maps_` | `HandleCategorical` | `{column: {value: 0/1}}`. |
| `ohe_columns_` | `HandleCategorical` | `{column: [one-hot column names]}`. |
| `freq_maps_` | `HandleCategorical` | `{column: {value: frequency}}`. |
| `target_encoders_` | `HandleCategorical` | `{column: fitted sklearn TargetEncoder}`. |
| `final_columns_` | `AutomatedEDA` | Full training-time output column order, used to align test-time output. |

This is especially useful for debugging a mismatch between expected and actual output columns, or for explaining preprocessing decisions in a report or model card.
