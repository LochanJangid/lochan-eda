# Full Example (end-to-end)

This script is fully self-contained — copy it into a file and run it to see the pipeline work on synthetic, realistically messy data.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda import AutomatedEDA

# 1. Create a messy, realistic dataset
rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame({
    "income": rng.exponential(scale=45000, size=n),
    "age": rng.normal(35, 10, n).clip(18, 90),
    "city": rng.choice(["Jaipur", "Delhi", "Mumbai", "Bangalore", "Chennai"], size=n),
    "occupation": rng.choice([f"occ_{i}" for i in range(15)], size=n),
    "gender": rng.choice(["M", "F"], size=n),
})
target = pd.Series(rng.integers(0, 2, size=n), name="churn")

# Inject missing values, like real-world data
for col in ["income", "age", "city"]:
    df.loc[rng.random(n) < 0.08, col] = np.nan

# 2. Split BEFORE preprocessing — this is what prevents leakage
X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42
)

# 3. Fit on train, transform test using the exact same learned rules
eda = AutomatedEDA()
X_train_clean = eda.run_pipeline(X_train, target=y_train, is_train=True)
X_test_clean = eda.run_pipeline(X_test, is_train=False)

print("Train shape:", X_train_clean.shape)
print("Test shape :", X_test_clean.shape)
print("Columns match:", list(X_train_clean.columns) == list(X_test_clean.columns))
```

## Feeding the output into a model

Because the output of `run_pipeline` is always fully numeric and `NaN`-free with matching train/test schemas, it drops straight into any scikit-learn estimator:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train_clean, y_train)

preds = model.predict_proba(X_test_clean)[:, 1]
print("Test AUC:", roc_auc_score(y_test, preds))
```
