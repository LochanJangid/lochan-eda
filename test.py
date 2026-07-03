import pandas
from lochan_eda.orchestrator import AutomatedEDA

df = pandas.read_csv("test.csv")

print(df.isna().sum())

auto = AutomatedEDA(df)

cleaned_df = auto.run_pipeline()

print(cleaned_df.isna().sum())