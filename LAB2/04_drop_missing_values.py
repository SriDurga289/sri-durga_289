import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age": [25, 30, np.nan, 40, 35],
    "Department": ["HR", "Finance", np.nan, "Finance", "IT"]
})

print("Original Data:")
print(df)

print("\nAfter Dropping Rows:")
print(df.dropna())

print("\nAfter Dropping Columns:")
print(df.dropna(axis=1))
