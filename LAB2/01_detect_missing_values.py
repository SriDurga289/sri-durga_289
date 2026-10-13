import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age": [25, 30, np.nan, 40, 35],
    "Department": ["HR", "Finance", "Finance", np.nan, "IT"]
})

print("Original Data:")
print(df)
print("\nMissing Values:")
print(df.isnull().sum())
