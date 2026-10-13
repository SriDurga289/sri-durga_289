import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age": [25, 30, np.nan, 40, 35],
    "Department": ["HR", "Finance", "Finance", np.nan, "IT"]
})

print("Forward Fill:")
print(df.ffill())

print("\nBackward Fill:")
print(df.bfill())
