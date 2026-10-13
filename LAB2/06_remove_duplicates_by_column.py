import pandas as pd

df = pd.DataFrame({
    "ID": [1, 2, 2, 3, 4, 4, 5],
    "Name": ["Alice", "Bob", "Bob", "Charlie", "David", "David", "David"],
    "Age": [25, 30, 30, 35, 40, 40, 40]
})

print("Remove duplicates by ID:")
print(df.drop_duplicates(subset=["ID"]))

print("\nRemove duplicates by Name:")
print(df.drop_duplicates(subset=["Name"]))
