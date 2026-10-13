import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("First 5 rows:")
print(df.head())
print("Mean values:")
print(df.mean())
print("Median values:")
print(df.median())
print("Mode values:")
print(df.mode().iloc[0])
