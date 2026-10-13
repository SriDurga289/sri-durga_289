import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Range:")
print(df.max() - df.min())
print("Variance:")
print(df.var())
print("Standard deviation:")
print(df.std())
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
print("IQR:")
print(q3 - q1)
