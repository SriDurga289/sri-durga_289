import pandas as pd
from scipy.stats import zscore
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
col = 'sepal length (cm)'
z_scores = zscore(df[col])
print("Z-score outliers:")
print(df[abs(z_scores) > 3])
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("IQR outliers:")
print(df[(df[col] < lower) | (df[col] > upper)])
