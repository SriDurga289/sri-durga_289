import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import ttest_ind, f_oneway

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
setosa = df[df['species'] == 0]['sepal length (cm)']
versicolor = df[df['species'] == 1]['sepal length (cm)']
virginica = df[df['species'] == 2]['sepal length (cm)']
print("T-test:", ttest_ind(setosa, versicolor))
print("ANOVA:", f_oneway(setosa, versicolor, virginica))
