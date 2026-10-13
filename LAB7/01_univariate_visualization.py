import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
sns.histplot(df['sepal length (cm)'], bins=20, kde=True)
plt.title('Histogram of Sepal Length')
plt.show()
sns.boxplot(x=df['sepal length (cm)'])
plt.title('Boxplot of Sepal Length')
plt.show()
counts = df['species'].value_counts().sort_index()
plt.pie(counts, labels=iris.target_names, autopct='%1.1f%%')
plt.title('Species Distribution')
plt.show()
