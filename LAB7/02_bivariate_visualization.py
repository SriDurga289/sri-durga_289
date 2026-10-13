import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.show()
plt.plot(df['sepal length (cm)'])
plt.title('Line Chart of Sepal Length')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.show()
sns.barplot(data=df, x='species', y='sepal length (cm)')
plt.xticks([0, 1, 2], iris.target_names)
plt.title('Mean Sepal Length by Species')
plt.show()
