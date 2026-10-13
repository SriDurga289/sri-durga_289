import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], s=df['petal length (cm)'] * 20, c=df['species'], alpha=0.5)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Bubble Chart')
plt.show()
sns.pairplot(df, hue='species')
plt.show()
