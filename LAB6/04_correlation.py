import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import spearmanr

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Pearson correlation:")
print(df.corr(method='pearson'))
print("Spearman correlation:")
print(df.corr(method='spearman'))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
