import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
sns.violinplot(data=df, x='species', y='petal length (cm)')
plt.title('Violin Plot of Petal Length by Species')
plt.show()
