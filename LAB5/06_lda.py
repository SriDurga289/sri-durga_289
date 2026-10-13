import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
X = iris.data
y = iris.target
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
lda_df['target'] = y
print("Explained variance ratio:", lda.explained_variance_ratio_)
print(lda_df.head())
for label in range(3):
    plt.scatter(lda_df.loc[lda_df['target'] == label, 'LD1'], lda_df.loc[lda_df['target'] == label, 'LD2'], label=iris.target_names[label])
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('LDA Projection of Iris Dataset')
plt.legend()
plt.show()
