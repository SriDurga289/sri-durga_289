import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

sns.histplot(df["age"].dropna(), bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
