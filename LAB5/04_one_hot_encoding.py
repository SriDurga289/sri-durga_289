import seaborn as sns
import pandas as pd

tips = sns.load_dataset("tips")
tips_onehot = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])
print(tips_onehot.head())
