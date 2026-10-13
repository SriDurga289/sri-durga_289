import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = sns.load_dataset("titanic")
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["fare"] = df["fare"].fillna(df["fare"].median())
df["family_size"] = df["sibsp"] + df["parch"]

X = df.drop(columns=["survived", "alive", "who", "adult_male", "deck", "embark_town"])
y = df["survived"]

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!")
print("Predictions:", model.predict(X_test.head()))
