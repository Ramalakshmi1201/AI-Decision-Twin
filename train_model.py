import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("AI_Decision_Twin/loan_data.csv")

X = df.drop(
    "approved",
    axis=1
)

y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

accuracy = model.score(
    X_test,
    y_test
)

print(
    f"Accuracy: {accuracy:.2f}"
)

joblib.dump(
    model,
    "loan_model.pkl"
)

print("Advanced Model Saved")
