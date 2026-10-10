
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Locate the project folder and dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "AI_Decision_Twin" / "loan_data.csv"
MODEL_PATH = BASE_DIR / "AI_Decision_Twin" / "loan_model.pkl"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate input features and target
X = df.drop("approved", axis=1)
y = df["approved"]

# Train and evaluate the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, MODEL_PATH, compress=3)
print(f"Model saved to: {MODEL_PATH}")
