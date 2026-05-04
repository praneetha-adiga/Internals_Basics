import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# -------------------------
# Load data
# -------------------------
old = pd.read_csv("data/training_data.csv")
new = pd.read_csv("data/new_data.csv")

combined = pd.concat([old, new], ignore_index=True)

X = combined.drop("response_quality_score", axis=1)
y = combined["response_quality_score"]

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Load champion model (Task 1)
# -------------------------
champion = joblib.load("models/best_model.pkl")

champ_pred = champion.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champ_pred))

# -------------------------
# Retrain SAME TYPE model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

new_pred = model.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, new_pred))

# -------------------------
# Compare models
# -------------------------
improvement = champion_rmse - retrained_rmse

if improvement > 0:
    action = "promoted"
    joblib.dump(model, "models/best_model.pkl")  # replace model
else:
    action = "kept_champion"

# -------------------------
# Save output JSON
# -------------------------
output = {
    "original_data_rows": int(len(old)),
    "new_data_rows": int(len(new)),
    "combined_data_rows": int(len(combined)),
    "champion_rmse": float(champion_rmse),
    "retrained_rmse": float(retrained_rmse),
    "improvement": float(improvement),
    "min_improvement_threshold": 0,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")