import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("response_quality_score", axis=1)
y = df["response_quality_score"]

# -------------------------
# Train-test split (MANDATORY)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_experiment("promptlab-response-quality-score")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge()
}

results = []
best_mae = float("inf")
best_model_name = None
best_model = None

# -------------------------
# Train models
# -------------------------
for name, model in models.items():

    with mlflow.start_run():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("priority", "high")

        mlflow.sklearn.log_model(model, "model")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model = model

# -------------------------
# Save best model
# -------------------------
joblib.dump(best_model, "models/best_model.pkl")

# -------------------------
# Save JSON output (REQUIRED)
# -------------------------
output = {
    "experiment_name": "promptlab-response-quality-score",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed successfully")