
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# PARALILOOP AI - RANDOM FOREST MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "prediction_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "paraliloop_model.pkl"

METRICS_FILE = MODEL_DIR / "model_metrics.json"

PREDICTIONS_FILE = (
    BASE_DIR / "data" / "processed" /
    "test_predictions.csv"
)

FEATURES = [
    "latitude_grid",
    "longitude_grid",
    "month",
    "day_of_year",
    "previous_day_count",
    "previous_3_day_count",
    "previous_7_day_count"
]

TARGET = "target_next_day"


def load_dataset():

    print("\nLoading prediction dataset...")

    if not INPUT_FILE.exists():
        print("ERROR: Prediction dataset not found!")
        return None

    data = pd.read_csv(INPUT_FILE)

    required = FEATURES + [TARGET, "year"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        print("Missing columns:", missing)
        return None

    data = data.dropna(
        subset=required
    ).copy()

    data["year"] = data["year"].astype(int)

    print("Total records:", len(data))
    print("Available years:", sorted(data["year"].unique()))

    return data


def train_model(data):

    print("\nSplitting data by year...")

    # Train on 2021-2024 and test on 2025
    train_data = data[data["year"] < 2025].copy()
    test_data = data[data["year"] == 2025].copy()

    if train_data.empty or test_data.empty:
        print("ERROR: Training or testing data is empty.")
        return None

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    if y_train.nunique() < 2:
        print("ERROR: Training data has only one target class.")
        return None

    if y_test.nunique() < 2:
        print("WARNING: Test data has only one target class.")
        print("ROC-AUC will not be calculated.")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):

    print("\n========== MODEL EVALUATION ==========")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, probabilities)
        print("ROC-AUC:", round(auc, 4))
    else:
        auc = None
        print("ROC-AUC: Not available (only one test class)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return predictions, probabilities, {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "training_years": "2021-2024",
        "testing_year": 2025
    }


def save_results(model, X_test, y_test,
                 predictions, probabilities, metrics):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save trained model
    joblib.dump(model, MODEL_FILE)

    # Save evaluation metrics
    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    # Save test predictions
    results = X_test.copy()

    results["actual"] = y_test.values
    results["predicted"] = predictions
    results["prediction_probability"] = probabilities

    results.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # Save feature importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    importance.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False
    )

    print("\n========== SAVED FILES ==========")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("Predictions:", PREDICTIONS_FILE)
    print("Feature importance:", MODEL_DIR / "feature_importance.csv")


def main():

    data = load_dataset()

    if data is None:
        return

    result = train_model(data)

    if result is None:
        return

    model, X_test, y_test = result

    predictions, probabilities, metrics = evaluate_model(
        model, X_test, y_test
    )

    save_results(
        model,
        X_test,
        y_test,
        predictions,
        probabilities,
        metrics
    )

    print("\nParaliLoop AI model training completed!")


if __name__ == "__main__":
    main()
