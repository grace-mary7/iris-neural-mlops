
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = Path("artifacts")
MINIMUM_ACCURACY = 0.90


def evaluate_model():
    model = tf.keras.models.load_model(
        MODEL_DIR / "iris_model.keras"
    )

    test_data = np.load(MODEL_DIR / "test_data.npz")
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = float(accuracy_score(y_test, predictions))

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "minimum_required_accuracy": MINIMUM_ACCURACY,
        "test_rows": int(len(y_test)),
        "passed": accuracy >= MINIMUM_ACCURACY,
    }

    with open(
        MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("MODEL EVALUATION")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Required accuracy: {MINIMUM_ACCURACY:.2f}")
    print(report)

    if not metrics["passed"]:
        raise RuntimeError(
            "Model failed the accuracy threshold. Deployment must stop."
        )

    print("MODEL EVALUATION PASSED")


if __name__ == "__main__":
    evaluate_model()
