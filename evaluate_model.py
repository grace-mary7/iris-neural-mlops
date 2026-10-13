
# import json
# from pathlib import Path

# import joblib
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, classification_report

# MODEL_DIR = Path("artifacts")
# MINIMUM_ACCURACY = 0.90


# def evaluate_model():
#     model = tf.keras.models.load_model(
#         MODEL_DIR / "iris_model.keras"
#     )

#     test_data = np.load(MODEL_DIR / "test_data.npz")
#     X_test = test_data["X_test"]
#     y_test = test_data["y_test"]

#     encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

#     probabilities = model.predict(X_test, verbose=0)
#     predictions = np.argmax(probabilities, axis=1)

#     accuracy = float(accuracy_score(y_test, predictions))

#     report = classification_report(
#         y_test,
#         predictions,
#         labels=list(range(len(encoder.classes_))),
#         target_names=encoder.classes_,
#         zero_division=0,
#     )

#     metrics = {
#         "accuracy": accuracy,
#         "minimum_required_accuracy": MINIMUM_ACCURACY,
#         "test_rows": int(len(y_test)),
#         "passed": accuracy >= MINIMUM_ACCURACY,
#     }

#     with open(
#         MODEL_DIR / "metrics.json",
#         "w",
#         encoding="utf-8",
#     ) as file:
#         json.dump(metrics, file, indent=2)

#     print("MODEL EVALUATION")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Required accuracy: {MINIMUM_ACCURACY:.2f}")
#     print(report)

#     if not metrics["passed"]:
#         raise RuntimeError(
#             "Model failed the accuracy threshold. Deployment must stop."
#         )

#     print("MODEL EVALUATION PASSED")


# if __name__ == "__main__":
#     evaluate_model()



import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = Path("artifacts")
MINIMUM_ACCURACY = 0.90

# Use the same MLflow database and experiment as training.
mlflow.set_tracking_uri("sqlite:///E:/projects/neural/mlflow.db")
mlflow.set_experiment("FloraAI-Iris")


def evaluate_model():
    with mlflow.start_run(run_name="iris-evaluation"):
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

        # Log evaluation metrics and parameters to MLflow.
        mlflow.log_params({
            "minimum_accuracy": MINIMUM_ACCURACY,
            "evaluation_dataset_rows": len(y_test),
        })

        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "minimum_required_accuracy": MINIMUM_ACCURACY,
            "test_rows": len(y_test),
            "passed_quality_gate": float(metrics["passed"]),
        })

        mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
        mlflow.log_text(report, "classification_report.txt")

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

