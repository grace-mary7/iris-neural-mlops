
# import json
# import random
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import joblib

# # Make training more reproducible.
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

# DATA_PATH = Path("iris.csv")
# MODEL_DIR = Path("artifacts")
# MODEL_DIR.mkdir(exist_ok=True)

# COLUMNS = [
#     "sepal_length",
#     "sepal_width",
#     "petal_length",
#     "petal_width",
#     "species",
# ]


# def train_model():
#     # Load the CSV, which has no header row.
#     df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

#     X = df[COLUMNS[:4]].astype(float).values
#     y_text = df["species"].values

#     # Convert species names into numeric labels.
#     encoder = LabelEncoder()
#     y = encoder.fit_transform(y_text)

#     # Split the data.
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y,
#     )

#     # Fit scaling on training data only.
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Build the neural network.
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(4,)),
#         tf.keras.layers.Dense(16, activation="relu"),
#         tf.keras.layers.Dense(8, activation="relu"),
#         tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
#     ])

#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.fit(
#         X_train,
#         y_train,
#         epochs=50,
#         batch_size=8,
#         verbose=0,
#     )

#     # Save model and preprocessing objects.
#     model.save(MODEL_DIR / "iris_model.keras")
#     joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
#     joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

#     # Save the exact test data for the evaluation stage.
#     np.savez(
#         MODEL_DIR / "test_data.npz",
#         X_test=X_test,
#         y_test=y_test,
#     )

#     metadata = {
#         "feature_names": COLUMNS[:4],
#         "class_names": encoder.classes_.tolist(),
#         "test_rows": len(y_test),
#         "epochs": 50,
#         "batch_size": 8,
#     }

#     with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as file:
#         json.dump(metadata, file, indent=2)

#     print("MODEL TRAINING COMPLETED")
#     print(f"Training rows: {len(y_train)}")
#     print(f"Testing rows: {len(y_test)}")
#     print(f"Saved artifacts to: {MODEL_DIR}")


# if __name__ == "__main__":
#     train_model()



# import json
# import random
# from pathlib import Path

# import joblib
# import mlflow
# import mlflow.tensorflow
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from mlflow.models import infer_signature
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# # Reproducibility
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

# DATA_PATH = Path("iris.csv")
# MODEL_DIR = Path("artifacts")
# MODEL_DIR.mkdir(exist_ok=True)

# COLUMNS = [
#     "sepal_length",
#     "sepal_width",
#     "petal_length",
#     "petal_width",
#     "species",
# ]

# EPOCHS = 50
# BATCH_SIZE = 8
# EXPERIMENT_NAME = "FloraAI-Iris"


# def train_model():
#     # Track this run locally in the project folder.
#     #mlflow.set_tracking_uri("file:./mlruns")
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")
#     mlflow.set_experiment(EXPERIMENT_NAME)

#     with mlflow.start_run(run_name="iris-neural-network"):
#         df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

#         X = df[COLUMNS[:4]].astype(float).values
#         y_text = df["species"].values

#         encoder = LabelEncoder()
#         y = encoder.fit_transform(y_text)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X,
#             y,
#             test_size=0.2,
#             random_state=42,
#             stratify=y,
#         )

#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         model = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(4,)),
#             tf.keras.layers.Dense(16, activation="relu"),
#             tf.keras.layers.Dense(8, activation="relu"),
#             tf.keras.layers.Dense(
#                 len(encoder.classes_), activation="softmax"
#             ),
#         ])

#         model.compile(
#             optimizer="adam",
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"],
#         )

#         # Log model configuration.
#         mlflow.log_params({
#             "epochs": EPOCHS,
#             "batch_size": BATCH_SIZE,
#             "optimizer": "adam",
#             "loss_function": "sparse_categorical_crossentropy",
#             "hidden_layers": "16,8",
#             "activation": "relu",
#             "output_activation": "softmax",
#             "training_rows": len(X_train),
#             "testing_rows": len(X_test),
#             "random_state": 42,
#         })

#         class MLflowMetricsCallback(tf.keras.callbacks.Callback):
#             def on_epoch_end(self, epoch, logs=None):
#                 logs = logs or {}
#                 for metric_name, metric_value in logs.items():
#                     mlflow.log_metric(
#                         f"train_{metric_name}",
#                         float(metric_value),
#                         step=epoch,
#                     )

#         model.fit(
#             X_train,
#             y_train,
#             epochs=EPOCHS,
#             batch_size=BATCH_SIZE,
#             verbose=0,
#             callbacks=[MLflowMetricsCallback()],
#         )

#         # Save the same artifacts used by the existing API and DVC.
#         model_path = MODEL_DIR / "iris_model.keras"
#         model.save(model_path)
#         joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
#         joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

#         np.savez(
#             MODEL_DIR / "test_data.npz",
#             X_test=X_test,
#             y_test=y_test,
#         )

#         metadata = {
#             "feature_names": COLUMNS[:4],
#             "class_names": encoder.classes_.tolist(),
#             "test_rows": len(y_test),
#             "epochs": EPOCHS,
#             "batch_size": BATCH_SIZE,
#         }

#         with open(
#             MODEL_DIR / "metadata.json", "w", encoding="utf-8"
#         ) as file:
#             json.dump(metadata, file, indent=2)

#         # Log the model and its input/output signature.
#         sample_input = X_train[:5].astype(np.float32)
#         sample_output = model.predict(sample_input, verbose=0)
#         signature = infer_signature(sample_input, sample_output)

#         mlflow.log_artifact(str(model_path), artifact_path="model")
#         mlflow.log_artifact(
#             str(MODEL_DIR / "metadata.json"),
#             artifact_path="metadata",
#         )

#         print("MODEL TRAINING COMPLETED")
#         print(f"Training rows: {len(y_train)}")
#         print(f"Testing rows: {len(y_test)}")
#         print(f"Saved artifacts to: {MODEL_DIR}")
#         print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


# if __name__ == "__main__":
#     train_model()




import json
import random
from pathlib import Path

import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = Path("iris.csv")
MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

EPOCHS = 50
BATCH_SIZE = 8
EXPERIMENT_NAME = "FloraAI-Iris"
REGISTERED_MODEL_NAME = "FloraAI-Iris-Classifier"


def train_model():
    # Configure MLflow tracking.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="iris-neural-network"):
        # Load dataset.
        df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

        X = df[COLUMNS[:4]].astype(float).values
        y_text = df["species"].values

        # Encode species names.
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_text)

        # Split dataset.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Scale features using training data only.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build neural network.
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(
                len(encoder.classes_),
                activation="softmax",
            ),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Log training configuration.
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "adam",
            "loss_function": "sparse_categorical_crossentropy",
            "hidden_layers": "16,8",
            "activation": "relu",
            "output_activation": "softmax",
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "random_state": 42,
        })

        # Log training metrics after each epoch.
        class MLflowMetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                for metric_name, metric_value in logs.items():
                    mlflow.log_metric(
                        f"train_{metric_name}",
                        float(metric_value),
                        step=epoch,
                    )

        # Train the model.
        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[MLflowMetricsCallback()],
        )

        # Save artifacts for the existing API and DVC pipeline.
        model_path = MODEL_DIR / "iris_model.keras"
        model.save(model_path)

        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

        np.savez(
            MODEL_DIR / "test_data.npz",
            X_test=X_test,
            y_test=y_test,
        )

        metadata = {
            "feature_names": COLUMNS[:4],
            "class_names": encoder.classes_.tolist(),
            "test_rows": len(y_test),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }

        metadata_path = MODEL_DIR / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        # Create input/output signature for MLflow.
        sample_input = X_train[:5].astype(np.float32)
        sample_output = model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_output)

        # Log and register the model in the MLflow Model Registry.
        model_info = mlflow.tensorflow.log_model(
            model=model,
            artifact_path="iris_classifier",
            signature=signature,
            input_example=sample_input,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Log metadata artifact.
        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="metadata",
        )

        print("MODEL TRAINING COMPLETED")
        print(f"Training rows: {len(y_train)}")
        print(f"Testing rows: {len(y_test)}")
        print(f"Saved artifacts to: {MODEL_DIR}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")
        print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    train_model()
