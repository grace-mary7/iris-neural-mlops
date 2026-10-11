
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Make training more reproducible.
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


def train_model():
    # Load the CSV, which has no header row.
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    X = df[COLUMNS[:4]].astype(float).values
    y_text = df["species"].values

    # Convert species names into numeric labels.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Fit scaling on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the neural network.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=8,
        verbose=0,
    )

    # Save model and preprocessing objects.
    model.save(MODEL_DIR / "iris_model.keras")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

    # Save the exact test data for the evaluation stage.
    np.savez(
        MODEL_DIR / "test_data.npz",
        X_test=X_test,
        y_test=y_test,
    )

    metadata = {
        "feature_names": COLUMNS[:4],
        "class_names": encoder.classes_.tolist(),
        "test_rows": len(y_test),
        "epochs": 50,
        "batch_size": 8,
    }

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("MODEL TRAINING COMPLETED")
    print(f"Training rows: {len(y_train)}")
    print(f"Testing rows: {len(y_test)}")
    print(f"Saved artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    train_model()
