# import os
# import numpy as np
# import joblib
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from tensorflow.keras.models import load_model

# # Load the existing trained model and preprocessing files
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# model = load_model(os.path.join(ARTIFACTS_DIR, "iris_model.keras"))
# scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
# label_encoder = joblib.load(
#     os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
# )

# app = FastAPI(
#     title="Iris Flower Prediction API",
#     description="Predict Iris flower species using a trained neural network.",
#     version="1.0.0",
# )

# from fastapi.middleware.cors import CORSMiddleware

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
# #     allow_credentials=False,
# #     allow_methods=["GET", "POST"],
# #     allow_headers=["Content-Type"],
# # )


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#         "https://floraai-8p2y.onrender.com",
#     ],
#     allow_credentials=False,
#     allow_methods=["GET", "POST"],
#     allow_headers=["Content-Type"],
# )

# class FlowerInput(BaseModel):
#     sepal_length: float = Field(gt=0, le=10)
#     sepal_width: float = Field(gt=0, le=10)
#     petal_length: float = Field(gt=0, le=10)
#     petal_width: float = Field(gt=0, le=10)


# @app.get("/")
# def home():
#     return {
#         "message": "Iris Flower Prediction API is running",
#         "docs": "/docs",
#     }


# @app.get("/health")
# def health():
#     return {"status": "healthy"}


# @app.post("/predict")
# def predict(flower: FlowerInput):
#     measurements = np.array([[
#         flower.sepal_length,
#         flower.sepal_width,
#         flower.petal_length,
#         flower.petal_width,
#     ]])

#     scaled_measurements = scaler.transform(measurements)
#     probabilities = model.predict(scaled_measurements, verbose=0)[0]
#     predicted_index = int(np.argmax(probabilities))
#     predicted_species = str(
#         label_encoder.inverse_transform([predicted_index])[0]
#     )

#     return {
#         "predicted_species": predicted_species,
#         "confidence": round(float(probabilities[predicted_index]), 4),
#         "probabilities": {
#             str(species): round(float(probability), 4)
#             for species, probability in zip(
#                 label_encoder.classes_, probabilities
#             )
#         },
#     }


import os

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from tensorflow.keras.models import load_model


# --------------------------------------------------
# Load trained model and preprocessing artifacts
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model = load_model(
    os.path.join(ARTIFACTS_DIR, "iris_model.keras")
)

scaler = joblib.load(
    os.path.join(ARTIFACTS_DIR, "scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
)


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Iris Flower Prediction API",
    description=(
        "Predict Iris flower species using a trained "
        "neural network with Prometheus monitoring."
    ),
    version="1.1.0",
)


# --------------------------------------------------
# Prometheus API monitoring
# --------------------------------------------------

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://floraai-8p2y.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# Input validation
# --------------------------------------------------

class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Iris Flower Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(flower: FlowerInput):
    measurements = np.array(
        [[
            flower.sepal_length,
            flower.sepal_width,
            flower.petal_length,
            flower.petal_width,
        ]]
    )

    scaled_measurements = scaler.transform(measurements)

    probabilities = model.predict(
        scaled_measurements,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_species = str(
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    )

    return {
        "predicted_species": predicted_species,
        "confidence": round(
            float(probabilities[predicted_index]), 4
        ),
        "probabilities": {
            str(species): round(float(probability), 4)
            for species, probability in zip(
                label_encoder.classes_,
                probabilities,
            )
        },
    }




