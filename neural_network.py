import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv(
    "iris.csv",
    header=None,
    names=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]
)

# Remove empty rows
data = data.dropna()

# Separate input and output
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

# Convert flower names into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale input data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create neural network
model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Make a prediction
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

new_flower = scaler.transform(new_flower)

prediction = model.predict(new_flower)

predicted_class = np.argmax(prediction)

print("Predicted Flower:", encoder.inverse_transform([predicted_class])[0])