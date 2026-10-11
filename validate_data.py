
import pandas as pd
from pathlib import Path

DATA_PATH = Path("iris.csv")

EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

EXPECTED_SPECIES = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = EXPECTED_COLUMNS

    # Check columns
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Incorrect columns. Found: {list(df.columns)}"
        )

    # Check missing values
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    # Check numeric measurements
    features = EXPECTED_COLUMNS[:4]
    for column in features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[features].isnull().values.any():
        raise ValueError("Some flower measurements are not numeric.")

    if not df[features].apply(lambda col: col.map(lambda x: x > 0)).all().all():
        raise ValueError("Flower measurements must be greater than zero.")

    # Check species names
    found_species = set(df["species"].unique())
    if found_species != EXPECTED_SPECIES:
        raise ValueError(
            f"Unexpected species names: {found_species}"
        )

    # Check dataset size
    if len(df) < 30:
        raise ValueError("Dataset has too few rows.")

    print("DATA VALIDATION PASSED")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(features)}")
    print(f"Species: {sorted(found_species)}")


if __name__ == "__main__":
    validate_data()
