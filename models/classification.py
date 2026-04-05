import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


def classify_pm25(pm25: float) -> str:
    """
    Convert PM2.5 into simple air-quality classes.
    """
    if pm25 <= 50:
        return "Good"
    elif pm25 <= 100:
        return "Moderate"
    return "Unhealthy"


def build_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build classification features X and target y.

    Target:
        AQI_Class derived from PM2.5

    Features:
        - Pollutants
        - Meteorological variables
        - Time variables
        - Station
        - Wind direction
    """
    if "PM2.5" not in df.columns:
        raise ValueError("Target source column 'PM2.5' not found in dataset.")

    candidate_features = [
        "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "year", "month", "day", "hour",
        "station", "wd"
    ]

    existing_features = [col for col in candidate_features if col in df.columns]
    if not existing_features:
        raise ValueError("No valid feature columns found in dataset.")

    df = df.dropna(subset=["PM2.5"]).copy()
    df["AQI_Class"] = df["PM2.5"].apply(classify_pm25)

    X = df[existing_features]
    y = df["AQI_Class"]

    return X, y


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def try_get_lgbm_classifier():
    """Try importing LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Create all classification models."""
    models: Dict[str, object] = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state
        ),
        "SGD Classifier": SGDClassifier(
            max_iter=2000,
            tol=1e-3,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),
    }

    LGBMClassifier = try_get_lgbm_classifier()
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state
        )

    return models


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict[str, object]
) -> pd.DataFrame:
    """Train and evaluate all classifiers."""
    results: List[Dict[str, float]] = []

    for model_name, model in models.items():
        preprocessor = get_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1_Macro": f1_score(y_test, y_pred, average="macro"),
                "F1_Weighted": f1_score(y_test, y_pred, average="weighted"),
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="F1_Macro", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


def print_results(results_df: pd.DataFrame) -> None:
    """Print formatted model comparison results."""
    print("\nCLASSIFICATION MODEL COMPARISON RESULTS")
    print("-" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 80)

    best_model = results_df.iloc[0]
    print(
        f"\nBest model by Macro F1: {best_model['Model']} "
        f"(Accuracy={best_model['Accuracy']:.4f}, "
        f"F1_Macro={best_model['F1_Macro']:.4f}, "
        f"F1_Weighted={best_model['F1_Weighted']:.4f})"
    )


def print_best_model_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_model_name: str,
    best_model: object
) -> None:
    """Print classification report for the best model."""
    preprocessor = get_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nDETAILED CLASSIFICATION REPORT: {best_model_name}")
    print("-" * 80)
    print(classification_report(y_test, y_pred))


def main():
    csv_path = "../data/beijing_combined.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"Loading dataset from: {csv_path}")
    df = load_data(csv_path)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    X, y = build_features_and_target(df)

    print("\nTask: Air Quality Classification")
    print("Target: AQI_Class (derived from PM2.5)")
    print(f"Samples used: {len(X)}")
    print(f"Features used: {X.shape[1]}")
    print(f"Feature list: {list(X.columns)}")

    print("\nClass distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    models = build_models(random_state=42)

    if "LightGBM" not in models:
        print("\nNote: LightGBM is not installed, so it will be skipped.")
        print("Install it with: pip install lightgbm")

    results_df = evaluate_models(X_train, X_test, y_train, y_test, models)
    print_results(results_df)

    results_df.to_csv("classification_comparison_results.csv", index=False)
    print("\nResults also saved to: classification_comparison_results.csv")

    best_model_name = results_df.iloc[0]["Model"]
    model_lookup = build_models(random_state=42)
    best_model = model_lookup[best_model_name]
    print_best_model_report(X_train, X_test, y_train, y_test, best_model_name, best_model)


if __name__ == "__main__":
    main()