import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y.

    Target:
        PM2.5

    Features:
        - Pollutants
        - Meteorological variables
        - Time variables
        - Station
    """
    if "PM2.5" not in df.columns:
        raise ValueError("Target column 'PM2.5' not found in dataset.")

    # Keep a practical feature set for prediction
    candidate_features = [
        "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "year", "month", "day", "hour",
        "station", "wd"
    ]

    existing_features = [col for col in candidate_features if col in df.columns]
    if not existing_features:
        raise ValueError("No valid feature columns found in dataset.")

    # Remove rows where target is missing
    df = df.dropna(subset=["PM2.5"]).copy()

    X = df[existing_features]
    y = df["PM2.5"]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def rmse_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def try_get_lgbm():
    """
    Try importing LightGBM.
    Returns the model class if installed, otherwise None.
    """
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    except ImportError:
        return None


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Create all regression models."""
    models: Dict[str, object] = {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "SGD Regressor": SGDRegressor(
            max_iter=2000,
            tol=1e-3,
            random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),
    }

    LGBMRegressor = try_get_lgbm()
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
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
    """Train and evaluate all models."""
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
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": rmse_score(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    return results_df


def print_results(results_df: pd.DataFrame) -> None:
    """Print nicely formatted results."""
    print("\nMODEL COMPARISON RESULTS")
    print("-" * 70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 70)

    best_model = results_df.iloc[0]
    print(
        f"\nBest model by R²: {best_model['Model']} "
        f"(MAE={best_model['MAE']:.4f}, RMSE={best_model['RMSE']:.4f}, R²={best_model['R2']:.4f})"
    )


def main():
    # Default path
    csv_path = "../data/beijing_combined.csv"

    # Optional CLI usage: python model_comparison.py path/to/file.csv
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"Loading dataset from: {csv_path}")
    df = load_data(csv_path)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    X, y = build_features(df)

    print(f"\nTarget: PM2.5")
    print(f"Number of samples used: {len(X)}")
    print(f"Number of features used: {X.shape[1]}")
    print(f"Features: {list(X.columns)}")

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    models = build_models(random_state=42)

    if "LightGBM" not in models:
        print("\nNote: LightGBM is not installed, so it will be skipped.")
        print("Install it with: pip install lightgbm")

    results_df = evaluate_models(X_train, X_test, y_train, y_test, models)
    print_results(results_df)

    # Optional: save results
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nResults also saved to: model_comparison_results.csv")


if __name__ == "__main__":
    main()