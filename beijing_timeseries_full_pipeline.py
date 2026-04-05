#!/usr/bin/env python3
"""
End-to-end Beijing air-quality time-series forecasting pipeline.

What this script does
---------------------
1. Loads the Beijing multi-site air quality dataset.
2. Performs robust preprocessing for time-series work.
3. Builds lag/rolling/seasonal features.
4. Trains a Random Forest forecaster with a strict time-based split.
5. Produces recursive future forecasts.
6. Produces seasonal forecasting summaries such as:
   - expected PM2.5 level by season
   - expected air-quality category by season
   - expected weather feel / temperature range by season
7. Generates proof artifacts for presentation claims:
   - pollutant interdependency correlations
   - weather-effect correlations
   - feature importance
   - seasonal PM2.5 behavior

Usage examples
--------------
python beijing_timeseries_full_pipeline.py \
  --csv data/beijing_combined.csv \
  --output_dir outputs \
  --station Aotizhongxin \
  --forecast_horizon_hours 168

python beijing_timeseries_full_pipeline.py \
  --csv data/beijing_combined.csv \
  --output_dir outputs_all \
  --station ALL \
  --forecast_horizon_hours 168

Notes
-----
- station=ALL aggregates numeric columns across stations by datetime.
- The future-weather forecast is scenario-based: it uses historical average
  weather by month+hour as future exogenous conditions.
- This is intentionally presentation-friendly and reproducible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

POLLUTANT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_COLUMNS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TIME_COLUMNS = ["year", "month", "day", "hour"]
OPTIONAL_CATEGORICAL_COLUMNS = ["wd", "station"]


@dataclass
class ModelArtifacts:
    model: RandomForestRegressor
    feature_columns: List[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    predictions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beijing PM2.5 time-series forecasting pipeline")
    parser.add_argument("--csv", required=True, help="Path to beijing_combined.csv")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save results")
    parser.add_argument(
        "--station",
        default="Aotizhongxin",
        help="Station name to model, or ALL to aggregate all stations by datetime",
    )
    parser.add_argument(
        "--forecast_horizon_hours",
        type=int,
        default=24 * 7,
        help="Number of future hours to forecast recursively",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.20,
        help="Final fraction of time-ordered data reserved for testing",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def pm25_category(pm25_value: float) -> str:
    if pd.isna(pm25_value):
        return "Unknown"
    if pm25_value <= 35:
        return "Good"
    if pm25_value <= 75:
        return "Moderate"
    return "Unhealthy"


def thermal_label(temp_c: float) -> str:
    if pd.isna(temp_c):
        return "Unknown"
    if temp_c <= 0:
        return "Freezing"
    if temp_c <= 10:
        return "Cold"
    if temp_c <= 18:
        return "Cool"
    if temp_c <= 26:
        return "Mild / Pleasant"
    if temp_c <= 32:
        return "Warm"
    return "Hot"


def rain_label(rain_mm: float) -> str:
    if pd.isna(rain_mm):
        return "Unknown"
    if rain_mm == 0:
        return "Dry"
    if rain_mm < 2.5:
        return "Light Rain"
    if rain_mm < 7.6:
        return "Moderate Rain"
    return "Heavy Rain"


def wind_label(wspm: float) -> str:
    if pd.isna(wspm):
        return "Unknown"
    if wspm < 2:
        return "Calm"
    if wspm < 4:
        return "Light Breeze"
    if wspm < 7:
        return "Breezy"
    return "Windy"


def cyclic_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * series / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(radians),
        f"{prefix}_cos": np.cos(radians),
    }, index=series.index)


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = set(TIME_COLUMNS + POLLUTANT_COLUMNS + WEATHER_COLUMNS + ["wd", "station"])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def preprocess_data(df: pd.DataFrame, station: str) -> pd.DataFrame:
    data = df.copy()

    # Build datetime and sort for proper time-series order.
    data["datetime"] = pd.to_datetime(data[["year", "month", "day", "hour"]], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values(["station", "datetime"]).reset_index(drop=True)

    # Keep raw station data or aggregate across all stations.
    if station.upper() != "ALL":
        data = data[data["station"] == station].copy()
        if data.empty:
            raise ValueError(f"No rows found for station={station!r}")
    else:
        numeric_cols = [c for c in data.columns if c not in OPTIONAL_CATEGORICAL_COLUMNS + ["datetime"]]
        aggregated = data.groupby("datetime", as_index=False)[numeric_cols].mean(numeric_only=True)
        aggregated["station"] = "ALL"
        data = aggregated.copy()

    data = data.sort_values("datetime").reset_index(drop=True)

    # Remove duplicate timestamps if they exist.
    data = data.drop_duplicates(subset=["datetime"]).copy()

    # Drop non-informative identifier if present.
    if "No" in data.columns:
        data = data.drop(columns=["No"])

    # Reconstruct calendar columns after aggregation.
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["dayofweek"] = data["datetime"].dt.dayofweek
    data["weekofyear"] = data["datetime"].dt.isocalendar().week.astype(int)
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["season"] = data["month"].apply(season_from_month)

    # Handle wind direction if available.
    if "wd" in data.columns:
        if data["wd"].isna().all():
            data["wd"] = "Unknown"
        else:
            mode_value = data["wd"].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            data["wd"] = data["wd"].fillna(fill_value)

    # Numeric imputation strategy appropriate for time-series.
    numeric_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    numeric_columns = [c for c in numeric_columns if c != "datetime"]
    data[numeric_columns] = data[numeric_columns].interpolate(method="linear", limit_direction="both")
    data[numeric_columns] = data[numeric_columns].ffill().bfill()

    # Final missing-value safeguard.
    for col in numeric_columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    # Cyclical encodings for seasonality.
    for src, period, prefix in [("hour", 24, "hour"), ("month", 12, "month"), ("dayofweek", 7, "dow")]:
        enc = cyclic_encode(data[src], period=period, prefix=prefix)
        data = pd.concat([data, enc], axis=1)

    # Derived labels for seasonal-weather storytelling.
    data["pm25_category"] = data["PM2.5"].apply(pm25_category)
    data["thermal_label"] = data["TEMP"].apply(thermal_label)
    data["rain_label"] = data["RAIN"].apply(rain_label)
    data["wind_label"] = data["WSPM"].apply(wind_label)

    return data


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("datetime").reset_index(drop=True)

    # Lag features: hourly, daily, multi-day, weekly memory.
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lag_hours:
        data[f"pm25_lag_{lag}"] = data["PM2.5"].shift(lag)

    # Rolling features.
    for window in [6, 12, 24, 72, 168]:
        data[f"pm25_roll_mean_{window}"] = data["PM2.5"].shift(1).rolling(window=window).mean()
        data[f"pm25_roll_std_{window}"] = data["PM2.5"].shift(1).rolling(window=window).std()
        data[f"temp_roll_mean_{window}"] = data["TEMP"].shift(1).rolling(window=window).mean()
        data[f"wind_roll_mean_{window}"] = data["WSPM"].shift(1).rolling(window=window).mean()

    # Simple change features.
    data["pm25_diff_1"] = data["PM2.5"].diff(1)
    data["pm25_diff_24"] = data["PM2.5"].diff(24)
    data["temp_diff_1"] = data["TEMP"].diff(1)
    data["wind_diff_1"] = data["WSPM"].diff(1)

    return data.dropna().reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    base_features = [
        # pollutant context
        "PM10", "SO2", "NO2", "CO", "O3",
        # weather context
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        # raw temporal context
        "year", "month", "day", "hour", "dayofweek", "weekofyear", "is_weekend",
        # cyclical temporal encodings
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    ]
    lag_features = [c for c in df.columns if c.startswith("pm25_lag_")]
    rolling_features = [c for c in df.columns if c.startswith("pm25_roll_") or c.startswith("temp_roll_") or c.startswith("wind_roll_")]
    diff_features = [c for c in ["pm25_diff_1", "pm25_diff_24", "temp_diff_1", "wind_diff_1"] if c in df.columns]
    feature_columns = base_features + lag_features + rolling_features + diff_features
    return df[feature_columns].copy(), feature_columns


def time_train_test_split(df: pd.DataFrame, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2), "MAPE": mape}


def train_forecaster(df: pd.DataFrame, test_fraction: float, random_state: int) -> ModelArtifacts:
    X, feature_columns = build_feature_matrix(df)
    model_df = pd.concat([df[["datetime", "PM2.5"]], X], axis=1).dropna().reset_index(drop=True)
    train_df, test_df = time_train_test_split(model_df, test_fraction=test_fraction)

    X_train = train_df[feature_columns]
    y_train = train_df["PM2.5"]
    X_test = test_df[feature_columns]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return ModelArtifacts(
        model=model,
        feature_columns=feature_columns,
        train_df=train_df,
        test_df=test_df,
        predictions=predictions,
    )


def build_weather_reference(df: pd.DataFrame) -> pd.DataFrame:
    weather_ref = (
        df.groupby(["month", "hour"], as_index=False)[["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]]
        .mean()
        .sort_values(["month", "hour"])
    )
    return weather_ref


def recursive_future_forecast(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    artifacts: ModelArtifacts,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Recursive forecast using historical average weather by month+hour as the future scenario.
    """
    history = feature_df.copy().sort_values("datetime").reset_index(drop=True)
    weather_ref = build_weather_reference(base_df)
    future_rows: List[Dict[str, float | str | pd.Timestamp]] = []

    for step in range(1, horizon_hours + 1):
        last_dt = history.iloc[-1]["datetime"]
        next_dt = last_dt + pd.Timedelta(hours=1)

        month = int(next_dt.month)
        hour = int(next_dt.hour)
        dow = int(next_dt.dayofweek)

        ref = weather_ref[(weather_ref["month"] == month) & (weather_ref["hour"] == hour)]
        if ref.empty:
            ref = weather_ref[weather_ref["month"] == month]
        if ref.empty:
            ref = weather_ref
        ref_row = ref.mean(numeric_only=True)

        row: Dict[str, float | str | pd.Timestamp] = {
            "datetime": next_dt,
            "year": next_dt.year,
            "month": month,
            "day": int(next_dt.day),
            "hour": hour,
            "dayofweek": dow,
            "weekofyear": int(next_dt.isocalendar().week),
            "is_weekend": int(dow in [5, 6]),
        }

        # Exogenous conditions projected from historical seasonal/hourly averages.
        for col in ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
            row[col] = float(ref_row[col])

        row["season"] = season_from_month(month)
        row["station"] = str(base_df["station"].iloc[0])

        encodings = {
            **cyclic_encode(pd.Series([hour]), 24, "hour").iloc[0].to_dict(),
            **cyclic_encode(pd.Series([month]), 12, "month").iloc[0].to_dict(),
            **cyclic_encode(pd.Series([dow]), 7, "dow").iloc[0].to_dict(),
        }
        row.update({k: float(v) for k, v in encodings.items()})

        # Lag features from the latest known/forecast history.
        pm_series = history["PM2.5"].tolist()
        temp_series = history["TEMP"].tolist()
        wind_series = history["WSPM"].tolist()

        def safe_lag(values: List[float], lag: int) -> float:
            if len(values) >= lag:
                return float(values[-lag])
            return float(values[0])

        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            row[f"pm25_lag_{lag}"] = safe_lag(pm_series, lag)

        def safe_roll(values: List[float], window: int) -> Tuple[float, float]:
            tail = values[-window:] if len(values) >= window else values
            arr = np.asarray(tail, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        for window in [6, 12, 24, 72, 168]:
            pm_mean, pm_std = safe_roll(pm_series, window)
            temp_mean, _ = safe_roll(temp_series, window)
            wind_mean, _ = safe_roll(wind_series, window)
            row[f"pm25_roll_mean_{window}"] = pm_mean
            row[f"pm25_roll_std_{window}"] = pm_std
            row[f"temp_roll_mean_{window}"] = temp_mean
            row[f"wind_roll_mean_{window}"] = wind_mean

        row["pm25_diff_1"] = float(pm_series[-1] - pm_series[-2]) if len(pm_series) >= 2 else 0.0
        row["pm25_diff_24"] = float(pm_series[-1] - pm_series[-24]) if len(pm_series) >= 24 else row["pm25_diff_1"]
        row["temp_diff_1"] = float(temp_series[-1] - temp_series[-2]) if len(temp_series) >= 2 else 0.0
        row["wind_diff_1"] = float(wind_series[-1] - wind_series[-2]) if len(wind_series) >= 2 else 0.0

        feature_vector = pd.DataFrame([row])[artifacts.feature_columns]
        pred = float(artifacts.model.predict(feature_vector)[0])
        row["PM2.5"] = max(pred, 0.0)
        row["pm25_category"] = pm25_category(row["PM2.5"])
        row["thermal_label"] = thermal_label(float(row["TEMP"]))
        row["rain_label"] = rain_label(float(row["RAIN"]))
        row["wind_label"] = wind_label(float(row["WSPM"]))

        future_rows.append(row)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    forecast_df = pd.DataFrame(future_rows)
    return forecast_df


def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("season", as_index=False)
        .agg(
            avg_pm25=("PM2.5", "mean"),
            median_pm25=("PM2.5", "median"),
            avg_temp=("TEMP", "mean"),
            avg_rain=("RAIN", "mean"),
            avg_wind=("WSPM", "mean"),
        )
    )
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    summary["season"] = pd.Categorical(summary["season"], categories=season_order, ordered=True)
    summary = summary.sort_values("season").reset_index(drop=True)
    summary["expected_pm25_category"] = summary["avg_pm25"].apply(pm25_category)
    summary["expected_temperature_feel"] = summary["avg_temp"].apply(thermal_label)
    summary["expected_rain_condition"] = summary["avg_rain"].apply(rain_label)
    summary["expected_wind_condition"] = summary["avg_wind"].apply(wind_label)
    return summary


def future_seasonal_projection(forecast_df: pd.DataFrame) -> pd.DataFrame:
    data = forecast_df.copy()
    data["season"] = data["month"].apply(season_from_month)
    projection = (
        data.groupby("season", as_index=False)
        .agg(
            forecast_avg_pm25=("PM2.5", "mean"),
            forecast_avg_temp=("TEMP", "mean"),
            forecast_avg_rain=("RAIN", "mean"),
            forecast_avg_wind=("WSPM", "mean"),
        )
    )
    if not projection.empty:
        projection["forecast_pm25_category"] = projection["forecast_avg_pm25"].apply(pm25_category)
        projection["forecast_temperature_feel"] = projection["forecast_avg_temp"].apply(thermal_label)
        projection["forecast_rain_condition"] = projection["forecast_avg_rain"].apply(rain_label)
        projection["forecast_wind_condition"] = projection["forecast_avg_wind"].apply(wind_label)
    return projection


def save_metrics(artifacts: ModelArtifacts, output_dir: Path) -> Dict[str, float]:
    metrics = compute_metrics(artifacts.test_df["PM2.5"].values, artifacts.predictions)
    metrics_path = output_dir / "time_series_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    pred_df = artifacts.test_df[["datetime", "PM2.5"]].copy()
    pred_df["prediction"] = artifacts.predictions
    pred_df["absolute_error"] = np.abs(pred_df["PM2.5"] - pred_df["prediction"])
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    return metrics


def save_preprocessing_report(raw_df: pd.DataFrame, processed_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    report = pd.DataFrame(
        [
            {"task": "Raw rows", "value": len(raw_df)},
            {"task": "Processed rows", "value": len(processed_df)},
            {"task": "Raw columns", "value": raw_df.shape[1]},
            {"task": "Processed columns", "value": processed_df.shape[1]},
            {"task": "Missing PM2.5 before", "value": int(raw_df["PM2.5"].isna().sum())},
            {"task": "Missing PM2.5 after", "value": int(processed_df["PM2.5"].isna().sum())},
            {"task": "Unique stations before", "value": int(raw_df["station"].nunique())},
            {"task": "Unique stations after", "value": int(processed_df["station"].nunique())},
        ]
    )
    report.to_csv(output_dir / "preprocessing_report.csv", index=False)
    return report


def prove_pollutant_interdependency(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    corr = df[POLLUTANT_COLUMNS].corr()
    corr.to_csv(output_dir / "pollutant_correlation_matrix.csv")

    focus = corr.loc["PM2.5", ["PM10", "NO2", "CO", "SO2", "O3"]].sort_values(ascending=False)
    focus_df = focus.rename("correlation_with_pm25").reset_index().rename(columns={"index": "variable"})
    focus_df.to_csv(output_dir / "proof_pm25_pollutant_interdependency.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Pollutant Correlation Heatmap")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_dir / "pollutant_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    return focus_df


def prove_weather_effect(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    weather_corr = df[["PM2.5", "TEMP", "RAIN", "WSPM", "PRES", "DEWP"]].corr().loc["PM2.5"]
    weather_df = weather_corr.rename("correlation_with_pm25").reset_index().rename(columns={"index": "variable"})
    weather_df.to_csv(output_dir / "proof_pm25_weather_relationships.csv", index=False)

    for x_col in ["WSPM", "TEMP", "RAIN"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sample = df[[x_col, "PM2.5"]].dropna().copy()
        if len(sample) > 6000:
            sample = sample.sample(6000, random_state=42)
        ax.scatter(sample[x_col], sample["PM2.5"], alpha=0.35, s=12)
        ax.set_xlabel(x_col)
        ax.set_ylabel("PM2.5")
        ax.set_title(f"PM2.5 vs {x_col}")
        plt.tight_layout()
        fig.savefig(output_dir / f"pm25_vs_{x_col.lower()}.png", dpi=160)
        plt.close(fig)

    return weather_df


def save_feature_importance(artifacts: ModelArtifacts, output_dir: Path) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature": artifacts.feature_columns,
            "importance": artifacts.model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    top = importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance_top20.png", dpi=160)
    plt.close(fig)
    return importance_df


def save_seasonal_plots(df: pd.DataFrame, forecast_df: pd.DataFrame, output_dir: Path) -> None:
    order = ["Spring", "Summer", "Autumn", "Winter"]
    seasonal = df.copy()
    seasonal["season"] = pd.Categorical(seasonal["season"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = [seasonal.loc[seasonal["season"] == s, "PM2.5"].dropna().values for s in order]
    ax.boxplot(grouped, labels=order, showfliers=False)
    ax.set_title("PM2.5 Distribution by Season")
    ax.set_ylabel("PM2.5")
    plt.tight_layout()
    fig.savefig(output_dir / "pm25_by_season_boxplot.png", dpi=160)
    plt.close(fig)

    if not forecast_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(forecast_df["datetime"], forecast_df["PM2.5"])
        ax.set_title("Future PM2.5 Forecast")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Forecast PM2.5")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(output_dir / "future_pm25_forecast.png", dpi=160)
        plt.close(fig)


def build_presentation_ready_claims(
    pollutant_proof: pd.DataFrame,
    weather_proof: pd.DataFrame,
    feature_importance: pd.DataFrame,
    seasonal_hist: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: Path,
) -> pd.DataFrame:
    top_pollutants = pollutant_proof.head(3)[["variable", "correlation_with_pm25"]]
    pm10_corr = pollutant_proof.loc[pollutant_proof["variable"] == "PM10", "correlation_with_pm25"].iloc[0]
    no2_corr = pollutant_proof.loc[pollutant_proof["variable"] == "NO2", "correlation_with_pm25"].iloc[0]
    co_corr = pollutant_proof.loc[pollutant_proof["variable"] == "CO", "correlation_with_pm25"].iloc[0]
    wind_corr = weather_proof.loc[weather_proof["variable"] == "WSPM", "correlation_with_pm25"].iloc[0]
    temp_corr = weather_proof.loc[weather_proof["variable"] == "TEMP", "correlation_with_pm25"].iloc[0]
    rain_corr = weather_proof.loc[weather_proof["variable"] == "RAIN", "correlation_with_pm25"].iloc[0]

    winter_row = seasonal_hist.loc[seasonal_hist["season"] == "Winter"].iloc[0]

    claims = pd.DataFrame(
        [
            {
                "claim": "PM2.5 interdependency with PM10, NO2, CO",
                "evidence": (
                    f"Correlations with PM2.5 -> PM10={pm10_corr:.3f}, NO2={no2_corr:.3f}, CO={co_corr:.3f}. "
                    f"Top correlated variables: {', '.join(top_pollutants['variable'].tolist())}."
                ),
            },
            {
                "claim": "Weather significantly affects pollution dispersion and variation",
                "evidence": (
                    f"Correlations with PM2.5 -> WSPM={wind_corr:.3f}, TEMP={temp_corr:.3f}, RAIN={rain_corr:.3f}. "
                    "Negative wind correlation supports dispersion; seasonal plots and scatter plots show time-varying behavior."
                ),
            },
            {
                "claim": "Winter has stronger pollution levels",
                "evidence": (
                    f"Historical winter average PM2.5={winter_row['avg_pm25']:.2f}, category={winter_row['expected_pm25_category']}, "
                    f"temperature feel={winter_row['expected_temperature_feel']}."
                ),
            },
            {
                "claim": "Time-series forecasting model quality",
                "evidence": (
                    f"RandomForest time-series forecast metrics -> MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, "
                    f"R2={metrics['R2']:.3f}, MAPE={metrics['MAPE']:.2f}%"
                ),
            },
            {
                "claim": "Most influential features for forecasting",
                "evidence": "Top features: " + ", ".join(feature_importance.head(10)["feature"].tolist()),
            },
        ]
    )
    claims.to_csv(output_dir / "presentation_claims_and_evidence.csv", index=False)
    return claims


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    raw_df = load_raw_data(args.csv)
    processed_df = preprocess_data(raw_df, station=args.station)
    ts_df = add_time_series_features(processed_df)

    preprocessing_report = save_preprocessing_report(raw_df, processed_df, output_dir)
    preprocessing_report.to_csv(output_dir / "preprocessing_report.csv", index=False)

    processed_df.to_csv(output_dir / "processed_data.csv", index=False)
    ts_df.to_csv(output_dir / "time_series_feature_data.csv", index=False)

    artifacts = train_forecaster(ts_df, test_fraction=args.test_fraction, random_state=args.random_state)
    metrics = save_metrics(artifacts, output_dir)

    forecast_df = recursive_future_forecast(
        base_df=processed_df,
        feature_df=ts_df,
        artifacts=artifacts,
        horizon_hours=args.forecast_horizon_hours,
    )
    forecast_df.to_csv(output_dir / "future_forecast.csv", index=False)

    seasonal_hist = seasonal_summary(processed_df)
    seasonal_hist.to_csv(output_dir / "historical_seasonal_summary.csv", index=False)

    seasonal_future = future_seasonal_projection(forecast_df)
    seasonal_future.to_csv(output_dir / "future_seasonal_projection.csv", index=False)

    pollutant_proof = prove_pollutant_interdependency(processed_df, output_dir)
    weather_proof = prove_weather_effect(processed_df, output_dir)
    feature_importance = save_feature_importance(artifacts, output_dir)
    save_seasonal_plots(processed_df, forecast_df, output_dir)
    claims = build_presentation_ready_claims(
        pollutant_proof=pollutant_proof,
        weather_proof=weather_proof,
        feature_importance=feature_importance,
        seasonal_hist=seasonal_hist,
        metrics=metrics,
        output_dir=output_dir,
    )

    summary_lines = [
        "Beijing PM2.5 Time-Series Forecasting Pipeline Complete",
        f"Station modeled: {args.station}",
        f"Processed rows: {len(processed_df):,}",
        f"Feature rows after lag engineering: {len(ts_df):,}",
        f"Forecast horizon: {args.forecast_horizon_hours} hours",
        f"MAE: {metrics['MAE']:.4f}",
        f"RMSE: {metrics['RMSE']:.4f}",
        f"R2: {metrics['R2']:.4f}",
        f"MAPE: {metrics['MAPE']:.2f}%",
        "Top 5 forecasting features:",
    ]
    summary_lines.extend([f"  - {feat}" for feat in feature_importance.head(5)["feature"].tolist()])
    summary_lines.append("\nHistorical seasonal summary:")
    summary_lines.extend(seasonal_hist.to_string(index=False).splitlines())
    if not seasonal_future.empty:
        summary_lines.append("\nFuture seasonal projection from forecast horizon:")
        summary_lines.extend(seasonal_future.to_string(index=False).splitlines())
    summary_lines.append("\nPresentation claims and evidence:")
    summary_lines.extend(claims.to_string(index=False).splitlines())

    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
