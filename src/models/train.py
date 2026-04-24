from __future__ import annotations

from pathlib import Path
import json
import joblib
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("disaster-recovery-cost-prediction")

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "processed_disasters.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH)
    return df


def define_features(df: pd.DataFrame):
    target_col = "target_log_total_obligated"

    numeric_features = [
        "incident_duration_days",
        "declaration_year",
        "declaration_month",
        "state_5yr_disaster_count",
        "project_count",
        "avg_project_amount",
    ]

    categorical_features = [
        "state",
        "incidentType",
        "declarationType",
        "region",
        "season",
        "high_cost_incident",
    ]

    # Keep only columns that actually exist
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    feature_cols = numeric_features + categorical_features

    clean_df = df[feature_cols + [target_col]].copy()
    clean_df = clean_df.dropna(subset=[target_col])

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    return X, y, numeric_features, categorical_features, target_col


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> dict:
    models = {
        "LinearRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", XGBRegressor(
                    n_estimators=300,
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=-1,
                )),
            ]
        ),
    }
    return models


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    rows = []

    for name, pipeline in models.items():
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "model": name,
            "mean_r2": np.mean(cv_results["test_r2"]),
            "std_r2": np.std(cv_results["test_r2"]),
            "mean_rmse": -np.mean(cv_results["test_rmse"]),
            "std_rmse": np.std(-cv_results["test_rmse"]),
            "mean_mae": -np.mean(cv_results["test_mae"]),
            "std_mae": np.std(-cv_results["test_mae"]),
        }
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)
    return results


def fit_best_model(best_model_name: str, models: dict, X: pd.DataFrame, y: pd.Series):
    best_pipeline = models[best_model_name]
    best_pipeline.fit(X, y)
    return best_pipeline


def log_to_mlflow(results: pd.DataFrame, best_model_name: str, best_pipeline, X: pd.DataFrame):
    mlflow.set_experiment("disaster-recovery-cost-prediction")

    with mlflow.start_run(run_name="baseline_training_pipeline"):
        for _, row in results.iterrows():
            prefix = row["model"].lower()
            mlflow.log_metric(f"{prefix}_mean_r2", float(row["mean_r2"]))
            mlflow.log_metric(f"{prefix}_mean_rmse", float(row["mean_rmse"]))
            mlflow.log_metric(f"{prefix}_mean_mae", float(row["mean_mae"]))

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("target", "target_log_total_obligated")
        mlflow.log_param("n_rows", int(X.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))

        signature_input = X.head(5)
        signature_output = best_pipeline.predict(signature_input)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="best_model_pipeline",
        )


def save_best_model(best_pipeline, results: pd.DataFrame, feature_info: dict) -> Path:
    out_path = MODELS_DIR / "best_model.pkl"

    payload = {
        "model_pipeline": best_pipeline,
        "cv_results": results.to_dict(orient="records"),
        "feature_info": feature_info,
    }

    joblib.dump(payload, out_path)
    return out_path


def main():
    print("Loading processed data...")
    df = load_data()
    print("Dataset shape:", df.shape)

    print("Defining features and target...")
    X, y, numeric_features, categorical_features, target_col = define_features(df)

    print("Building preprocessor...")
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print("Building models...")
    models = build_models(preprocessor)

    print("Running 5-fold cross-validation...")
    results = evaluate_models(models, X, y)

    print("\nCross-validation results:")
    print(results)

    best_model_name = results.iloc[0]["model"]
    print(f"\nBest model by mean CV R²: {best_model_name}")

    print("Fitting best model on full dataset...")
    best_pipeline = fit_best_model(best_model_name, models, X, y)

    print("Logging run to MLflow...")
    log_to_mlflow(results, best_model_name, best_pipeline, X)

    feature_info = {
        "target": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    print("Saving best model artifact...")
    model_path = save_best_model(best_pipeline, results, feature_info)

    print(f"\nBest model saved to: {model_path}")
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()