from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "extracted"
OUTPUT = ROOT / "outputs"
TARGET = "target_next_month_expenses"
SPLITS = ("train", "validation", "test")
EXCLUDED_FEATURES = {
    "user_id",
    "year",
    "month",
    "split",
    "target_year",
    TARGET,
}


def validate_and_save_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Validate the generated forecast table and save clean, unscaled splits."""
    forecast = pd.read_csv(SOURCE / "monthly_expense_forecasting.csv")
    summary = pd.read_csv(SOURCE / "monthly_financial_summary.csv")

    required = EXCLUDED_FEATURES | {"target_month", "employment_type"}
    missing_columns = sorted(required - set(forecast.columns))
    if missing_columns:
        raise ValueError(f"Forecast table is missing required columns: {missing_columns}")

    if forecast[["user_id", "year", "month"]].duplicated().any():
        raise ValueError("Duplicate user/month forecast records found.")
    if forecast[TARGET].isna().any() or not np.isfinite(forecast[TARGET]).all():
        raise ValueError("Forecast target contains missing or non-finite values.")
    if not forecast["split"].isin(SPLITS).all():
        raise ValueError("Forecast table contains an unknown split label.")

    expected_split = np.select(
        [forecast["target_year"] <= 2023, forecast["target_year"] == 2024],
        ["train", "validation"],
        default="test",
    )
    if not np.array_equal(forecast["split"].to_numpy(), expected_split):
        raise ValueError("Split labels do not match the target-year chronology.")
    if (forecast["target_year"] > 2025).any() or (forecast["target_year"] < 2022).any():
        raise ValueError("Forecast target year is outside the expected 2022–2025 period.")

    # Confirm that each forecast label still matches its source accounting summary.
    expected_targets = summary[
        ["user_id", "year", "month", "net_expenses"]
    ].rename(
        columns={
            "year": "target_year",
            "month": "target_month",
            "net_expenses": "summary_target",
        }
    )
    reconciled = forecast.merge(
        expected_targets,
        on=["user_id", "target_year", "target_month"],
        how="left",
        validate="one_to_one",
    )
    if reconciled["summary_target"].isna().any():
        raise ValueError("Some forecast targets have no matching accounting summary.")
    if not np.allclose(
        reconciled[TARGET], reconciled["summary_target"], atol=0.011, rtol=0
    ):
        raise ValueError("Forecast targets do not reconcile to monthly net expenses.")

    feature_columns = [column for column in forecast.columns if column not in EXCLUDED_FEATURES]
    numeric_features = forecast[feature_columns].select_dtypes(include=np.number)
    if np.isinf(numeric_features.to_numpy(dtype=float)).any():
        raise ValueError("Forecast features contain infinite values.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT / "forecast_clean.csv", index=False)
    split_frames = {split: forecast.loc[forecast["split"] == split].copy() for split in SPLITS}
    for split, frame in split_frames.items():
        frame.to_csv(OUTPUT / f"forecast_{split}.csv", index=False)

    quality_report = {
        "input": "data/extracted/monthly_expense_forecasting.csv",
        "rows": int(len(forecast)),
        "rows_removed": 0,
        "duplicate_user_month_rows": 0,
        "missing_targets": 0,
        "non_finite_targets": 0,
        "infinite_numeric_features": 0,
        "targets_reconciled_to_monthly_summary": int(len(reconciled)),
        "negative_targets_retained": int((forecast[TARGET] < 0).sum()),
        "negative_target_note": (
            "Valid net-expense outcomes: linked refunds exceeded expenses in those months."
        ),
        "split_rows": {split: int(len(frame)) for split, frame in split_frames.items()},
        "missing_feature_values": {
            column: int(count)
            for column, count in forecast[feature_columns].isna().sum().items()
            if count
        },
        "feature_columns": feature_columns,
    }
    (OUTPUT / "forecast_data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
    return forecast, split_frames, quality_report


def make_pipeline(model, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "MAE_INR": float(mean_absolute_error(actual, predicted)),
        "RMSE_INR": float(mean_squared_error(actual, predicted) ** 0.5),
        "rows": int(len(actual)),
    }


def main() -> None:
    forecast, splits, quality_report = validate_and_save_data()
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    feature_columns = [column for column in forecast.columns if column not in EXCLUDED_FEATURES]
    categorical_features = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(forecast[column])
    ]
    numeric_features = [
        column for column in feature_columns if column not in categorical_features
    ]

    candidates = {
        "12-month historical average": None,
        "LinearRegression": LinearRegression(),
        "ExtraTrees absolute error": ExtraTreesRegressor(
            n_estimators=220,
            criterion="absolute_error",
            max_depth=16,
            min_samples_leaf=3,
            max_features=0.9,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoosting Huber": GradientBoostingRegressor(
            loss="huber",
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ),
    }
    boosting_configs = [
        (0.05, 20, 200, 5),
        (0.08, 10, 200, 1),
        (0.03, 15, 350, 5),
        (0.03, 10, 500, 1),
        (0.03, 10, 350, 1),
        (0.05, 10, 200, 5),
        (0.03, 15, 200, 5),
        (0.08, 10, 350, 1),
        (0.03, 10, 350, 5),
        (0.05, 15, 200, 5),
        (0.05, 10, 350, 5),
        (0.05, 20, 250, 5),
    ]
    for learning_rate, leaf_nodes, iterations, regularization in boosting_configs:
        name = (
            f"HistGradientBoosting absolute lr={learning_rate} "
            f"leaves={leaf_nodes} iterations={iterations} l2={regularization}"
        )
        candidates[name] = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=learning_rate,
            max_iter=iterations,
            max_leaf_nodes=leaf_nodes,
            l2_regularization=regularization,
            early_stopping=False,
            random_state=42,
        )

    results: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in candidates.items():
        if model is None:
            prediction = validation["exp_roll_mean_12"].fillna(
                train[TARGET].median()
            ).to_numpy()
        else:
            pipeline = make_pipeline(model, numeric_features, categorical_features)
            pipeline.fit(train[feature_columns], train[TARGET])
            prediction = pipeline.predict(validation[feature_columns])
            fitted[name] = pipeline
        score = metrics(validation[TARGET], prediction)
        results.append(
            {
                "model": name,
                "validation_MAE_INR": score["MAE_INR"],
                "validation_RMSE_INR": score["RMSE_INR"],
                "validation_rows": score["rows"],
            }
        )

    leaderboard = pd.DataFrame(results).sort_values(
        "validation_MAE_INR", kind="stable"
    )
    leaderboard.to_csv(OUTPUT / "forecast_validation_model_comparison.csv", index=False)
    best_name = str(leaderboard.iloc[0]["model"])
    if best_name not in fitted:
        raise RuntimeError(
            "The historical-average baseline won validation; a fitted model cannot be selected."
        )
    best_estimator = candidates[best_name]

    # Refit the validation-selected estimator on all pre-test data; test is used once below.
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = make_pipeline(best_estimator, numeric_features, categorical_features)
    final_model.fit(train_validation[feature_columns], train_validation[TARGET])
    test_prediction = final_model.predict(test[feature_columns])
    test_score = metrics(test[TARGET], test_prediction)

    baseline_prediction = test["exp_roll_mean_12"].fillna(
        train_validation[TARGET].median()
    ).to_numpy()
    baseline_score = metrics(test[TARGET], baseline_prediction)
    test_metrics = pd.DataFrame(
        [
            {"model": best_name, **test_score},
            {"model": "12-month historical average", **baseline_score},
        ]
    )
    test_metrics.to_csv(OUTPUT / "forecast_test_metrics.csv", index=False)

    predictions = test[
        ["user_id", "year", "month", "target_year", "target_month", TARGET]
    ].copy()
    predictions["predicted_net_expenses"] = test_prediction
    predictions["absolute_error"] = np.abs(
        predictions[TARGET] - predictions["predicted_net_expenses"]
    )
    predictions.to_csv(OUTPUT / "forecast_test_predictions.csv", index=False)

    joblib.dump(
        {
            "pipeline": final_model,
            "model_name": best_name,
            "feature_columns": feature_columns,
            "target_column": TARGET,
            "validation_MAE_INR": float(leaderboard.iloc[0]["validation_MAE_INR"]),
            "test_MAE_INR": test_score["MAE_INR"],
        },
        OUTPUT / "best_forecast_model.joblib",
    )
    quality_report["selected_model"] = best_name
    quality_report["validation_MAE_INR"] = float(
        leaderboard.iloc[0]["validation_MAE_INR"]
    )
    quality_report["test_MAE_INR"] = test_score["MAE_INR"]
    (OUTPUT / "forecast_data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )

    print("Validation leaderboard (lowest MAE first):")
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nFinal held-out 2025 test metrics:")
    print(test_metrics.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSaved clean data, chronological splits, model and metrics in: {OUTPUT}")


if __name__ == "__main__":
    main()
