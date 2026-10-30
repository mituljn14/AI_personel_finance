
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_forecast_model.joblib"


def load_forecast_model(model_path=None):
    """Load the trained forecasting model and its metadata."""
    path = Path(model_path) if model_path else MODEL_PATH

    if not path.is_file():
        raise FileNotFoundError(
            f"Forecast model not found: {path}"
        )

    artifact = joblib.load(path)

    required_keys = {"pipeline", "model_name", "feature_columns"}
    missing_keys = required_keys - set(artifact)

    if missing_keys:
        raise ValueError(
            f"Model artifact is missing metadata: {sorted(missing_keys)}"
        )

    return artifact


def predict_next_month(feature_row, model_path=None):
    """
    Predict next-month net expenses from one prepared feature row.

    feature_row must contain every feature expected by the saved model.
    This function does not calculate missing features or invent defaults.
    """
    artifact = load_forecast_model(model_path)
    feature_columns = artifact["feature_columns"]

    if isinstance(feature_row, dict):
        input_data = pd.DataFrame([feature_row])
    elif isinstance(feature_row, pd.DataFrame):
        input_data = feature_row.copy()
    else:
        raise TypeError(
            "feature_row must be a dictionary or pandas DataFrame."
        )

    if len(input_data) != 1:
        raise ValueError(
            "Exactly one feature row is required per prediction."
        )

    missing = [
        column for column in feature_columns
        if column not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    # Preserve the exact feature order used during training.
    input_data = input_data[feature_columns]

    prediction = float(
        artifact["pipeline"].predict(input_data)[0]
    )

    return {
        "model_name": artifact["model_name"],
        "predicted_net_expenses": round(prediction, 2),
        "currency": "INR",
    }



def forecast_user_expenses(
    user_id,
    target_month,
    db_path=None,
    model_path=None,
):
    """
    Generate a next-month expense forecast for a user.

    Builds the required features from the user's financial data
    and passes them to the trained forecasting model.
    """
    from backend.feature_engineering import build_forecast_features

    # Step 1: Generate all required model features.
    feature_row = build_forecast_features(
        user_id=user_id,
        target_month=target_month,
        db_path=db_path,
    )

    # Step 2: Predict next-month net expenses.
    prediction = predict_next_month(
        feature_row=feature_row,
        model_path=model_path,
    )

    # Step 3: Include the forecast month in the result.
    return {
        "user_id": user_id,
        "target_month": target_month,
        **prediction,
    }

