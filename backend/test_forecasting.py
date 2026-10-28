
import unittest
from pathlib import Path

import pandas as pd

from backend.forecasting import (
    load_forecast_model,
    predict_next_month,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_forecast_model.joblib"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "extracted"
    / "monthly_expense_forecasting.csv"
)


class TestForecasting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.artifact = load_forecast_model()
        cls.feature_columns = cls.artifact["feature_columns"]

        # Use synthetic historical data only to test model compatibility.
        data = pd.read_csv(DATA_PATH)
        cls.feature_row = (
            data.iloc[-1][cls.feature_columns].to_dict()
        )

    def test_model_loads(self):
        self.assertIn("pipeline", self.artifact)
        self.assertIn("model_name", self.artifact)
        self.assertTrue(self.artifact["feature_columns"])

    def test_prediction_returns_expected_structure(self):
        result = predict_next_month(self.feature_row)

        self.assertEqual(
            set(result.keys()),
            {
                "model_name",
                "predicted_net_expenses",
                "currency",
            },
        )
        self.assertEqual(result["currency"], "INR")
        self.assertIsInstance(
            result["predicted_net_expenses"], float
        )

    def test_missing_features_are_rejected(self):
        incomplete_row = dict(self.feature_row)
        incomplete_row.pop(self.feature_columns[0])

        with self.assertRaisesRegex(ValueError, "Missing required"):
            predict_next_month(incomplete_row)

    def test_multiple_rows_are_rejected(self):
        two_rows = pd.DataFrame(
            [self.feature_row, self.feature_row]
        )

        with self.assertRaisesRegex(
            ValueError, "Exactly one feature row"
        ):
            predict_next_month(two_rows)


if __name__ == "__main__":
    unittest.main()
