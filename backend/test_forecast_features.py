
import tempfile
import unittest
from pathlib import Path

from backend.database import initialize_database
from backend.profiles import save_user_profile
from backend.transactions import add_transaction
from backend.feature_engineering import build_forecast_features
from backend.forecasting import forecast_user_expenses

class TestForecastFeatures(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "forecast_test.db"
        initialize_database(self.db_path)

        save_user_profile(
            user_id="TEST_USER",
            age=25,
            household_size=3,
            monthly_income_estimate=45000,
            employment_type="Salaried",
            state="Madhya Pradesh",
            db_path=self.db_path,
        )

        # Add completed transactions across four consecutive months.
        for month, income, expense in [
            ("2026-07", 40000, 18000),
            ("2026-08", 42000, 20000),
            ("2026-09", 43000, 19000),
            ("2026-10", 45000, 22000),
        ]:
            add_transaction(
                user_id="TEST_USER",
                transaction_date=f"{month}-05",
                transaction_type="income",
                amount=income,
                category="Salary",
                db_path=self.db_path,
            )
            add_transaction(
                user_id="TEST_USER",
                transaction_date=f"{month}-10",
                transaction_type="expense",
                amount=expense,
                category="Housing",
                db_path=self.db_path,
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_builds_forecast_features(self):
        features = build_forecast_features(
            "TEST_USER",
            "2026-11",
            db_path=self.db_path,
        )

        self.assertEqual(features["age"], 25)
        self.assertEqual(features["employment_type"], "Salaried")
        self.assertEqual(features["target_month"], 11)
        self.assertEqual(features["exp_m0"], 22000.0)
        self.assertEqual(features["cat_housing_m0"], 22000.0)

        print("\nGenerated feature count:", len(features))
        print("\nMissing model features:")
        expected_features = [
            "age", "household_size", "monthly_income_estimate", "employment_type",
            "exp_m0", "exp_lag1", "exp_lag2", "exp_lag3",
            "exp_target_month_last_year", "exp_roll_mean_3", "exp_roll_mean_6",
            "exp_roll_mean_12", "exp_roll_std_6", "history_months",
            "essential_exp_m0", "discretionary_exp_m0", "income_m0",
            "income_roll_mean_3", "income_roll_mean_12", "savings_m0",
            "txn_count_m0", "avg_txn_amount_m0",
            *[f"cat_{c}_m0" for c in [
                "housing", "food", "transportation", "utilities", "healthcare",
                "education", "shopping", "entertainment", "personal_care",
                "travel", "gifts_donations", "financial_expenses"
            ]],
            *[f"cat_{c}_roll3" for c in [
                "housing", "food", "transportation", "utilities", "healthcare",
                "education", "shopping", "entertainment", "personal_care",
                "travel", "gifts_donations", "financial_expenses"
            ]],
            "budget_total_m0", "budget_total_target", "budget_utilization_m0",
            "categories_over_budget_m0", "expected_recurring_target",
            "target_month", "target_month_sin", "target_month_cos",
            "target_festival_index", "target_is_festive_peak",
            "target_is_summer_travel", "target_is_year_end",
            "target_is_academic_fee_month",
        ]
        print(sorted(set(expected_features) - set(features)))
        print("\nUnexpected features:")
        print(sorted(set(features) - set(expected_features)))

    def test_generates_next_month_prediction(self):
        result = forecast_user_expenses(
            user_id="TEST_USER",
            target_month="2026-11",
            db_path=self.db_path,
        )

        self.assertEqual(result["user_id"], "TEST_USER")
        self.assertEqual(result["target_month"], "2026-11")
        self.assertIn("model_name", result)
        self.assertIn("predicted_net_expenses", result)
        self.assertEqual(result["currency"], "INR")

        self.assertIsInstance(
            result["predicted_net_expenses"],
            float,
        )

        self.assertGreaterEqual(
            result["predicted_net_expenses"],
            0,
        )

        print("\nEnd-to-end forecast result:")
        print(result)

if __name__ == "__main__":
    unittest.main()
