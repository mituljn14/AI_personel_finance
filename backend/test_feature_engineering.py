import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.database import initialize_database
from backend.transactions import add_transaction
from backend.feature_engineering import get_monthly_transaction_summary
from backend.feature_engineering import build_expense_history_features
from backend.feature_engineering import get_monthly_category_expenses

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        initialize_database(self.db_path)

        add_transaction(
            user_id="TEST_USER",
            transaction_date="2026-09-05",
            transaction_type="income",
            amount=30000,
            category="Salary",
            db_path=self.db_path,
        )
        add_transaction(
            user_id="TEST_USER",
            transaction_date="2026-09-10",
            transaction_type="expense",
            amount=8000,
            category="Housing",
            db_path=self.db_path,
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO transactions (
                    transaction_id, user_id, transaction_date,
                    transaction_type, amount, currency, category,
                    payment_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "REFUND-001",
                    "TEST_USER",
                    "2026-09-15",
                    "refund",
                    500,
                    "INR",
                    "Housing",
                    "completed",
                ),
            )
        
        add_transaction(
            user_id="TEST_USER",
            transaction_date="2026-10-03",
            transaction_type="expense",
            amount=2000,
            category="Food",
            db_path=self.db_path,
        )

    def tearDown(self):
        gc.collect()

        for suffix in ("", "-journal", "-wal", "-shm"):
            db_file = Path(str(self.db_path) + suffix)

            if db_file.exists():
                try:
                    db_file.unlink()
                except PermissionError:
                    pass
        self.temp_dir.cleanup()

    def test_monthly_totals_and_refunds(self):
        summary = get_monthly_transaction_summary(
            "TEST_USER", db_path=self.db_path
        )

        september = summary.loc[
            summary["year_month"] == "2026-09"
        ].iloc[0]

        self.assertEqual(september["total_income"], 30000)
        self.assertEqual(september["total_expenses"], 8000)
        self.assertEqual(september["refunds_received"], 500)
        self.assertEqual(september["net_expenses"], 7500)
        self.assertEqual(september["net_cash_flow"], 22500)

    def test_month_filter(self):
        summary = get_monthly_transaction_summary(
            "TEST_USER",
            db_path=self.db_path,
            through_month="2026-09",
        )

        self.assertEqual(summary["year_month"].tolist(), ["2026-09"])

    def test_unknown_user_returns_empty_summary(self):
        summary = get_monthly_transaction_summary(
            "UNKNOWN_USER", db_path=self.db_path
        )

        self.assertTrue(summary.empty)
        
    
    def test_builds_historical_features(self):
        import pandas as pd

        history = pd.DataFrame({
            "year_month": [
                "2026-06", "2026-07", "2026-08", "2026-09"
            ],
            "net_expenses": [4000, 5000, 6000, 7500],
            "total_income": [25000, 26000, 27000, 30000],
            "total_expenses": [4000, 5000, 6000, 8000],
            "expense_count": [4, 5, 6, 8],
            "transaction_count": [6, 7, 8, 10],
        })

        features = build_expense_history_features(history)

        self.assertEqual(features["exp_m0"], 7500.0)
        self.assertEqual(features["exp_lag1"], 6000.0)
        self.assertEqual(features["exp_lag2"], 5000.0)
        self.assertEqual(features["exp_lag3"], 4000.0)
        self.assertEqual(features["history_months"], 4)
        self.assertEqual(features["income_m0"], 30000.0)
        self.assertEqual(features["txn_count_m0"], 10.0)
        self.assertEqual(features["avg_txn_amount_m0"], 1000.0)
        

    def test_rejects_empty_history(self):
        import pandas as pd

        history = pd.DataFrame(columns=[
            "year_month",
            "net_expenses",
            "total_income",
            "total_expenses",
            "expense_count",
            "transaction_count",
        ])

        with self.assertRaisesRegex(ValueError, "No monthly history"):
            build_expense_history_features(history)

    def test_rejects_missing_history_columns(self):
        import pandas as pd

        history = pd.DataFrame({
            "year_month": ["2026-09"],
            "net_expenses": [7500],
        })

        with self.assertRaisesRegex(ValueError, "missing columns"):
            build_expense_history_features(history)

    def test_monthly_category_expenses_net_of_refunds(self):
        result = get_monthly_category_expenses(
            "TEST_USER",
            db_path=self.db_path,
        )

        housing = result[
            (result["year_month"] == "2026-09")
            & (result["category"] == "Housing")
        ]

        self.assertEqual(len(housing), 1)
        self.assertEqual(housing.iloc[0]["net_amount"], 7500)

if __name__ == "__main__":
    unittest.main()
