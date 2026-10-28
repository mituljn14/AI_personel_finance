
import tempfile
import unittest
from pathlib import Path

from backend.database import initialize_database
from backend.profiles import save_user_profile
from backend.budgets import (
    save_monthly_budget,
    get_monthly_budgets,
    get_budget_summary,
)


class TestBudgets(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        initialize_database(self.db_path)

        save_user_profile(
            user_id="user1",
            age=22,
            household_size=3,
            monthly_income_estimate=30000,
            employment_type="Student",
            state="Madhya Pradesh",
            db_path=self.db_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_budget(self):
        save_monthly_budget(
            "user1", "2026-10", "Food", 5000, db_path=self.db_path
        )
        budgets = get_monthly_budgets(
            "user1", "2026-10", db_path=self.db_path
        )

        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0]["category"], "Food")
        self.assertEqual(budgets[0]["budget_amount"], 5000)

    def test_saving_same_budget_updates_it(self):
        save_monthly_budget(
            "user1", "2026-10", "Food", 5000, db_path=self.db_path
        )
        save_monthly_budget(
            "user1", "2026-10", "Food", 6500, db_path=self.db_path
        )

        budgets = get_monthly_budgets(
            "user1", "2026-10", db_path=self.db_path
        )
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0]["budget_amount"], 6500)

    def test_invalid_budget_amount_is_rejected(self):
        with self.assertRaises(ValueError):
            save_monthly_budget(
                "user1", "2026-10", "Food", -100, db_path=self.db_path
            )

    def test_budget_requires_user_profile(self):
        with self.assertRaises(ValueError):
            save_monthly_budget(
                "unknown", "2026-10", "Food", 1000, db_path=self.db_path
            )

    def test_budget_summary(self):
        save_monthly_budget(
            "user1", "2026-10", "Food", 5000, db_path=self.db_path
        )
        save_monthly_budget(
            "user1", "2026-10", "Transport", 2000, db_path=self.db_path
        )

        with __import__("backend.database", fromlist=["get_connection"]).get_connection(
            self.db_path
        ) as connection:
            connection.execute(
                """
                INSERT INTO transactions (
                    transaction_id, user_id, transaction_date,
                    transaction_type, amount, category, payment_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "txn1", "user1", "2026-10-05",
                    "expense", 6000, "Food", "completed"
                ),
            )
            connection.execute(
                """
                INSERT INTO transactions (
                    transaction_id, user_id, transaction_date,
                    transaction_type, amount, category, payment_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "txn2", "user1", "2026-10-06",
                    "expense", 1000, "Transport", "completed"
                ),
            )

        summary = get_budget_summary(
            "user1", "2026-10", db_path=self.db_path
        )

        self.assertEqual(summary["total_budget"], 7000)
        self.assertEqual(summary["actual_budgeted_spending"], 7000)
        self.assertEqual(summary["categories_over_budget"], 1)
        self.assertAlmostEqual(
            summary["budget_utilization_percentage"], 100.0
        )


if __name__ == "__main__":
    unittest.main()
