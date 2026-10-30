
import tempfile
import unittest
from pathlib import Path

from backend.database import initialize_database
from backend.profiles import save_user_profile
from backend.recurring_transactions import (
    create_recurring_transaction,
    get_recurring_transaction,
    list_recurring_transactions,
    set_recurring_transaction_active,
    delete_recurring_transaction,
)


class TestRecurringTransactions(unittest.TestCase):
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

    def create_sample(self, recurring_id="rec1"):
        return create_recurring_transaction(
            recurring_id=recurring_id,
            user_id="user1",
            category="Housing",
            expected_amount=8000,
            frequency="monthly",
            start_date="2026-01-01",
            description="Monthly rent",
            transaction_type="expense",
            db_path=self.db_path,
        )

    def test_create_and_retrieve(self):
        created = self.create_sample()

        self.assertEqual(created["recurring_id"], "rec1")
        self.assertEqual(created["category"], "Housing")
        self.assertEqual(created["expected_amount"], 8000)
        self.assertEqual(
            get_recurring_transaction("rec1", self.db_path)["frequency"],
            "monthly",
        )

    def test_list_active_transactions(self):
        self.create_sample()
        results = list_recurring_transactions(
            "user1", db_path=self.db_path
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["recurring_id"], "rec1")

    def test_deactivate_and_reactivate(self):
        self.create_sample()

        result = set_recurring_transaction_active(
            "rec1", False, db_path=self.db_path
        )
        self.assertEqual(result["is_active"], 0)

        active = list_recurring_transactions(
            "user1", db_path=self.db_path
        )
        self.assertEqual(active, [])

        set_recurring_transaction_active(
            "rec1", True, db_path=self.db_path
        )
        active = list_recurring_transactions(
            "user1", db_path=self.db_path
        )
        self.assertEqual(len(active), 1)

    def test_delete_transaction(self):
        self.create_sample()

        self.assertTrue(
            delete_recurring_transaction("rec1", db_path=self.db_path)
        )
        self.assertIsNone(
            get_recurring_transaction("rec1", db_path=self.db_path)
        )
        self.assertFalse(
            delete_recurring_transaction("rec1", db_path=self.db_path)
        )

    def test_invalid_frequency_is_rejected(self):
        with self.assertRaises(ValueError):
            create_recurring_transaction(
                recurring_id="rec2",
                user_id="user1",
                category="Housing",
                expected_amount=8000,
                frequency="daily",
                start_date="2026-01-01",
                db_path=self.db_path,
            )

    def test_end_date_before_start_date_is_rejected(self):
        with self.assertRaises(ValueError):
            create_recurring_transaction(
                recurring_id="rec2",
                user_id="user1",
                category="Housing",
                expected_amount=8000,
                frequency="monthly",
                start_date="2026-05-01",
                end_date="2026-04-01",
                db_path=self.db_path,
            )

    def test_unknown_user_is_rejected(self):
        with self.assertRaises(ValueError):
            create_recurring_transaction(
                recurring_id="rec2",
                user_id="unknown",
                category="Housing",
                expected_amount=8000,
                frequency="monthly",
                start_date="2026-01-01",
                db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main()
