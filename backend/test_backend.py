from backend.duplicate_detector import find_possible_duplicates
from backend.analytics import (
    get_financial_summary,
    get_category_expenses,
    get_monthly_summary,
)
import csv
import tempfile
import unittest
from pathlib import Path

from backend.database import initialize_database, get_connection
from backend.transactions import (
    add_transaction,
    get_transactions,
    update_transaction,
    delete_transaction,
)
from backend.search import search_transactions
from backend.csv_handler import export_transactions, import_transactions


class BackendTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_finance.db"
        initialize_database(self.db_path)

        self.user_id = "TEST_USER"
        self.transaction_id = add_transaction(
            user_id=self.user_id,
            transaction_date="2026-01-15",
            transaction_type="expense",
            amount=500,
            category="Food",
            description="Test meal",
            merchant_name="Test Cafe",
            db_path=self.db_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_read(self):
        transactions = get_transactions(
            self.user_id, db_path=self.db_path
        )
        self.assertTrue(any(
            row["transaction_id"] == self.transaction_id
            for row in transactions
        ))

    def test_search_by_category(self):
        results = search_transactions(
            user_id=self.user_id,
            category="Food",
            db_path=self.db_path,
        )
        self.assertTrue(any(
            row["transaction_id"] == self.transaction_id
            for row in results
        ))

    def test_update(self):
        updated = update_transaction(
            self.transaction_id,
            db_path=self.db_path,
            amount=750,
            description="Updated meal",
        )
        self.assertTrue(updated)

        transactions = get_transactions(
            self.user_id, db_path=self.db_path
        )
        row = next(
            item for item in transactions
            if item["transaction_id"] == self.transaction_id
        )
        self.assertEqual(row["amount"], 750)
        self.assertEqual(row["description"], "Updated meal")

    def test_delete(self):
        self.assertTrue(delete_transaction(
            self.transaction_id, db_path=self.db_path
        ))
        self.assertFalse(delete_transaction(
            self.transaction_id, db_path=self.db_path
        ))

    def test_csv_export_and_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "transactions.csv"

            exported = export_transactions(
                csv_path, self.user_id, db_path=self.db_path
            )
            self.assertEqual(exported, 1)

            with csv_path.open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["transaction_id"], self.transaction_id
            )

            # Importing the same transaction should skip its duplicate ID.
            result = import_transactions(
                csv_path, db_path=self.db_path
            )
            self.assertEqual(result["imported"], 0)
            self.assertEqual(result["skipped"], 1)

    def test_invalid_amount(self):
        with self.assertRaises(ValueError):
            add_transaction(
                user_id=self.user_id,
                transaction_date="2026-01-15",
                transaction_type="expense",
                amount=-100,
                category="Food",
                db_path=self.db_path,
            )

    def test_invalid_transaction_type(self):
        with self.assertRaises(ValueError):
            add_transaction(
                user_id=self.user_id,
                transaction_date="2026-01-15",
                transaction_type="unknown",
                amount=100,
                category="Food",
                db_path=self.db_path,
            )
            
    
    def test_financial_summary(self):
        add_transaction(
            user_id=self.user_id,
            transaction_date="2026-01-16",
            transaction_type="income",
            amount=2000,
            category="Salary",
            db_path=self.db_path,
        )

        add_transaction(
            user_id=self.user_id,
            transaction_date="2026-01-17",
            transaction_type="transfer",
            amount=300,
            category="Savings",
            db_path=self.db_path,
        )

        summary = get_financial_summary(
            user_id=self.user_id,
            db_path=self.db_path,
        )

        self.assertEqual(summary["total_income"], 2000)
        self.assertEqual(summary["total_expenses"], 500)
        self.assertEqual(summary["net_savings"], 1500)

    def test_category_expenses(self):
        categories = get_category_expenses(
            user_id=self.user_id,
            db_path=self.db_path,
        )

        self.assertEqual(
            categories,
            [{"category": "Food", "total": 500.0}],
        )

    def test_monthly_summary(self):
        summaries = get_monthly_summary(
            user_id=self.user_id,
            db_path=self.db_path,
        )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["month"], "2026-01")
        self.assertEqual(summaries[0]["total_expenses"], 500.0)
       
    
    def test_duplicate_detection(self):
        add_transaction(
            user_id=self.user_id,
            transaction_date="2026-01-15",
            transaction_type="expense",
            amount=500,
            category="Food",
            description="Test meal",
            merchant_name="Test Cafe",
            db_path=self.db_path,
        )

        duplicates = find_possible_duplicates(
            user_id=self.user_id,
            db_path=self.db_path,
        )

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(duplicates[0]["transactions"]), 2)

    
    def test_rejects_invalid_date_format(self):
        with self.assertRaises(ValueError):
            add_transaction(
                user_id="user1",
                transaction_date="2026-1-05",
                transaction_type="expense",
                amount=100,
                category="Food",
                db_path=self.db_path,
            )

    def test_rejects_nan_amount(self):
        with self.assertRaises(ValueError):
            add_transaction(
                user_id="user1",
                transaction_date="2026-01-05",
                transaction_type="expense",
                amount=float("nan"),
                category="Food",
                db_path=self.db_path,
            )

    def test_rejects_infinite_amount(self):
        with self.assertRaises(ValueError):
            add_transaction(
                user_id="user1",
                transaction_date="2026-01-05",
                transaction_type="expense",
                amount=float("inf"),
                category="Food",
                db_path=self.db_path,
            )

    def test_rejects_empty_category(self):
        with self.assertRaises(ValueError):
            add_transaction(
                user_id="user1",
                transaction_date="2026-01-05",
                transaction_type="expense",
                amount=100,
                category="   ",
                db_path=self.db_path,
            )

    
    def test_csv_import_new_transaction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "new_transactions.csv"

            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "transaction_id", "user_id", "transaction_date",
                        "transaction_type", "amount", "currency",
                        "category", "subcategory", "description",
                        "payment_method", "merchant_name", "payment_status",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "transaction_id": "CSV_NEW_1",
                    "user_id": self.user_id,
                    "transaction_date": "2026-02-10",
                    "transaction_type": "expense",
                    "amount": "250.50",
                    "currency": "INR",
                    "category": "Food",
                    "subcategory": "Groceries",
                    "description": "Imported groceries",
                    "payment_method": "UPI",
                    "merchant_name": "Local Store",
                    "payment_status": "completed",
                })

            result = import_transactions(csv_path, db_path=self.db_path)

            self.assertEqual(result, {"imported": 1, "skipped": 0})

            transactions = get_transactions(
                self.user_id, db_path=self.db_path
            )
            imported = next(
                row for row in transactions
                if row["transaction_id"] == "CSV_NEW_1"
            )
            self.assertEqual(imported["amount"], 250.50)

    def test_csv_import_rejects_non_finite_amounts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "invalid_amounts.csv"

            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "transaction_id", "user_id", "transaction_date",
                    "transaction_type", "amount", "currency", "category",
                    "subcategory", "description", "payment_method",
                    "merchant_name", "payment_status",
                ])
                writer.writeheader()

                for transaction_id, amount in [
                    ("CSV_NAN", "nan"),
                    ("CSV_INF", "inf"),
                    ("CSV_NEG_INF", "-inf"),
                ]:
                    writer.writerow({
                        "transaction_id": transaction_id,
                        "user_id": self.user_id,
                        "transaction_date": "2026-02-10",
                        "transaction_type": "expense",
                        "amount": amount,
                        "currency": "INR",
                        "category": "Food",
                        "payment_status": "completed",
                    })

            result = import_transactions(csv_path, db_path=self.db_path)

            self.assertEqual(result, {"imported": 0, "skipped": 3})

    def test_csv_import_rejects_malformed_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "invalid_date.csv"

            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "transaction_id", "user_id", "transaction_date",
                    "transaction_type", "amount", "currency", "category",
                    "subcategory", "description", "payment_method",
                    "merchant_name", "payment_status",
                ])
                writer.writeheader()
                writer.writerow({
                    "transaction_id": "CSV_BAD_DATE",
                    "user_id": self.user_id,
                    "transaction_date": "2026-2-10",
                    "transaction_type": "expense",
                    "amount": "100",
                    "currency": "INR",
                    "category": "Food",
                    "payment_status": "completed",
                })

            result = import_transactions(csv_path, db_path=self.db_path)

            self.assertEqual(result, {"imported": 0, "skipped": 1})

    def test_csv_import_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "missing_columns.csv"
            csv_path.write_text(
                "transaction_id,user_id,amount\n"
                "CSV_1,TEST_USER,100\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                import_transactions(csv_path, db_path=self.db_path)

    def test_duplicate_detector_returns_empty_when_no_duplicates(self):
        duplicates = find_possible_duplicates(
            user_id=self.user_id,
            db_path=self.db_path,
        )

        self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
