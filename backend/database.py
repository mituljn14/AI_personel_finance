
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "personal_finance.db"


@contextmanager
def get_connection(db_path=None):
    """Yield a database connection and always close it."""
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(db_path=None):
    """Initialize database tables, migrate existing transactions, and create indexes."""
    with get_connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                transaction_date TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL CHECK (amount >= 0),
                currency TEXT NOT NULL DEFAULT 'INR',
                category TEXT NOT NULL,
                subcategory TEXT,
                description TEXT,
                payment_method TEXT,
                merchant_name TEXT,
                payment_status TEXT DEFAULT 'completed',
                data_source TEXT DEFAULT 'manual_entry',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add new transaction columns safely to existing databases.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(transactions)"
            ).fetchall()
        }

        migrations = {
            "is_essential": "INTEGER",
            "related_transaction_id": "TEXT",
        }

        for column, definition in migrations.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE transactions ADD COLUMN {column} {definition}"
                )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                age INTEGER CHECK (age IS NULL OR age >= 0),
                household_size INTEGER CHECK (
                    household_size IS NULL OR household_size >= 1
                ),
                monthly_income_estimate REAL CHECK (
                    monthly_income_estimate IS NULL
                    OR monthly_income_estimate >= 0
                ),
                employment_type TEXT,
                state TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS monthly_budgets (
                budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                year_month TEXT NOT NULL,
                category TEXT NOT NULL,
                budget_amount REAL NOT NULL CHECK (budget_amount >= 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, year_month, category),
                FOREIGN KEY (user_id)
                    REFERENCES user_profiles(user_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS recurring_transactions (
                recurring_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                expected_amount REAL NOT NULL CHECK (expected_amount >= 0),
                frequency TEXT NOT NULL CHECK (
                    frequency IN (
                        'weekly', 'monthly', 'quarterly',
                        'half_yearly', 'yearly'
                    )
                ),
                start_date TEXT NOT NULL,
                end_date TEXT,
                transaction_type TEXT NOT NULL DEFAULT 'expense'
                    CHECK (transaction_type IN ('income', 'expense')),
                is_active INTEGER NOT NULL DEFAULT 1
                    CHECK (is_active IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id)
                    REFERENCES user_profiles(user_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_date
            ON transactions(transaction_date)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_category
            ON transactions(category)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_user_date
            ON transactions(user_id, transaction_date)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_budgets_user_month
            ON monthly_budgets(user_id, year_month)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_recurring_user_dates
            ON recurring_transactions(user_id, start_date, end_date)
        """)


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized at: {DB_PATH}")
