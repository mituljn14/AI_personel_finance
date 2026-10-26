
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
    """Initialize the selected database and its indexes."""
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

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_date
            ON transactions(transaction_date)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_category
            ON transactions(category)
        """)


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized at: {DB_PATH}")
