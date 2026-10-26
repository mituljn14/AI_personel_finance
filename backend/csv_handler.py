
import csv
from datetime import date
from pathlib import Path

from backend.database import get_connection


CSV_FIELDS = [
    "transaction_id",
    "user_id",
    "transaction_date",
    "transaction_type",
    "amount",
    "currency",
    "category",
    "subcategory",
    "description",
    "payment_method",
    "merchant_name",
    "payment_status",
]


def export_transactions(file_path, user_id=None, db_path=None):
    """Export transactions from SQLite to CSV."""
    query = """
        SELECT transaction_id, user_id, transaction_date,
               transaction_type, amount, currency, category,
               subcategory, description, payment_method,
               merchant_name, payment_status
        FROM transactions
    """
    params = []

    if user_id:
        query += " WHERE user_id = ?"
        params.append(user_id)

    query += " ORDER BY transaction_date DESC"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])

    return len(rows)


def import_transactions(file_path, db_path=None):
    """Import CSV records, skipping duplicates and invalid rows."""
    imported = 0
    skipped = 0
    required_fields = set(CSV_FIELDS)

    with open(file_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError("CSV file is missing required transaction columns.")

        with get_connection(db_path) as connection:
            for row in reader:
                transaction_id = (row.get("transaction_id") or "").strip()
                user_id = (row.get("user_id") or "").strip()
                transaction_date = (row.get("transaction_date") or "").strip()
                transaction_type = (row.get("transaction_type") or "").strip()
                category = (row.get("category") or "").strip()

                if not all([
                    transaction_id, user_id, transaction_date,
                    transaction_type, category,
                ]):
                    skipped += 1
                    continue

                if transaction_type not in {"income", "expense", "transfer"}:
                    skipped += 1
                    continue

                try:
                    date.fromisoformat(transaction_date)
                    amount = float(row["amount"])
                    if amount < 0:
                        skipped += 1
                        continue
                except (TypeError, ValueError):
                    skipped += 1
                    continue

                existing = connection.execute(
                    "SELECT 1 FROM transactions WHERE transaction_id = ?",
                    (transaction_id,),
                ).fetchone()

                if existing:
                    skipped += 1
                    continue

                connection.execute(
                    """
                    INSERT INTO transactions (
                        transaction_id, user_id, transaction_date,
                        transaction_type, amount, currency, category,
                        subcategory, description, payment_method,
                        merchant_name, payment_status, data_source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'csv_import')
                    """,
                    (
                        transaction_id,
                        user_id,
                        transaction_date,
                        transaction_type,
                        amount,
                        (row.get("currency") or "INR").strip(),
                        category,
                        (row.get("subcategory") or "").strip(),
                        (row.get("description") or "").strip(),
                        (row.get("payment_method") or "").strip(),
                        (row.get("merchant_name") or "").strip(),
                        (row.get("payment_status") or "completed").strip(),
                    ),
                )
                imported += 1

    return {"imported": imported, "skipped": skipped}
