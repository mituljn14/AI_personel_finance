
from datetime import date
from uuid import uuid4

from backend.database import get_connection


def add_transaction(
    user_id,
    transaction_date,
    transaction_type,
    amount,
    category,
    description="",
    currency="INR",
    subcategory="",
    payment_method="",
    merchant_name="",
    db_path=None,
):
    """Add a transaction and return its generated ID."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("User ID is required.")

    if transaction_type not in {"income", "expense", "transfer"}:
        raise ValueError("Transaction type must be income, expense, or transfer.")

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError("Amount must be a valid number.") from None

    if amount < 0:
        raise ValueError("Amount cannot be negative.")

    try:
        date.fromisoformat(transaction_date)
    except (TypeError, ValueError):
        raise ValueError("Date must use YYYY-MM-DD format.") from None

    transaction_id = f"MAN-{uuid4().hex[:12].upper()}"

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO transactions (
                transaction_id, user_id, transaction_date,
                transaction_type, amount, currency, category,
                subcategory, description, payment_method, merchant_name,
                data_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual_entry')
            """,
            (
                transaction_id, user_id.strip(), transaction_date,
                transaction_type, amount, currency, category,
                subcategory, description, payment_method, merchant_name,
            ),
        )

    return transaction_id


def get_transactions(user_id=None, db_path=None):
    """Return transactions, optionally filtered by user ID."""
    with get_connection(db_path) as connection:
        if user_id:
            rows = connection.execute(
                """
                SELECT * FROM transactions
                WHERE user_id = ?
                ORDER BY transaction_date DESC
                """,
                (user_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM transactions ORDER BY transaction_date DESC"
            ).fetchall()

    return [dict(row) for row in rows]


def update_transaction(transaction_id, db_path=None, **fields):
    """Update supported fields of an existing transaction."""
    allowed_fields = {
        "transaction_date", "transaction_type", "amount", "category",
        "subcategory", "description", "payment_method", "merchant_name",
        "payment_status",
    }

    updates = {key: value for key, value in fields.items() if key in allowed_fields}
    if not updates:
        raise ValueError("No valid fields supplied for update.")

    if "transaction_type" in updates and updates["transaction_type"] not in {
        "income", "expense", "transfer"
    }:
        raise ValueError("Invalid transaction type.")

    if "amount" in updates:
        try:
            updates["amount"] = float(updates["amount"])
        except (TypeError, ValueError):
            raise ValueError("Amount must be a valid number.") from None

        if updates["amount"] < 0:
            raise ValueError("Amount cannot be negative.")

    if "transaction_date" in updates:
        try:
            date.fromisoformat(updates["transaction_date"])
        except (TypeError, ValueError):
            raise ValueError("Date must use YYYY-MM-DD format.") from None

    assignments = ", ".join(f"{field} = ?" for field in updates)
    values = list(updates.values()) + [transaction_id]

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            f"UPDATE transactions SET {assignments} WHERE transaction_id = ?",
            values,
        )

    return cursor.rowcount > 0


def delete_transaction(transaction_id, db_path=None):
    """Delete a transaction and report whether it existed."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM transactions WHERE transaction_id = ?",
            (transaction_id,),
        )

    return cursor.rowcount > 0
