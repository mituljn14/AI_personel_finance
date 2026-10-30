
from datetime import date

from backend.database import get_connection


VALID_FREQUENCIES = {
    "weekly",
    "monthly",
    "quarterly",
    "half_yearly",
    "yearly",
}

VALID_TYPES = {"income", "expense"}


def _validate_date(value, field_name):
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.")

    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        ) from None

    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.")


def create_recurring_transaction(
    recurring_id,
    user_id,
    category,
    expected_amount,
    frequency,
    start_date,
    description=None,
    end_date=None,
    transaction_type="expense",
    db_path=None,
):
    """Create a recurring income or expense schedule."""
    if not isinstance(recurring_id, str) or not recurring_id.strip():
        raise ValueError("A valid recurring_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    if not isinstance(category, str) or not category.strip():
        raise ValueError("A valid category is required.")

    if (
        isinstance(expected_amount, bool)
        or not isinstance(expected_amount, (int, float))
        or expected_amount < 0
    ):
        raise ValueError("expected_amount must be non-negative.")

    if frequency not in VALID_FREQUENCIES:
        raise ValueError(
            f"frequency must be one of {sorted(VALID_FREQUENCIES)}."
        )

    if transaction_type not in VALID_TYPES:
        raise ValueError("transaction_type must be income or expense.")

    _validate_date(start_date, "start_date")

    if end_date is not None:
        _validate_date(end_date, "end_date")
        if end_date < start_date:
            raise ValueError("end_date cannot be before start_date.")

    with get_connection(db_path) as connection:
        profile = connection.execute(
            "SELECT 1 FROM user_profiles WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchone()

        if profile is None:
            raise ValueError("Create the user profile first.")

        connection.execute(
            """
            INSERT INTO recurring_transactions (
                recurring_id, user_id, category, description,
                expected_amount, frequency, start_date, end_date,
                transaction_type, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                recurring_id.strip(),
                user_id.strip(),
                category.strip(),
                description,
                float(expected_amount),
                frequency,
                start_date,
                end_date,
                transaction_type,
            ),
        )

    return get_recurring_transaction(
        recurring_id, db_path=db_path
    )


def get_recurring_transaction(recurring_id, db_path=None):
    """Retrieve a recurring transaction by ID."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM recurring_transactions
            WHERE recurring_id = ?
            """,
            (recurring_id,),
        ).fetchone()

    return dict(row) if row else None


def list_recurring_transactions(
    user_id,
    active_only=True,
    db_path=None,
):
    """List recurring schedules for a user."""
    query = """
        SELECT *
        FROM recurring_transactions
        WHERE user_id = ?
    """

    if active_only:
        query += " AND is_active = 1"

    query += " ORDER BY start_date, recurring_id"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, (user_id,)).fetchall()

    return [dict(row) for row in rows]


def set_recurring_transaction_active(
    recurring_id,
    is_active,
    db_path=None,
):
    """Activate or deactivate a recurring schedule."""
    if not isinstance(is_active, bool):
        raise ValueError("is_active must be True or False.")

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE recurring_transactions
            SET is_active = ?
            WHERE recurring_id = ?
            """,
            (int(is_active), recurring_id),
        )

        if cursor.rowcount == 0:
            return None

    return get_recurring_transaction(
        recurring_id, db_path=db_path
    )


def delete_recurring_transaction(recurring_id, db_path=None):
    """Delete a recurring schedule. Returns True if deleted."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            DELETE FROM recurring_transactions
            WHERE recurring_id = ?
            """,
            (recurring_id,),
        )

    return cursor.rowcount > 0
