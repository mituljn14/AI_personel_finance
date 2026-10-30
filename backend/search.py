
from datetime import date

from backend.database import get_connection


def search_transactions(
    user_id=None,
    category=None,
    transaction_type=None,
    start_date=None,
    end_date=None,
    keyword=None,
    db_path=None,
):
    """Search transactions using optional filters."""

    if transaction_type and transaction_type not in {
        "income", "expense", "transfer"
    }:
        raise ValueError("Invalid transaction type.")

    for value in (start_date, end_date):
        if value:
            try:
                date.fromisoformat(value)
            except (TypeError, ValueError):
                raise ValueError("Dates must use YYYY-MM-DD format.") from None

    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date cannot be after end date.")

    query = "SELECT * FROM transactions WHERE 1 = 1"
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    if category:
        query += " AND category = ?"
        params.append(category)

    if transaction_type:
        query += " AND transaction_type = ?"
        params.append(transaction_type)

    if start_date:
        query += " AND transaction_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND transaction_date <= ?"
        params.append(end_date)

    if keyword:
        query += " AND (description LIKE ? OR merchant_name LIKE ?)"
        pattern = f"%{keyword}%"
        params.extend([pattern, pattern])

    query += " ORDER BY transaction_date DESC"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return [dict(row) for row in rows]
