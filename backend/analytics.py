
from backend.database import get_connection


def get_financial_summary(user_id=None, start_date=None, end_date=None, db_path=None):
    """Calculate total income, expenses, and net savings.

    Transfers are excluded from income and expense totals.
    """
    query = """
        SELECT
            COALESCE(SUM(CASE
                WHEN transaction_type = 'income' THEN amount
                ELSE 0
            END), 0) AS total_income,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'expense' THEN amount
                ELSE 0
            END), 0) AS total_expenses
        FROM transactions
        WHERE 1 = 1
    """
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    if start_date:
        query += " AND transaction_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND transaction_date <= ?"
        params.append(end_date)

    with get_connection(db_path) as connection:
        row = connection.execute(query, params).fetchone()

    income = float(row["total_income"])
    expenses = float(row["total_expenses"])

    return {
        "total_income": income,
        "total_expenses": expenses,
        "net_savings": income - expenses,
    }


def get_category_expenses(user_id=None, start_date=None, end_date=None, db_path=None):
    """Return expense totals grouped by category."""
    query = """
        SELECT category, SUM(amount) AS total
        FROM transactions
        WHERE transaction_type = 'expense'
    """
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    if start_date:
        query += " AND transaction_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND transaction_date <= ?"
        params.append(end_date)

    query += " GROUP BY category ORDER BY total DESC"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return [
        {"category": row["category"], "total": float(row["total"])}
        for row in rows
    ]


def get_monthly_summary(user_id=None, db_path=None):
    """Return monthly income and expense totals."""
    query = """
        SELECT
            substr(transaction_date, 1, 7) AS month,
            SUM(CASE
                WHEN transaction_type = 'income' THEN amount
                ELSE 0
            END) AS total_income,
            SUM(CASE
                WHEN transaction_type = 'expense' THEN amount
                ELSE 0
            END) AS total_expenses
        FROM transactions
        WHERE 1 = 1
    """
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    query += " GROUP BY substr(transaction_date, 1, 7) ORDER BY month"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return [
        {
            "month": row["month"],
            "total_income": float(row["total_income"] or 0),
            "total_expenses": float(row["total_expenses"] or 0),
        }
        for row in rows
    ]
