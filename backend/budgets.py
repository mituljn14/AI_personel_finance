
from datetime import date

from backend.database import get_connection


def _validate_month(year_month):
    if not isinstance(year_month, str):
        raise ValueError("year_month must use YYYY-MM format.")

    try:
        parsed = date.fromisoformat(year_month + "-01")
    except ValueError:
        raise ValueError("year_month must use YYYY-MM format.") from None

    if parsed.strftime("%Y-%m") != year_month:
        raise ValueError("year_month must use YYYY-MM format.")


def save_monthly_budget(
    user_id,
    year_month,
    category,
    budget_amount,
    db_path=None,
):
    """Insert or update a user's budget for one category and month."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    _validate_month(year_month)

    if not isinstance(category, str) or not category.strip():
        raise ValueError("A valid category is required.")

    if (
        isinstance(budget_amount, bool)
        or not isinstance(budget_amount, (int, float))
        or budget_amount < 0
    ):
        raise ValueError("budget_amount must be non-negative.")

    with get_connection(db_path) as connection:
        profile = connection.execute(
            "SELECT 1 FROM user_profiles WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchone()

        if profile is None:
            raise ValueError("Create the user profile before saving budgets.")

        connection.execute(
            """
            INSERT INTO monthly_budgets (
                user_id, year_month, category, budget_amount
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, year_month, category)
            DO UPDATE SET budget_amount = excluded.budget_amount
            """,
            (
                user_id.strip(),
                year_month,
                category.strip(),
                float(budget_amount),
            ),
        )

    return get_monthly_budgets(user_id, year_month, db_path=db_path)


def get_monthly_budgets(user_id, year_month, db_path=None):
    """Return category budgets for a user and month."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    _validate_month(year_month)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT category, budget_amount
            FROM monthly_budgets
            WHERE user_id = ? AND year_month = ?
            ORDER BY category
            """,
            (user_id.strip(), year_month),
        ).fetchall()

    return [dict(row) for row in rows]


def get_budget_summary(user_id, year_month, db_path=None):
    """Calculate total budget and utilization for a given month."""
    budgets = get_monthly_budgets(user_id, year_month, db_path=db_path)
    total_budget = sum(item["budget_amount"] for item in budgets)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT category, SUM(amount) AS spent
            FROM transactions
            WHERE user_id = ?
              AND substr(transaction_date, 1, 7) = ?
              AND payment_status = 'completed'
              AND transaction_type = 'expense'
            GROUP BY category
            """,
            (user_id.strip(), year_month),
        ).fetchall()

    spending = {row["category"]: row["spent"] for row in rows}

    actual_budgeted_spending = sum(
        spending.get(item["category"], 0.0)
        for item in budgets
    )

    over_budget_count = sum(
        1
        for item in budgets
        if spending.get(item["category"], 0.0) > item["budget_amount"]
    )

    utilization = (
        actual_budgeted_spending / total_budget * 100
        if total_budget > 0 else None
    )

    return {
        "year_month": year_month,
        "total_budget": total_budget,
        "actual_budgeted_spending": actual_budgeted_spending,
        "budget_utilization_percentage": utilization,
        "categories_over_budget": over_budget_count,
    }
