
from datetime import date

import pandas as pd

from backend.database import get_connection


def _validate_through_month(through_month):
    """Validate an optional YYYY-MM month filter."""
    if through_month is None:
        return

    if not isinstance(through_month, str):
        raise ValueError("through_month must use YYYY-MM format.")

    try:
        parsed_month = date.fromisoformat(through_month + "-01")
    except ValueError:
        raise ValueError(
            "through_month must use YYYY-MM format."
        ) from None

    if parsed_month.strftime("%Y-%m") != through_month:
        raise ValueError("through_month must use YYYY-MM format.")


def get_monthly_transaction_summary(
    user_id,
    db_path=None,
    through_month=None,
):
    """
    Calculate monthly income, gross expenses, refunds and net expenses
    from completed SQLite transactions.

    Missing months between the first and last transaction month are
    included with zero values. through_month uses YYYY-MM format.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    _validate_through_month(through_month)

    query = """
        SELECT
            substr(transaction_date, 1, 7) AS year_month,
            SUM(CASE
                WHEN transaction_type = 'income' THEN amount
                ELSE 0
            END) AS total_income,
            SUM(CASE
                WHEN transaction_type = 'expense' THEN amount
                ELSE 0
            END) AS total_expenses,
            SUM(CASE
                WHEN transaction_type = 'refund' THEN amount
                ELSE 0
            END) AS refunds_received,
            COUNT(CASE
                WHEN transaction_type = 'expense' THEN 1
            END) AS expense_count,
            COUNT(*) AS transaction_count
        FROM transactions
        WHERE user_id = ?
          AND payment_status = 'completed'
          AND transaction_type IN (
              'income', 'expense', 'refund', 'transfer'
          )
    """

    params = [user_id.strip()]

    if through_month is not None:
        query += " AND substr(transaction_date, 1, 7) <= ?"
        params.append(through_month)

    query += """
        GROUP BY substr(transaction_date, 1, 7)
        ORDER BY year_month
    """

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    columns = [
        "year_month",
        "total_income",
        "total_expenses",
        "refunds_received",
        "expense_count",
        "transaction_count",
    ]

    summary = pd.DataFrame(
        [dict(row) for row in rows],
        columns=columns,
    )

    # Preserve the empty result for users without transactions.
    if summary.empty:
        summary["net_expenses"] = pd.Series(dtype=float)
        summary["net_cash_flow"] = pd.Series(dtype=float)
        summary["average_transaction_amount"] = pd.Series(dtype=float)
        return summary

    # Insert missing calendar months between observed months.
    summary = summary.sort_values("year_month")
    summary = summary.set_index("year_month")

    calendar = pd.period_range(
        start=summary.index.min(),
        end=summary.index.max(),
        freq="M",
    ).astype(str)

    summary = summary.reindex(calendar, fill_value=0)
    summary.index.name = "year_month"
    summary = summary.reset_index()

    summary["net_expenses"] = (
        summary["total_expenses"] - summary["refunds_received"]
    )

    summary["net_cash_flow"] = (
        summary["total_income"] - summary["net_expenses"]
    )

    summary["average_transaction_amount"] = (
        summary["total_expenses"]
        .div(summary["expense_count"].replace(0, float("nan")))
        .fillna(0)
    )

    return summary


def build_expense_history_features(monthly_summary):
    """
    Prepare history-based features from monthly transaction summaries.

    Missing calendar months should already be represented as zero
    values by get_monthly_transaction_summary().
    """
    required_columns = {
        "year_month",
        "net_expenses",
        "total_income",
        "total_expenses",
        "expense_count",
        "transaction_count",
    }

    missing = required_columns - set(monthly_summary.columns)

    if missing:
        raise ValueError(
            f"Monthly summary is missing columns: {sorted(missing)}"
        )

    if monthly_summary.empty:
        raise ValueError("No monthly history is available.")

    history = monthly_summary.copy()

    history["year_month"] = pd.to_datetime(
        history["year_month"],
        format="%Y-%m",
        errors="raise",
    )

    history = (
        history.sort_values("year_month")
        .drop_duplicates(subset="year_month", keep="last")
        .reset_index(drop=True)
    )

    latest = history.iloc[-1]
    expenses = history["net_expenses"].astype(float)
    income = history["total_income"].astype(float)

    result = {
        "exp_m0": float(latest["net_expenses"]),
        "exp_lag1": (
            float(expenses.iloc[-2])
            if len(expenses) >= 2 else float("nan")
        ),
        "exp_lag2": (
            float(expenses.iloc[-3])
            if len(expenses) >= 3 else float("nan")
        ),
        "exp_lag3": (
            float(expenses.iloc[-4])
            if len(expenses) >= 4 else float("nan")
        ),
        "exp_target_month_last_year": (
            float(expenses.iloc[-12])
            if len(expenses) >= 12 else float("nan")
        ),
        "exp_roll_mean_3": float(expenses.tail(3).mean()),
        "exp_roll_mean_6": float(expenses.tail(6).mean()),
        "exp_roll_mean_12": float(expenses.tail(12).mean()),
        "exp_roll_std_6": (
            float(expenses.tail(6).std(ddof=1))
            if len(expenses.tail(6)) >= 2 else float("nan")
        ),
        "history_months": len(history),
        "income_m0": float(latest["total_income"]),
        "income_roll_mean_3": float(income.tail(3).mean()),
        "income_roll_mean_12": float(income.tail(12).mean()),
        "txn_count_m0": float(latest["transaction_count"]),
        "avg_txn_amount_m0": (
            float(latest["total_expenses"] / latest["expense_count"])
            if latest["expense_count"] > 0 else 0.0
        ),
    }

    return result


def get_monthly_category_expenses(
    user_id,
    db_path=None,
    through_month=None,
):
    """Return net expense totals by category for each month."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    _validate_through_month(through_month)

    query = """
        SELECT
            substr(transaction_date, 1, 7) AS year_month,
            category,
            SUM(
                CASE
                    WHEN transaction_type = 'expense' THEN amount
                    WHEN transaction_type = 'refund' THEN -amount
                    ELSE 0
                END
            ) AS net_amount
        FROM transactions
        WHERE user_id = ?
          AND payment_status = 'completed'
          AND transaction_type IN ('expense', 'refund')
    """

    params = [user_id.strip()]

    if through_month is not None:
        query += " AND substr(transaction_date, 1, 7) <= ?"
        params.append(through_month)

    query += """
        GROUP BY substr(transaction_date, 1, 7), category
        ORDER BY year_month, category
    """

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=["year_month", "category", "net_amount"],
    )
