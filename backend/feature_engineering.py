import math
from datetime import date, datetime, timedelta

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


EXPENSE_CATEGORIES = {
    "Housing": "housing",
    "Food": "food",
    "Transportation": "transportation",
    "Utilities": "utilities",
    "Healthcare": "healthcare",
    "Education": "education",
    "Shopping": "shopping",
    "Entertainment": "entertainment",
    "Personal Care": "personal_care",
    "Travel": "travel",
    "Gifts & Donations": "gifts_donations",
    "Financial Expenses": "financial_expenses",
}


def build_category_features(category_history, current_month):
    """Build current-month net spending and trailing 3-month averages."""
    if not isinstance(current_month, str):
        raise ValueError("current_month must use YYYY-MM format.")

    result = {}

    for category, slug in EXPENSE_CATEGORIES.items():
        feature_name = f"cat_{slug}"

        if category_history.empty:
            monthly = pd.Series(dtype=float)
        else:
            monthly = (
                category_history.loc[
                    category_history["category"] == category,
                    ["year_month", "net_amount"],
                ]
                .drop_duplicates("year_month", keep="last")
                .set_index("year_month")["net_amount"]
                .astype(float)
            )

        # Include the three calendar months ending at current_month.
        period = pd.Period(current_month, freq="M")
        months = [str(period - offset) for offset in (2, 1, 0)]
        trailing = monthly.reindex(months, fill_value=0.0)

        result[f"{feature_name}_m0"] = float(trailing.iloc[-1])
        result[f"{feature_name}_roll3"] = float(trailing.mean())

    return result



def build_monthly_detail_features(user_id, current_month, db_path=None):
    """Build essential, discretionary, and savings features for one month."""
    from backend.database import get_connection

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                transaction_type,
                category,
                subcategory,
                amount,
                is_essential
            FROM transactions
            WHERE user_id = ?
              AND substr(transaction_date, 1, 7) = ?
              AND payment_status = 'completed'
            """,
            (user_id, current_month),
        ).fetchall()

    essential_expenses = 0.0
    discretionary_expenses = 0.0
    savings = 0.0

    for row in rows:
        transaction_type = str(row["transaction_type"]).lower()
        category = str(row["category"] or "")
        subcategory = str(row["subcategory"] or "")
        amount = float(row["amount"] or 0.0)

        if transaction_type == "expense":
            if row["is_essential"] == 1:
                essential_expenses += amount
            else:
                discretionary_expenses += amount

        elif transaction_type == "transfer" and category == "Savings and Investments":
            if subcategory == "savings_withdrawal":
                savings -= amount
            else:
                savings += amount

    return {
        "essential_exp_m0": essential_expenses,
        "discretionary_exp_m0": discretionary_expenses,
        "savings_m0": savings,
    }


def build_budget_features(user_id, current_month, target_month, db_path=None):
    """Build budget features using net category spending."""
    from backend.database import get_connection

    with get_connection(db_path) as conn:
        budget_rows = conn.execute(
            """
            SELECT year_month, category, budget_amount
            FROM monthly_budgets
            WHERE user_id = ?
              AND year_month IN (?, ?)
            """,
            (user_id, current_month, target_month),
        ).fetchall()

    current_budgets = {}
    target_budgets = {}

    for row in budget_rows:
        month = row["year_month"]
        category = row["category"]
        amount = float(row["budget_amount"] or 0.0)

        if month == current_month:
            current_budgets[category] = amount
        elif month == target_month:
            target_budgets[category] = amount

    with get_connection(db_path) as conn:
        spending_rows = conn.execute(
            """
            SELECT
                category,
                SUM(
                    CASE
                        WHEN lower(transaction_type) = 'expense'
                            THEN amount
                        WHEN lower(transaction_type) = 'refund'
                            THEN -amount
                        ELSE 0
                    END
                ) AS net_spending
            FROM transactions
            WHERE user_id = ?
              AND substr(transaction_date, 1, 7) = ?
              AND payment_status = 'completed'
            GROUP BY category
            """,
            (user_id, current_month),
        ).fetchall()

    net_spending = {
        row["category"]: float(row["net_spending"] or 0.0)
        for row in spending_rows
    }

    total_budget = sum(current_budgets.values())
    target_budget = sum(target_budgets.values())

    budgeted_categories = [
        category
        for category, amount in current_budgets.items()
        if amount > 0
    ]

    if budgeted_categories:
        budgeted_spending = sum(
            net_spending.get(category, 0.0)
            for category in budgeted_categories
        )
        budget_utilization = budgeted_spending / sum(
            current_budgets[category]
            for category in budgeted_categories
        )
    else:
        budget_utilization = 0.0

    categories_over_budget = sum(
        1
        for category in budgeted_categories
        if net_spending.get(category, 0.0) > current_budgets[category]
    )

    return {
        "budget_total_m0": float(total_budget),
        "budget_total_target": float(target_budget),
        "budget_utilization_m0": float(budget_utilization),
        "categories_over_budget_m0": int(categories_over_budget),
    }

FESTIVALS = {
    "Diwali": ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"],
    "Ganesh Chaturthi": ["2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27"],
    "Durga Puja": ["2022-10-03", "2023-10-22", "2024-10-11", "2025-09-30"],
    "Onam": ["2022-09-08", "2023-08-29", "2024-09-15", "2025-09-05"],
    "Pongal": ["2022-01-14", "2023-01-15", "2024-01-15", "2025-01-14"],
    "Holi": ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14"],
    "Dussehra": ["2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02"],
}

REGIONAL = {
    "Maharashtra": "Ganesh Chaturthi",
    "Gujarat": "Ganesh Chaturthi",
    "Goa": "Ganesh Chaturthi",
    "West Bengal": "Durga Puja",
    "Odisha": "Durga Puja",
    "Assam": "Durga Puja",
    "Kerala": "Onam",
    "Tamil Nadu": "Pongal",
    "Delhi": "Holi",
    "Uttar Pradesh": "Holi",
    "Rajasthan": "Holi",
    "Punjab": "Holi",
    "Haryana": "Holi",
    "Chandigarh": "Holi",
    "Madhya Pradesh": "Holi",
    "Bihar": "Holi",
    "Karnataka": "Dussehra",
    "Telangana": "Dussehra",
    "Andhra Pradesh": "Dussehra",
    "Chhattisgarh": "Dussehra",
}


def _festival_month_weights(festival_name, year, month):
    """Calculate the share of festival shopping windows in a month."""
    month_start = date(year, month, 1)

    if month == 12:
        next_month_start = date(year + 1, 1, 1)
    else:
        next_month_start = date(year, month + 1, 1)

    month_end = next_month_start - timedelta(days=1)
    days_in_month = (month_end - month_start).days + 1
    total_weight = 0.0

    for date_text in FESTIVALS[festival_name]:
        festival_date = datetime.strptime(date_text, "%Y-%m-%d").date()
        window_start = festival_date - timedelta(days=21)
        window_end = festival_date + timedelta(days=2)

        overlap_start = max(window_start, month_start)
        overlap_end = min(window_end, month_end)

        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days + 1
            total_weight += overlap_days / 24.0

    return total_weight


def build_calendar_features(target_month, state="Madhya Pradesh"):
    """Build target-month calendar and approximate festival features."""
    try:
        parsed = datetime.strptime(target_month, "%Y-%m")
        if parsed.strftime("%Y-%m") != target_month:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("target_month must use YYYY-MM format.")

    month = parsed.month
    regional_festival = REGIONAL.get(state, "Dussehra")

    diwali_weight = _festival_month_weights(
        "Diwali", parsed.year, month
    )
    regional_weight = _festival_month_weights(
        regional_festival, parsed.year, month
    )

    festival_index = min(
        diwali_weight + 0.6 * regional_weight,
        1.0,
    )

    angle = 2 * math.pi * month / 12

    return {
        "target_month": month,
        "target_month_sin": round(math.sin(angle), 6),
        "target_month_cos": round(math.cos(angle), 6),
        "target_festival_index": round(festival_index, 4),
        "target_is_festive_peak": int(festival_index >= 0.5),
        "target_is_summer_travel": int(month in (4, 5, 6)),
        "target_is_year_end": int(month == 12),
        "target_is_academic_fee_month": int(month in (6, 7)),
    }


def get_forecast_profile_features(user_id, db_path=None):
    """Fetch the user profile required for expense forecasting."""
    from backend.database import get_connection

    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                age,
                household_size,
                monthly_income_estimate,
                employment_type,
                state
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        raise ValueError(
            f"No saved profile found for user_id={user_id!r}."
        )

    return {
        "age": row["age"],
        "household_size": row["household_size"],
        "monthly_income_estimate": row["monthly_income_estimate"],
        "employment_type": row["employment_type"],
        "state": row["state"] or "Madhya Pradesh",
    }


def build_forecast_features(user_id, target_month, db_path=None):
    """Assemble the currently implemented features for a forecast."""
    from datetime import datetime
    import pandas as pd

    try:
        parsed = datetime.strptime(target_month, "%Y-%m")
        if parsed.strftime("%Y-%m") != target_month:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("target_month must use YYYY-MM format.")

    # The prediction month must follow the latest completed month.
    current_period = pd.Period(target_month, freq="M") - 1
    current_month = str(current_period)

    profile = get_forecast_profile_features(user_id, db_path=db_path)

    monthly_summary = get_monthly_transaction_summary(
        user_id,
        db_path=db_path,
        through_month=current_month,
    )
    history_features = build_expense_history_features(monthly_summary)

    category_history = get_monthly_category_expenses(
        user_id,
        db_path=db_path,
        through_month=current_month,
    )
    category_features = build_category_features(
        category_history,
        current_month,
    )

    detail_features = build_monthly_detail_features(
        user_id,
        current_month,
        db_path=db_path,
    )

    budget_features = build_budget_features(
        user_id,
        current_month,
        target_month,
        db_path=db_path,
    )

    calendar_features = build_calendar_features(
        target_month,
        state=profile["state"],
    )
    
    recurring_features = build_expected_recurring_feature(
        user_id,
        target_month,
        db_path=db_path,
    )

    features = {
        "age": profile["age"],
        "household_size": profile["household_size"],
        "monthly_income_estimate": profile["monthly_income_estimate"],
        "employment_type": profile["employment_type"],
        **history_features,
        **category_features,
        **detail_features,
        **budget_features,
        **calendar_features,
        **recurring_features,
    }

    return features


def build_expected_recurring_feature(
    user_id,
    target_month,
    db_path=None,
):
    """Estimate recurring expenses scheduled during the target month."""
    from datetime import date
    from calendar import monthrange
    from backend.database import get_connection

    try:
        parsed = date.fromisoformat(f"{target_month}-01")
        if parsed.strftime("%Y-%m") != target_month:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("target_month must use YYYY-MM format.")

    month_start = parsed
    month_end = date(
        parsed.year,
        parsed.month,
        monthrange(parsed.year, parsed.month)[1],
    )

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT expected_amount, frequency, start_date, end_date
            FROM recurring_transactions
            WHERE user_id = ?
              AND is_active = 1
              AND transaction_type = 'expense'
              AND start_date <= ?
              AND (end_date IS NULL OR end_date >= ?)
            """,
            (user_id, month_end.isoformat(), month_start.isoformat()),
        ).fetchall()

    total = 0.0

    for row in rows:
        amount = float(row["expected_amount"] or 0.0)
        frequency = row["frequency"]

        if frequency == "weekly":
            # Approximate number of weekly occurrences in this month.
            total += amount * (month_end.day / 7.0)

        elif frequency == "monthly":
            total += amount

        elif frequency == "quarterly":
            total += amount / 3.0

        elif frequency == "half_yearly":
            total += amount / 6.0

        elif frequency == "yearly":
            total += amount / 12.0

    return {"expected_recurring_target": float(total)}
