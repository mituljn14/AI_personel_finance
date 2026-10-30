
from backend.database import get_connection


VALID_EMPLOYMENT_TYPES = {
    "Salaried",
    "Freelancer",
    "Business Owner",
    "Gig/Part-time",
    "Student",
    "Retired",
}


def save_user_profile(
    user_id,
    age,
    household_size,
    monthly_income_estimate,
    employment_type,
    state,
    db_path=None,
):
    """Insert or update a user's forecasting profile."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise ValueError("age must be a non-negative integer.")

    if (
        isinstance(household_size, bool)
        or not isinstance(household_size, int)
        or household_size < 1
    ):
        raise ValueError("household_size must be a positive integer.")

    if (
        isinstance(monthly_income_estimate, bool)
        or not isinstance(monthly_income_estimate, (int, float))
        or monthly_income_estimate < 0
    ):
        raise ValueError("monthly_income_estimate must be non-negative.")

    if employment_type not in VALID_EMPLOYMENT_TYPES:
        raise ValueError(
            f"employment_type must be one of: "
            f"{sorted(VALID_EMPLOYMENT_TYPES)}"
        )

    if not isinstance(state, str) or not state.strip():
        raise ValueError("A valid state is required.")

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_profiles (
                user_id, age, household_size,
                monthly_income_estimate, employment_type, state
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age = excluded.age,
                household_size = excluded.household_size,
                monthly_income_estimate = excluded.monthly_income_estimate,
                employment_type = excluded.employment_type,
                state = excluded.state,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                user_id.strip(),
                age,
                household_size,
                float(monthly_income_estimate),
                employment_type,
                state.strip(),
            ),
        )

    return get_user_profile(user_id, db_path=db_path)


def get_user_profile(user_id, db_path=None):
    """Retrieve a user's profile, or return None if not found."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT user_id, age, household_size,
                   monthly_income_estimate, employment_type, state
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id.strip(),),
        ).fetchone()

    return dict(row) if row else None
