
from backend.database import get_connection


def find_possible_duplicates(user_id=None, db_path=None):
    """Find transactions that share key details and may be duplicates.

    Matching records are flagged for review, not automatically deleted.
    """
    query = """
        SELECT
            transaction_id,
            user_id,
            transaction_date,
            transaction_type,
            amount,
            category,
            description,
            merchant_name
        FROM transactions
        WHERE 1 = 1
    """
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    query += """
        ORDER BY user_id, transaction_date, amount, category,
                 transaction_type, transaction_id
    """

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    records = [dict(row) for row in rows]
    groups = {}

    for record in records:
        key = (
            record["user_id"],
            record["transaction_date"],
            record["transaction_type"],
            round(float(record["amount"]), 2),
            (record["category"] or "").strip().casefold(),
            (record["merchant_name"] or "").strip().casefold(),
            (record["description"] or "").strip().casefold(),
        )
        groups.setdefault(key, []).append(record)

    duplicates = []

    for matching_records in groups.values():
        if len(matching_records) > 1:
            duplicates.append({
                "reason": "Matching user, date, type, amount, category, "
                          "merchant, and description",
                "transactions": matching_records,
            })

    return duplicates
