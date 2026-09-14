from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "fraud_detection.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                transaction_amount REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                recommended_action TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction_record(
    transaction_amount: float,
    fraud_probability: float,
    prediction: str,
    risk_level: str,
    recommended_action: str,
) -> int:
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_records (
                timestamp,
                transaction_amount,
                fraud_probability,
                prediction,
                risk_level,
                recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, float(transaction_amount), float(fraud_probability), prediction, risk_level, recommended_action),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_recent_predictions(limit: int = 10) -> list[dict[str, object]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, transaction_amount, fraud_probability, prediction, risk_level, recommended_action
            FROM prediction_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
