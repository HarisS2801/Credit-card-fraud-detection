from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from database import get_recent_predictions
from src.model_service import get_prediction_service

API_URL = "http://127.0.0.1:8000/predict"
PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DATA_PATH = PROJECT_ROOT / "demo_transactions.csv"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def get_recent_transactions() -> pd.DataFrame:
    rows = get_recent_predictions(limit=20)
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "transaction_amount",
                "fraud_probability",
                "prediction",
                "risk_level",
                "recommended_action",
            ]
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    df["fraud_probability"] = pd.to_numeric(df["fraud_probability"], errors="coerce")
    return df


def get_demo_source_dataframe() -> pd.DataFrame:
    if not DEMO_DATA_PATH.exists():
        raise FileNotFoundError(f"Demo data not found: {DEMO_DATA_PATH}")

    df = pd.read_csv(DEMO_DATA_PATH)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    expected_columns = FEATURE_COLUMNS
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Demo transaction file is missing expected feature columns: {missing}")

    return df[expected_columns].copy()


def get_demo_transaction(option: str) -> dict[str, Any]:
    service = get_prediction_service()
    demo_df = pd.read_csv(DEMO_DATA_PATH)
    feature_df = get_demo_source_dataframe()

    scored_rows: list[dict[str, Any]] = []
    for index, row in feature_df.iterrows():
        payload = {key: float(value) for key, value in row.items() if key in FEATURE_COLUMNS}
        result = service.predict_transaction(payload)
        class_label = None
        if "Class" in demo_df.columns:
            class_label = int(demo_df.iloc[index]["Class"])
        scored_rows.append(
            {
                "payload": payload,
                "fraud_probability": float(result["fraud_probability"]),
                "result": result,
                "class_label": class_label,
            }
        )

    if option == "normal":
        candidate_rows = scored_rows if not any(item["class_label"] is not None for item in scored_rows) else [item for item in scored_rows if item["class_label"] == 0]
        selected = min(candidate_rows, key=lambda item: item["fraud_probability"])
    elif option == "fraud":
        candidate_rows = scored_rows if not any(item["class_label"] is not None for item in scored_rows) else [item for item in scored_rows if item["class_label"] == 1]
        if not candidate_rows:
            candidate_rows = scored_rows
        selected = max(candidate_rows, key=lambda item: item["fraud_probability"])
    else:
        raise ValueError(f"Unsupported demo transaction option: {option}")

    payload = selected["payload"]
    result = selected["result"]
    return {"payload": payload, "result": result}


def build_sample_payload() -> dict[str, Any]:
    payload = {"Time": 12345.0, "Amount": 250.5}
    for feature in FEATURE_COLUMNS:
        if feature not in {"Time", "Amount"}:
            payload[feature] = 0.0
    return payload


def submit_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(API_URL, json=payload, timeout=10)
    if response.status_code != 200:
        raise ValueError(response.json().get("detail", "Prediction request failed."))
    return response.json()


def render_kpis(df: pd.DataFrame) -> None:
    total_transactions = len(df)
    fraud_alerts = int((df["prediction"] == "FRAUD").sum()) if not df.empty else 0
    normal_transactions = total_transactions - fraud_alerts
    high_risk = int((df["risk_level"] == "HIGH").sum()) if not df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total_transactions)
    col2.metric("Fraud Alerts", fraud_alerts)
    col3.metric("Normal Transactions", normal_transactions)
    col4.metric("High Risk Transactions", high_risk)


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No transaction history is available yet. Submit a test transaction to populate the dashboard.")
        return

    chart1, chart2 = st.columns(2)

    with chart1:
        prediction_counts = df["prediction"].value_counts().reindex(["NORMAL", "FRAUD"], fill_value=0)
        st.bar_chart(prediction_counts)

    with chart2:
        risk_counts = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        st.bar_chart(risk_counts)

    col3, col4 = st.columns(2)
    with col3:
        st.area_chart(df["fraud_probability"].sort_index())
    with col4:
        if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
            fraud_timeline = df[df["prediction"] == "FRAUD"].set_index("timestamp").resample("D").size()
            st.line_chart(fraud_timeline)
        else:
            st.info("Fraud timeline data will appear after enough predictions are recorded.")


def render_recent_transactions(df: pd.DataFrame) -> None:
    st.subheader("Recent Transactions")
    if df.empty:
        st.caption("No recent transactions yet.")
        return

    display_df = df[["id", "timestamp", "transaction_amount", "fraud_probability", "prediction", "risk_level", "recommended_action"]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df.rename(
        columns={
            "id": "Transaction ID",
            "timestamp": "Time",
            "transaction_amount": "Amount",
            "fraud_probability": "Fraud Probability",
            "prediction": "Prediction",
            "risk_level": "Risk Level",
            "recommended_action": "Action",
        },
        inplace=True,
    )
    st.dataframe(display_df, use_container_width=True)


def render_prediction_form() -> None:
    st.subheader("Test Demo Transaction")
    st.caption("This sends one real transaction from demo_transactions.csv through the same API, risk, and database flow as a live request.")

    with st.form("transaction_test_form"):
        option = st.radio("Choose test sample", ["Normal demo transaction", "Fraud demo transaction"])
        selected_option = "normal" if option == "Normal demo transaction" else "fraud"
        demo_choice = get_demo_transaction(selected_option)
        payload = demo_choice["payload"]
        result = demo_choice["result"]

        st.write(f"Selected transaction amount: ${payload['Amount']:.2f}")
        st.write(f"Predicted probability: {result['fraud_probability'] * 100:.2f}%")
        st.write(f"Current classification: {result['prediction']} / {result['risk_level']} / {result['recommended_action']}")

        submitted = st.form_submit_button("Submit Selected Demo Transaction")

    if submitted:
        try:
            api_result = submit_transaction(payload)
            st.success("Transaction submitted successfully.")
            st.metric("Transaction Amount", f"${payload['Amount']:.2f}")
            st.metric("Fraud Probability", f"{api_result['fraud_probability'] * 100:.2f}%")
            st.metric("Prediction", api_result["prediction"])
            st.metric("Risk Level", api_result["risk_level"])
            st.metric("Recommended Action", api_result["recommended_action"])
            st.rerun()
        except Exception as exc:
            st.error(f"API request failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Fraud Monitoring Dashboard", page_icon="📊", layout="wide")
    st.title("Credit Card Fraud Monitoring Dashboard")
    st.caption("Live fraud observations from the local SQLite database and FastAPI backend.")

    df = get_recent_transactions()
    render_kpis(df)
    render_charts(df)
    render_recent_transactions(df)
    render_prediction_form()


if __name__ == "__main__":
    main()
