from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def ensure_dataset_exists() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return

    rng = np.random.default_rng(RANDOM_STATE)
    fraud_rows = 492
    normal_rows = 10_000
    feature_count = 28

    fraud_features = rng.normal(loc=2.0, scale=2.5, size=(fraud_rows, feature_count))
    normal_features = rng.normal(loc=0.0, scale=1.0, size=(normal_rows, feature_count))

    time_values = rng.integers(0, 86400, size=normal_rows + fraud_rows)
    amount_values = np.abs(rng.lognormal(mean=0.5, sigma=1.2, size=normal_rows + fraud_rows))

    feature_names = [f"V{i}" for i in range(1, 29)]
    fraud_df = pd.DataFrame(fraud_features, columns=feature_names)
    normal_df = pd.DataFrame(normal_features, columns=feature_names)

    fraud_df["Time"] = time_values[:fraud_rows]
    normal_df["Time"] = time_values[fraud_rows:]
    fraud_df["Amount"] = amount_values[:fraud_rows]
    normal_df["Amount"] = amount_values[fraud_rows:]

    fraud_df["Class"] = 1
    normal_df["Class"] = 0

    synthetic_df = pd.concat([normal_df, fraud_df], ignore_index=True)
    synthetic_df = synthetic_df[["Time", *feature_names, "Amount", "Class"]]
    synthetic_df.to_csv(DATA_PATH, index=False)

    print(f"Synthetic dataset created at: {DATA_PATH}")


def get_dataset_summary(df: pd.DataFrame) -> None:
    print("\nDataset Overview")
    print("-" * 60)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Data types:\n{df.dtypes}")

    class_counts = df["Class"].value_counts().sort_index()
    print("Class distribution:")
    print(class_counts)
    print(f"Fraud cases: {int(class_counts.get(1, 0))}")
    print(f"Normal cases: {int(class_counts.get(0, 0))}")


def plot_class_distribution(df: pd.DataFrame) -> None:
    class_labels = df["Class"].map({0: "Normal", 1: "Fraud"})
    plt.figure(figsize=(7, 5))
    sns.countplot(x=class_labels, palette="Set2")
    plt.title("Normal vs Fraud Transaction Count")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "class_distribution.png", dpi=200)
    plt.close()


def plot_amount_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=40,
        stat="density",
        common_norm=False,
        palette="Set2",
    )
    plt.title("Transaction Amount Distribution by Class")
    plt.xlabel("Amount")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "amount_distribution.png", dpi=200)
    plt.close()


def evaluate_model(model, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def train_models() -> tuple[dict, dict, dict]:
    ensure_dataset_exists()
    df = pd.read_csv(DATA_PATH)
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows before removal: {duplicate_count}")
    df = df.drop_duplicates().copy()
    print(f"Shape after duplicate removal: {df.shape}")
    print("Class distribution after duplicate removal:")
    print(df["Class"].value_counts().sort_index())
    get_dataset_summary(df)

    plot_class_distribution(df)
    plot_amount_distribution(df)

    fraud_df = df[df["Class"] == 1].copy()
    normal_df = df[df["Class"] == 0].sample(n=min(10_000, len(df[df["Class"] == 0])), random_state=RANDOM_STATE)
    sampled_df = pd.concat([fraud_df, normal_df], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    feature_columns = [col for col in sampled_df.columns if col != "Class"]
    X = sampled_df[feature_columns]
    y = sampled_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)

    logistic_model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
    logistic_model.fit(X_train_lr, y_train)
    lr_pred = logistic_model.predict(X_test_lr)
    lr_prob = logistic_model.predict_proba(X_test_lr)[:, 1]
    lr_metrics = evaluate_model(logistic_model, y_test, lr_pred, lr_prob)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(rf_model, y_test, rf_pred, rf_prob)

    comparison = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Precision": [lr_metrics["precision"], rf_metrics["precision"]],
            "Recall": [lr_metrics["recall"], rf_metrics["recall"]],
            "F1-score": [lr_metrics["f1"], rf_metrics["f1"]],
            "ROC-AUC": [lr_metrics["roc_auc"], rf_metrics["roc_auc"]],
        }
    )

    print("\nModel Comparison")
    print("-" * 60)
    print(comparison.to_string(index=False))

    final_metrics = lr_metrics if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"] else rf_metrics
    final_model_name = "Logistic Regression" if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"] else "Random Forest"

    if final_model_name == "Logistic Regression":
        final_model = logistic_model
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
    else:
        final_model = rf_model
        scaler_path = MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            scaler_path.unlink()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "fraud_model.pkl")
    joblib.dump(list(X.columns), MODELS_DIR / "features.pkl")

    print(f"\nSelected production model: {final_model_name}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1-score: {final_metrics['f1']:.4f}")
    print(f"ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")

    return lr_metrics, rf_metrics, {"winner": final_model_name, "metrics": final_metrics}


if __name__ == "__main__":
    train_models()
