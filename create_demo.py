from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_PATH = PROJECT_ROOT / "demo_transactions.csv"


def create_demo_dataset() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place your creditcard.csv file in the data folder before creating a demo file."
        )

    df = pd.read_csv(DATA_PATH)
    if "Class" in df.columns:
        fraud_df = df[df["Class"] == 1].sample(n=min(20, len(df[df["Class"] == 1])), random_state=42)
        normal_df = df[df["Class"] == 0].sample(n=min(80, len(df[df["Class"] == 0])), random_state=42)
        demo_df = pd.concat([fraud_df, normal_df], ignore_index=True).sample(frac=1, random_state=42)
    else:
        demo_df = df.head(100).copy()

    demo_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Demo dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(demo_df)}")
    print(f"Columns: {len(demo_df.columns)}")


if __name__ == "__main__":
    create_demo_dataset()
