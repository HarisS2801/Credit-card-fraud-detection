import pandas as pd
import json
from pathlib import Path

DATASET = Path("data/creditcard.csv")

df = pd.read_csv(DATASET)

# Select one normal transaction
normal = df[df["Class"] == 0].iloc[0]

# Select one fraud transaction
fraud = df[df["Class"] == 1].iloc[0]

# Remove Class because the API must not receive the target
normal_payload = normal.drop("Class").to_dict()
fraud_payload = fraud.drop("Class").to_dict()

# Save JSON files
with open("normal_transaction.json", "w") as f:
    json.dump(normal_payload, f, indent=2)

with open("fraud_transaction.json", "w") as f:
    json.dump(fraud_payload, f, indent=2)

print("Demo payloads created successfully.")
print()
print("Normal transaction:")
print("normal_transaction.json")
print()
print("Fraud transaction:")
print("fraud_transaction.json")