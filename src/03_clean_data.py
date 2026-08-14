import pandas as pd
import os

input_folder = "data/raw"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(input_folder)

for file in files:

    if not file.endswith(".csv"):
        continue

    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)

    print(f"Processing {file}")

    df = pd.read_csv(input_path)

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically
    df = df.sort_values("Date")

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["Date"])

    # Remove rows where essential prices are missing
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    df = df.dropna(subset=required_columns)

    # Remove invalid prices
    df = df[df["Close"] > 0]

    # Save
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows")

print("\nCleaning completed.")