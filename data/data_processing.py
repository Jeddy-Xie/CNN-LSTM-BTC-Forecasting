import os
import pandas as pd

# Define paths for raw and processed data
raw_data_path = "data/raw"
processed_data_path = "data/processed"

# Create the processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# List of CSV files to combine (ensure these file names match the ones in your raw directory)
files = [
    "BTC-2017min.csv",
    "BTC-2018min.csv",
    "BTC-2019min.csv",
    "BTC-2020min.csv",
    "BTC-2021min.csv",
]

dataframes = []

# Read each CSV file and append its DataFrame to the list
for file in files:
    file_path = os.path.join(raw_data_path, file)
    print("Processing file:", file_path)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a CSV file in the processed data folder
combined_csv_path = os.path.join(processed_data_path, "BTC-USD.csv")
combined_df.to_csv(combined_csv_path, index=False)
print("Combined CSV saved to:", combined_csv_path)
