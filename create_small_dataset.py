import pandas as pd
import os

# Read the original dataset
original_data = pd.read_csv('data/processed/BTC-USD.csv')

# Take the first 1000 rows
small_data = original_data.head(1000)

# Save to the same location
small_data.to_csv('data/processed/BTC-USD-small.csv', index=False)

print(f"Created small dataset with {len(small_data)} rows")
print("Saved to: data/processed/BTC-USD-small.csv") 