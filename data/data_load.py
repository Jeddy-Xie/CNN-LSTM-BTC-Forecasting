from kaggle import api
import os

# Define the path where you want to store the downloaded dataset
download_path = "data/raw"

# Create the directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download and unzip the dataset
print("Downloading dataset...")
# Download and unzip the dataset from Kaggle
api.dataset_download_files('prasoonkottarathil/btcinusd', path=download_path, unzip=True)
print("Dataset downloaded and unzipped in:", download_path)
