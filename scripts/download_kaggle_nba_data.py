import kaggle
import os
import zipfile
import yaml
import time # Import time for backoff

# Load configuration
with open('/Users/jackweekly/Desktop/NBA/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the dataset ID
dataset_id = "eoinamoore/historical-nba-data-and-player-box-scores"

# Define download path from config, relative to the project root
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
download_path = os.path.join(project_root, config['data_paths']['raw'])

# Ensure the download directory exists
os.makedirs(download_path, exist_ok=True)

absolute_download_path = os.path.abspath(download_path)
print(f"Downloading dataset '{dataset_id}' to '{absolute_download_path}'...")

# --- Retry Logic with Backoff ---
max_retries = 3
retry_delay_seconds = 5

for attempt in range(max_retries):
    try:
        # Download the dataset files
        kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=False)
        print("Download complete. Unzipping files...")

        # Unzip all downloaded zip files
        for item in os.listdir(download_path):
            if item.endswith('.zip'):
                file_path = os.path.join(download_path, item)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                os.remove(file_path) # Remove the zip file after extraction
                print(f"Extracted and removed '{item}'")

        print("All files unzipped successfully.")
        break # Exit loop if successful
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
            # Classify errors: transient vs fatal. For Kaggle API, most network errors are transient.
            # Fatal errors (e.g., invalid credentials) might warrant immediate exit.
        else:
            print(f"All {max_retries} attempts failed. An error occurred: {e}")
            print("Please ensure your Kaggle API credentials are correctly set up.")
            print("Refer to the instructions for generating and configuring your kaggle.json file.")
            # Optionally, re-raise the exception if it's considered fatal
            # raise