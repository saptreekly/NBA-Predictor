
import pandas as pd
import os

# Define file paths
raw_data_path = 'data/raw'
processed_data_path = 'data/processed'
players_csv_path = os.path.join(raw_data_path, 'Players.csv')
output_csv_path = os.path.join(processed_data_path, 'players.csv')

# Create processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Read the raw data
players_df = pd.read_csv(players_csv_path)

# Create a single position column
def get_position(row):
    if row['guard']:
        return 'G'
    if row['forward']:
        return 'F'
    if row['center']:
        return 'C'
    return None

players_df['position'] = players_df.apply(get_position, axis=1)

# Select and rename columns
players_df = players_df[['personId', 'firstName', 'lastName', 'birthdate', 'height', 'bodyWeight', 'position']]

# Save the processed data
players_df.to_csv(output_csv_path, index=False)

print(f"Canonical player data saved to {output_csv_path}")
