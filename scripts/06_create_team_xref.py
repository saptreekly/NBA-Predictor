
import pandas as pd
import os

# Define file paths
raw_data_path = 'data/raw'
processed_data_path = 'data/processed'
team_histories_csv_path = os.path.join(raw_data_path, 'TeamHistories.csv')
output_csv_path = os.path.join(processed_data_path, 'team_xref.csv')

# Create processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Read the raw data
team_histories_df = pd.read_csv(team_histories_csv_path)

# Create a crosswalk dataframe
team_xref_df = team_histories_df[['teamId', 'teamCity', 'teamName', 'teamAbbrev']]

# Save the processed data
team_xref_df.to_csv(output_csv_path, index=False)

print(f"Team crosswalk data saved to {output_csv_path}")
