
import pandas as pd
import os

# Define file paths
raw_data_path = 'data/raw'
processed_data_path = 'data/processed'
team_histories_csv_path = os.path.join(raw_data_path, 'TeamHistories.csv')
output_csv_path = os.path.join(processed_data_path, 'teams.csv')

# Create processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Read the raw data
team_histories_df = pd.read_csv(team_histories_csv_path)

# Sort by teamId and seasonActiveTill in descending order
team_histories_df = team_histories_df.sort_values(by=['teamId', 'seasonActiveTill'], ascending=[True, False])

# Drop duplicates, keeping the first entry for each teamId (which will be the most recent)
canonical_teams_df = team_histories_df.drop_duplicates(subset='teamId', keep='first')

# Select and rename columns
canonical_teams_df = canonical_teams_df[['teamId', 'teamCity', 'teamName', 'teamAbbrev']]

# Save the processed data
canonical_teams_df.to_csv(output_csv_path, index=False)

print(f"Canonical team data saved to {output_csv_path}")
