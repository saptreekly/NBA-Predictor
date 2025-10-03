
import pandas as pd
import os

# Define file paths
raw_data_path = 'data/raw'
processed_data_path = 'data/processed'
games_csv_path = os.path.join(raw_data_path, 'Games.csv')
schedule_csv_path = os.path.join(raw_data_path, 'LeagueSchedule24_25.csv')
output_csv_path = os.path.join(processed_data_path, 'arenas.csv')

# Create processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Read the raw data
games_df = pd.read_csv(games_csv_path, dtype={'gameId': str})
schedule_df = pd.read_csv(schedule_csv_path, dtype={'gameId': str})

# Select relevant columns
games_arenas = games_df[['gameId', 'arenaId']].dropna()
schedule_arenas = schedule_df[['gameId', 'arenaName', 'arenaCity', 'arenaState']].dropna()

# Merge the two dataframes on gameId
merged_arenas = pd.merge(games_arenas, schedule_arenas, on='gameId')

# Create canonical list of arenas
canonical_arenas = merged_arenas.drop_duplicates(subset=['arenaId', 'arenaName'])

# Select and rename columns
canonical_arenas = canonical_arenas[['arenaId', 'arenaName', 'arenaCity', 'arenaState']]

# Save the processed data
canonical_arenas.to_csv(output_csv_path, index=False)

print(f"Canonical arena data saved to {output_csv_path}")
