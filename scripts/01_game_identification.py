
import pandas as pd
import os

# Define file paths
raw_data_path = 'data/raw'
processed_data_path = 'data/processed'
games_csv_path = os.path.join(raw_data_path, 'Games.csv')
schedule_csv_path = os.path.join(raw_data_path, 'LeagueSchedule24_25.csv')
output_csv_path = os.path.join(processed_data_path, 'games.csv')

# Create processed data directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Read the raw data
games_df = pd.read_csv(games_csv_path)
schedule_df = pd.read_csv(schedule_csv_path)

# Select and rename columns from Games.csv
games_df = games_df[['gameId', 'gameDate', 'gameType', 'seriesGameNumber']]
games_df = games_df.rename(columns={'gameId': 'gameID', 'gameDate': 'gameDateTime', 'gameType': 'gameType', 'seriesGameNumber': 'seriesGameNumber'})

# Select and rename columns from LeagueSchedule24_25.csv
schedule_df = schedule_df[['gameId', 'gameDateTimeEst', 'seriesGameNumber', 'seriesText']]
schedule_df = schedule_df.rename(columns={'gameId': 'gameID', 'gameDateTimeEst': 'gameDateTime', 'seriesGameNumber': 'seriesGameNumber', 'seriesText': 'seriesText'})
schedule_df['gameType'] = 'Regular Season' # Assuming future games are regular season

# Convert gameDateTime columns to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime']).dt.tz_localize('UTC')
schedule_df['gameDateTime'] = pd.to_datetime(schedule_df['gameDateTime'], format='ISO8601')

# Combine the two dataframes
combined_df = pd.concat([games_df, schedule_df], ignore_index=True)

# Remove duplicate gameIDs, keeping the first occurrence
combined_df = combined_df.drop_duplicates(subset='gameID', keep='first')

# Sort by gameDateTime
combined_df = combined_df.sort_values(by='gameDateTime')

# Save the processed data
combined_df.to_csv(output_csv_path, index=False)

print(f"Processed game data saved to {output_csv_path}")
