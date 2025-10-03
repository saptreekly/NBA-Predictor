
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
output_csv_path = os.path.join(processed_data_path, 'seasons.csv')

# Read the processed data
games_df = pd.read_csv(games_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])

# Extract season from gameDateTime
def get_season(date):
    if date.month >= 10:
        return date.year + 1
    else:
        return date.year

games_df['season'] = games_df['gameDateTime'].apply(get_season)

# Group by season to get start and end dates
seasons_df = games_df.groupby('season')['gameDateTime'].agg(['min', 'max']).reset_index()
seasons_df = seasons_df.rename(columns={'min': 'start_date', 'max': 'end_date'})

# Get game types for each season
season_game_types = games_df.groupby('season')['gameType'].unique().apply(list).reset_index()
seasons_df = pd.merge(seasons_df, season_game_types, on='season')

# Save the processed data
seasons_df.to_csv(output_csv_path, index=False)

print(f"Season lookup table saved to {output_csv_path}")
