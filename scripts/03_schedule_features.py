
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
raw_data_path = 'data/raw'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
output_csv_path = os.path.join(processed_data_path, 'schedule_features.csv')

# Read the processed and raw data
games_df = pd.read_csv(games_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])

# Merge games_df with team_stats_df to get teamId for each game
game_teams_df = pd.merge(games_df, team_stats_df[['gameId', 'teamId']], left_on='gameID', right_on='gameId')

# Sort by teamId and gameDateTime
game_teams_df = game_teams_df.sort_values(by=['teamId', 'gameDateTime'])

# Calculate rest days
game_teams_df['previous_gameDateTime'] = game_teams_df.groupby('teamId')['gameDateTime'].shift(1)
game_teams_df['rest_days'] = (game_teams_df['gameDateTime'] - game_teams_df['previous_gameDateTime']).dt.days

# Calendar features
game_teams_df['day_of_week'] = game_teams_df['gameDateTime'].dt.dayofweek
game_teams_df['week_of_year'] = game_teams_df['gameDateTime'].dt.isocalendar().week

# Select and rename columns
schedule_features_df = game_teams_df[['gameID', 'teamId', 'rest_days', 'day_of_week', 'week_of_year']]

# Save the processed data
schedule_features_df.to_csv(output_csv_path, index=False)

print(f"Schedule features saved to {output_csv_path}")
