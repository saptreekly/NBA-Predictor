
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
raw_data_path = 'data/raw'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
output_csv_path = os.path.join(processed_data_path, 'team_season_state.csv')

# Read the processed and raw data
games_df = pd.read_csv(games_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])

# Merge games_df with team_stats_df to get win/loss for each game
game_outcomes_df = pd.merge(games_df, team_stats_df[['gameId', 'teamId', 'win']], left_on='gameID', right_on='gameId')

# Sort by teamId and gameDateTime
game_outcomes_df = game_outcomes_df.sort_values(by=['teamId', 'gameDateTime'])

# Calculate cumulative wins and losses
game_outcomes_df['cumulative_wins'] = game_outcomes_df.groupby('teamId')['win'].cumsum().shift(1).fillna(0)
game_outcomes_df['cumulative_games'] = game_outcomes_df.groupby('teamId').cumcount()

# Calculate running win percentage
game_outcomes_df['win_percentage'] = (game_outcomes_df['cumulative_wins'] / game_outcomes_df['cumulative_games']).fillna(0)

# Select and rename columns
season_state_df = game_outcomes_df[['gameID', 'teamId', 'win_percentage']]

# Save the processed data
season_state_df.to_csv(output_csv_path, index=False)

print(f"Team season state features saved to {output_csv_path}")
