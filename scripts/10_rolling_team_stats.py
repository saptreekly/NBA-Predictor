
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
raw_data_path = 'data/raw'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
output_csv_path = os.path.join(processed_data_path, 'rolling_team_stats.csv')

# Read data
games_df = pd.read_csv(games_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])

# Merge team_stats_df with gameDateTime for sorting
team_stats_df = pd.merge(team_stats_df, games_df[['gameID', 'gameDateTime']], left_on='gameId', right_on='gameID')

# Sort by teamId and gameDateTime
team_stats_df = team_stats_df.sort_values(by=['teamId', 'gameDateTime'])

# Calculate possessions
# Possessions = FGA + 0.44*FTA + TO
team_stats_df['possessions'] = team_stats_df['fieldGoalsAttempted'] + 0.44 * team_stats_df['freeThrowsAttempted'] + team_stats_df['turnovers']

# Calculate Net Rating, Offensive Rating, Defensive Rating, Pace
team_stats_df['net_rating'] = (team_stats_df['teamScore'] - team_stats_df['opponentScore']) / team_stats_df['possessions'] * 100
team_stats_df['offensive_rating'] = team_stats_df['teamScore'] / team_stats_df['possessions'] * 100
team_stats_df['defensive_rating'] = team_stats_df['opponentScore'] / team_stats_df['possessions'] * 100
team_stats_df['pace'] = team_stats_df['possessions'] / (team_stats_df['numMinutes'] / 48) # numMinutes is total minutes played by team

# Define rolling windows
windows = [5, 10]

# Calculate rolling aggregates
for window in windows:
    team_stats_df[f'rolling_net_rating_{window}'] = team_stats_df.groupby('teamId')['net_rating'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    team_stats_df[f'rolling_offensive_rating_{window}'] = team_stats_df.groupby('teamId')['offensive_rating'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    team_stats_df[f'rolling_defensive_rating_{window}'] = team_stats_df.groupby('teamId')['defensive_rating'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    team_stats_df[f'rolling_pace_{window}'] = team_stats_df.groupby('teamId')['pace'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    team_stats_df[f'rolling_win_pct_{window}'] = team_stats_df.groupby('teamId')['win'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    team_stats_df[f'rolling_avg_margin_{window}'] = team_stats_df.groupby('teamId')['plusMinusPoints'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())

# Select relevant columns for output
output_cols = ['gameId', 'teamId']
for window in windows:
    output_cols.extend([
        f'rolling_net_rating_{window}',
        f'rolling_offensive_rating_{window}',
        f'rolling_defensive_rating_{window}',
        f'rolling_pace_{window}',
        f'rolling_win_pct_{window}',
        f'rolling_avg_margin_{window}'
    ])

rolling_team_stats_df = team_stats_df[output_cols]

# Save the processed data
rolling_team_stats_df.to_csv(output_csv_path, index=False)

print(f"Rolling team statistics saved to {output_csv_path}")
