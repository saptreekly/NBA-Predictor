
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
raw_data_path = 'data/raw'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
output_csv_path = os.path.join(processed_data_path, 'opponent_tendencies.csv')

# Read data
games_df = pd.read_csv(games_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])

# Merge team_stats_df with gameDateTime for sorting
team_stats_df = pd.merge(team_stats_df, games_df[['gameID', 'gameDateTime']], left_on='gameId', right_on='gameID')

# Sort by teamId and gameDateTime
team_stats_df = team_stats_df.sort_values(by=['teamId', 'gameDateTime'])

# Calculate possessions (if not already calculated)
team_stats_df['possessions'] = team_stats_df['fieldGoalsAttempted'] + 0.44 * team_stats_df['freeThrowsAttempted'] + team_stats_df['turnovers']

# Calculate raw stats
team_stats_df['3PA_rate'] = team_stats_df['threePointersAttempted'] / team_stats_df['fieldGoalsAttempted']
team_stats_df['ORB_pct'] = team_stats_df['reboundsOffensive'] / (team_stats_df['reboundsOffensive'] + team_stats_df['reboundsDefensive'].shift(-1)) # Opponent DRB
team_stats_df['FTr'] = team_stats_df['freeThrowsAttempted'] / team_stats_df['fieldGoalsAttempted']
team_stats_df['eFG_pct'] = (team_stats_df['fieldGoalsMade'] + 0.5 * team_stats_df['threePointersMade']) / team_stats_df['fieldGoalsAttempted']
team_stats_df['TOV_pct'] = team_stats_df['turnovers'] / team_stats_df['possessions']

# Define rolling windows
windows = [5, 10]

# Calculate rolling aggregates for team
for window in windows:
    for stat in ['3PA_rate', 'ORB_pct', 'FTr', 'eFG_pct', 'TOV_pct', 'possessions']:
        team_stats_df[f'rolling_{stat}_{window}'] = team_stats_df.groupby('teamId')[stat].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())

# Prepare opponent_stats_df for merging
opponent_stats_for_merge = team_stats_df.copy()
opponent_stats_for_merge = opponent_stats_for_merge.rename(columns={'teamId': 'opponentTeamId', 'opponentTeamId': 'teamId_original'})

# Merge to get opponent's rolling stats
merged_df = pd.merge(team_stats_df, opponent_stats_for_merge, 
                     left_on=['gameId', 'opponentTeamId'], right_on=['gameId', 'opponentTeamId'], 
                     suffixes=('_team', '_opponent'))

# Rename gameId to gameId and teamId_team to teamId
merged_df = merged_df.rename(columns={'teamId_team': 'teamId'})

# Select relevant columns for output
output_cols = ['gameId', 'teamId']
for window in windows:
    for stat in ['3PA_rate', 'ORB_pct', 'FTr', 'eFG_pct', 'TOV_pct', 'possessions']:
        output_cols.append(f'rolling_{stat}_{window}_team')
        output_cols.append(f'rolling_{stat}_{window}_opponent')

opponent_tendencies_df = merged_df[output_cols]

# Save the processed data
opponent_tendencies_df.to_csv(output_csv_path, index=False)

print(f"Opponent tendencies saved to {output_csv_path}")
