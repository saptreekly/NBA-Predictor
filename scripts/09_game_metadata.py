
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
raw_data_path = 'data/raw'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
schedule_features_csv_path = os.path.join(processed_data_path, 'schedule_features.csv')
teams_csv_path = os.path.join(processed_data_path, 'teams.csv')
arenas_csv_path = os.path.join(processed_data_path, 'arenas.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
output_csv_path = os.path.join(processed_data_path, 'game_metadata.csv')

# Read data
games_df = pd.read_csv(games_csv_path)
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])
schedule_features_df = pd.read_csv(schedule_features_csv_path)
teams_df = pd.read_csv(teams_csv_path)
arenas_df = pd.read_csv(arenas_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Merge dataframes
# Start with games_df
game_metadata_df = pd.merge(games_df, schedule_features_df, on='gameID')

# Add home/away flag from team_stats_df
game_metadata_df = pd.merge(game_metadata_df, team_stats_df[['gameId', 'teamId', 'home']], left_on=['gameID', 'teamId'], right_on=['gameId', 'teamId'])

raw_games_df = pd.read_csv(os.path.join(raw_data_path, 'Games.csv'))
game_metadata_df = pd.merge(game_metadata_df, raw_games_df[['gameId', 'arenaId']], left_on='gameID', right_on='gameId', how='left')

# Add arena information
game_metadata_df = pd.merge(game_metadata_df, arenas_df, on='arenaId', how='left')


game_metadata_df['is_b2b'] = (game_metadata_df['rest_days'] == 1).astype(int)

def calculate_rest_categories(group):
    group = group.sort_values('gameDateTime')
    
    is_3_in_4 = []
    for i in range(len(group)):
        current_date = group.iloc[i]['gameDateTime']
        past_4_days = group[(group['gameDateTime'] <= current_date) & (group['gameDateTime'] > current_date - pd.Timedelta(days=4))]
        is_3_in_4.append(len(past_4_days) >= 3)
    group['is_3_in_4'] = is_3_in_4

    is_4_in_6 = []
    for i in range(len(group)):
        current_date = group.iloc[i]['gameDateTime']
        past_6_days = group[(group['gameDateTime'] <= current_date) & (group['gameDateTime'] > current_date - pd.Timedelta(days=6))]
        is_4_in_6.append(len(past_6_days) >= 4)
    group['is_4_in_6'] = is_4_in_6
    
    return group

game_metadata_df = game_metadata_df.groupby('teamId').apply(calculate_rest_categories)




# Select and rename columns
game_metadata_df = game_metadata_df[[
    'gameID', 'teamId', 'home', 'gameType', 'seriesGameNumber', 'week_of_year', 'day_of_week',
    'arenaId', 'arenaName', 'arenaCity', 'arenaState', 'rest_days', 'is_b2b', 'is_3_in_4', 'is_4_in_6'
]]

# Save the processed data
game_metadata_df.to_csv(output_csv_path, index=False)

print(f"Game metadata saved to {output_csv_path}")
