
import pandas as pd
import os

# Define file paths
raw_data_path = 'data/raw'
processed_data_path = 'data/processed'
team_histories_csv_path = os.path.join(raw_data_path, 'TeamHistories.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
games_csv_path = os.path.join(processed_data_path, 'games.csv')

# Read the data
team_histories_df = pd.read_csv(team_histories_csv_path)
games_df = pd.read_csv(games_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])
games_df['game_year'] = games_df['gameDateTime'].dt.year

# Merge to get game year for each team stat entry
stats_with_year_df = pd.merge(team_stats_df, games_df[['gameID', 'game_year']], left_on='gameId', right_on='gameID')

# Check for inconsistencies
inconsistencies = []
for index, row in stats_with_year_df.iterrows():
    team_id = row['teamId']
    game_year = row['game_year']
    team_history = team_histories_df[team_histories_df['teamId'] == team_id]
    
    is_active = False
    for _, history_row in team_history.iterrows():
        if history_row['seasonFounded'] <= game_year <= history_row['seasonActiveTill']:
            is_active = True
            break
    
    if not is_active:
        inconsistencies.append(row)

# Report inconsistencies
if inconsistencies:
    print("Found inconsistencies where teams appear outside their active seasons:")
    inconsistent_df = pd.DataFrame(inconsistencies)
    print(inconsistent_df[['gameId', 'teamId', 'game_year']])
else:
    print("No inconsistencies found. All teams appear within their active seasons.")
