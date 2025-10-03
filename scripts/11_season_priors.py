
import pandas as pd
import os

# Define file paths
processed_data_path = 'data/processed'
raw_data_path = 'data/raw'
games_csv_path = os.path.join(processed_data_path, 'games.csv')
team_stats_csv_path = os.path.join(raw_data_path, 'TeamStatistics.csv')
output_csv_path = os.path.join(processed_data_path, 'season_priors.csv')

# Read data
games_df = pd.read_csv(games_csv_path)
team_stats_df = pd.read_csv(team_stats_csv_path)

# Convert gameDateTime to datetime objects
games_df['gameDateTime'] = pd.to_datetime(games_df['gameDateTime'])

# Merge team_stats_df with gameDateTime for sorting
team_stats_df = pd.merge(team_stats_df, games_df[['gameID', 'gameDateTime']], left_on='gameId', right_on='gameID')

# Sort by teamId and gameDateTime
team_stats_df = team_stats_df.sort_values(by=['teamId', 'gameDateTime'])

# Calculate possessions (same as in 10_rolling_team_stats.py)
team_stats_df['possessions'] = team_stats_df['fieldGoalsAttempted'] + 0.44 * team_stats_df['freeThrowsAttempted'] + team_stats_df['turnovers']

# Calculate Net Rating
team_stats_df['net_rating'] = (team_stats_df['teamScore'] - team_stats_df['opponentScore']) / team_stats_df['possessions'] * 100

# Determine season for each game
def get_season(date):
    if date.month >= 10:
        return date.year + 1
    else:
        return date.year

team_stats_df['season'] = team_stats_df['gameDateTime'].apply(get_season)

# Calculate last season's net rating
season_net_ratings = team_stats_df.groupby(['teamId', 'season'])['net_rating'].mean().reset_index()
season_net_ratings = season_net_ratings.rename(columns={'net_rating': 'avg_net_rating'})

# Shift to get previous season's net rating
season_net_ratings['previous_season'] = season_net_ratings['season'] + 1
last_season_net_rating = pd.merge(team_stats_df[['gameId', 'teamId', 'season']], season_net_ratings[['teamId', 'previous_season', 'avg_net_rating']],
                                  left_on=['teamId', 'season'], right_on=['teamId', 'previous_season'], how='left')
last_season_net_rating = last_season_net_rating.rename(columns={'avg_net_rating': 'last_season_net_rating'})
last_season_net_rating = last_season_net_rating[['gameId', 'teamId', 'last_season_net_rating']]

# --- Elo Rating System ---

# Initialize Elo ratings
initial_elo = 1500
k_factor = 20
regression_factor = 2/3 # Regress 1/3 towards the mean

elo_ratings = {team_id: initial_elo for team_id in team_stats_df['teamId'].unique()}
season_start_elos = {}

# Prepare dataframe for Elo calculation
elo_df = team_stats_df[['gameId', 'gameDateTime', 'teamId', 'opponentTeamId', 'teamScore', 'opponentScore', 'win', 'season']].copy()
elo_df = elo_df.sort_values(by=['gameDateTime', 'gameId'])

# Store Elo ratings before each game
elo_df['elo_rating'] = 0.0

for index, row in elo_df.iterrows():
    game_id = row['gameId']
    team_id = row['teamId']
    opponent_team_id = row['opponentTeamId']
    team_score = row['teamScore']
    opponent_score = row['opponentScore']
    win = row['win']
    season = row['season']

    # Apply season regression at the start of a new season
    if season not in season_start_elos:
        for tid, elo in elo_ratings.items():
            elo_ratings[tid] = initial_elo + regression_factor * (elo - initial_elo)
        season_start_elos[season] = True

    # Get current Elo ratings
    team_elo = elo_ratings.get(team_id, initial_elo)
    opponent_elo = elo_ratings.get(opponent_team_id, initial_elo)

    # Store Elo rating before the game
    elo_df.loc[index, 'elo_rating'] = team_elo

    # Calculate expected outcome
    expected_win_team = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

    # Determine actual outcome
    actual_win_team = 1 if win == 1 else 0

    # Update Elo ratings
    elo_ratings[team_id] = team_elo + k_factor * (actual_win_team - expected_win_team)
    elo_ratings[opponent_team_id] = opponent_elo + k_factor * (expected_win_team - actual_win_team)

# Merge Elo ratings with last season's net rating
season_priors_df = pd.merge(last_season_net_rating, elo_df[['gameId', 'teamId', 'elo_rating']], on=['gameId', 'teamId'], how='left')

# Save the processed data
season_priors_df.to_csv(output_csv_path, index=False)

print(f"Season priors saved to {output_csv_path}")
