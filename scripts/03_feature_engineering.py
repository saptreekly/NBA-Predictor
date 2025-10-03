import pandas as pd
import os
import yaml
import numpy as np

# Load configuration
with open('/Users/jackweekly/Desktop/NBA/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths from config, using absolute paths
processed_data_path = os.path.join(project_root, config['data_paths']['processed'])
raw_data_path = os.path.join(project_root, config['data_paths']['raw'])

# Helper function to estimate possessions
def estimate_possessions(df, team_prefix):
    fga = df[f'fieldGoalsAttempted_{team_prefix}_agg']
    fta = df[f'freeThrowsAttempted_{team_prefix}_agg']
    oreb = df[f'reboundsOffensive_{team_prefix}_agg'] # Assuming offensive rebounds are available
    tov = df[f'turnovers_{team_prefix}_agg']

    # Dean Oliver's formula for possessions
    possessions = fga + 0.44 * fta - oreb + tov
    return possessions

# Load processed data
print("Loading processed data...")
df_games = pd.read_parquet(os.path.join(processed_data_path, "games_processed.parquet"))
df_player_stats = pd.read_parquet(os.path.join(processed_data_path, "player_stats_processed.parquet"))

# Load advanced features
df_advanced_features = pd.read_csv(os.path.join(processed_data_path, "TeamStatistics_AdvancedFeatures.csv"))
df_advanced_features['gameDate'] = pd.to_datetime(df_advanced_features['gameDate'], utc=True)

print(f"df_games shape before home merge: {df_games.shape}")
print(f"df_advanced_features shape: {df_advanced_features.shape}")

# Merge advanced features for home team
home_adv_features = df_advanced_features.copy()
home_adv_features = home_adv_features.rename(columns={'teamId': 'hometeamId'})
for col in [c for c in home_adv_features.columns if c not in ['gameId', 'gameDate', 'hometeamId']]:
    home_adv_features = home_adv_features.rename(columns={col: f'{col}_home_adv'})
df_games = pd.merge(
    df_games,
    home_adv_features,
    on=['gameId', 'gameDate', 'hometeamId'],
    how='left'
)
print(f"df_games shape after home merge: {df_games.shape}")
print(f"df_games columns after home merge: {df_games.columns.tolist()}")

# Merge advanced features for away team
away_adv_features = df_advanced_features.copy()
away_adv_features = away_adv_features.rename(columns={'teamId': 'awayteamId'})
for col in [c for c in away_adv_features.columns if c not in ['gameId', 'gameDate', 'awayteamId']]:
    away_adv_features = away_adv_features.rename(columns={col: f'{col}_away_adv'})
df_games = pd.merge(
    df_games,
    away_adv_features,
    on=['gameId', 'gameDate', 'awayteamId'],
    how='left'
)
print(f"df_games shape after away merge: {df_games.shape}")
print(f"df_games columns after away merge: {df_games.columns.tolist()}")
print("Game data transformed.")

# Determine the actual game winner based on scores
df_games['game_winner'] = np.where(df_games['homeScore'] > df_games['awayScore'], df_games['hometeamId'], df_games['awayteamId'])

# Ensure gameDate is datetime for sorting, as it might be read as object from CSV
df_games['gameDate'] = pd.to_datetime(df_games['gameDate'])


# --- Feature Engineering: Aggregate Player Stats to Team Level ---
print("Aggregating player statistics to team level...")

# Define stats to aggregate
stats_to_aggregate = [
    'points', 'assists', 'blocks', 'steals', 'reboundsTotal', 'reboundsOffensive',
    'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted',
    'freeThrowsMade', 'freeThrowsAttempted', 'turnovers', 'foulsPersonal',
    'plusMinusPoints', 'numMinutes' # Include numMinutes for weighted averages later
]

# Aggregate stats for each player's team in each game
# We need to differentiate between home and away teams for aggregation

# First, identify which team the player belongs to in that specific game
# and then aggregate based on gameId and teamId

# Create a temporary DataFrame for aggregation
df_player_stats_agg = df_player_stats.copy()

# Group by gameId and playerteamId and sum the stats
team_game_stats = df_player_stats_agg.groupby(['gameId', 'playerteamId'])[stats_to_aggregate].sum().reset_index()

print("Player statistics aggregated.")

# --- Merge Aggregated Team Stats into Games DataFrame ---
print("Merging aggregated team stats into games DataFrame...")

# Merge home team stats
home_team_game_stats = team_game_stats.copy()
home_team_game_stats = home_team_game_stats.rename(columns={'playerteamId': 'hometeamId'})
# Suffix all columns except merge keys
for col in [c for c in home_team_game_stats.columns if c not in ['gameId', 'hometeamId']]:
    home_team_game_stats = home_team_game_stats.rename(columns={col: f'{col}_home_agg'})
df_games = pd.merge(
    df_games,
    home_team_game_stats,
    left_on=['gameId', 'hometeamId'],
    right_on=['gameId', 'hometeamId'],
    how='left'
)

# Merge away team stats
away_team_game_stats = team_game_stats.copy()
away_team_game_stats = away_team_game_stats.rename(columns={'playerteamId': 'awayteamId'})
# Suffix all columns except merge keys
for col in [c for c in away_team_game_stats.columns if c not in ['gameId', 'awayteamId']]:
    away_team_game_stats = away_team_game_stats.rename(columns={col: f'{col}_away_agg'})
df_games = pd.merge(
    df_games,
    away_team_game_stats,
    left_on=['gameId', 'awayteamId'],
    right_on=['gameId', 'awayteamId'],
    how='left'
)

print("Aggregated team stats merged.")

# Calculate score differential for clutch game identification
df_games['score_differential'] = abs(df_games['homeScore'] - df_games['awayScore'])

# --- Feature Engineering: Team-level Clutch Efficiency ---
print("Calculating team-level clutch efficiency metrics...")

CLUTCH_DIFFERENTIAL_THRESHOLD = 5

df_clutch_games_only = df_games[df_games['score_differential'] <= CLUTCH_DIFFERENTIAL_THRESHOLD].copy()

# Estimate possessions for home and away teams in clutch games
df_clutch_games_only['home_possessions'] = estimate_possessions(df_clutch_games_only, 'home')
df_clutch_games_only['away_possessions'] = estimate_possessions(df_clutch_games_only, 'away')

# Calculate clutch efficiency metrics for home team
df_clutch_games_only['home_clutch_points_per_100_possessions'] = (df_clutch_games_only['homeScore'] / df_clutch_games_only['home_possessions']) * 100
df_clutch_games_only['home_clutch_eFG%'] = ((df_clutch_games_only['fieldGoalsMade_home_agg'] + 0.5 * df_clutch_games_only['threePointersMade_home_agg']) / df_clutch_games_only['fieldGoalsAttempted_home_agg'])
df_clutch_games_only['home_clutch_TOV%'] = (df_clutch_games_only['turnovers_home_agg'] / df_clutch_games_only['home_possessions'])

# Calculate clutch efficiency metrics for away team
df_clutch_games_only['away_clutch_points_per_100_possessions'] = (df_clutch_games_only['awayScore'] / df_clutch_games_only['away_possessions']) * 100
df_clutch_games_only['away_clutch_eFG%'] = ((df_clutch_games_only['fieldGoalsMade_away_agg'] + 0.5 * df_clutch_games_only['threePointersMade_away_agg']) / df_clutch_games_only['fieldGoalsAttempted_away_agg'])
df_clutch_games_only['away_clutch_TOV%'] = (df_clutch_games_only['turnovers_away_agg'] / df_clutch_games_only['away_possessions'])

# Aggregate these clutch stats to team level (average over all clutch games played)
team_clutch_efficiency = pd.DataFrame()

home_clutch_eff_agg = df_clutch_games_only.groupby('hometeamId').agg(
    clutch_pp100=('home_clutch_points_per_100_possessions', 'mean'),
    clutch_eFG=('home_clutch_eFG%', 'mean'),
    clutch_TOV=('home_clutch_TOV%', 'mean')
).reset_index().rename(columns={'hometeamId': 'teamId'})

away_clutch_eff_agg = df_clutch_games_only.groupby('awayteamId').agg(
    clutch_pp100=('away_clutch_points_per_100_possessions', 'mean'),
    clutch_eFG=('away_clutch_eFG%', 'mean'),
    clutch_TOV=('away_clutch_TOV%', 'mean')
).reset_index().rename(columns={'awayteamId': 'teamId'})

team_clutch_efficiency = pd.concat([home_clutch_eff_agg, away_clutch_eff_agg]).groupby('teamId').mean().reset_index()

# Merge team clutch efficiency back into df_games
df_games = pd.merge(
    df_games,
    team_clutch_efficiency[['teamId', 'clutch_pp100', 'clutch_eFG', 'clutch_TOV']],
    left_on='hometeamId',
    right_on='teamId',
    how='left',
    suffixes=('', '_home_clutch_eff')
).drop(columns=['teamId'])

df_games = pd.merge(
    df_games,
    team_clutch_efficiency[['teamId', 'clutch_pp100', 'clutch_eFG', 'clutch_TOV']],
    left_on='awayteamId',
    right_on='teamId',
    how='left',
    suffixes=('', '_away_clutch_eff')
).drop(columns=['teamId'])

print("Team-level clutch efficiency metrics calculated and merged.")

# --- Feature Engineering: Team Clutch Win/Loss Rate ---
print("Calculating team clutch win/loss rates...")

CLUTCH_DIFFERENTIAL_THRESHOLD = 5

df_clutch_games = df_games[df_games['score_differential'] <= CLUTCH_DIFFERENTIAL_THRESHOLD].copy()

# Determine home team clutch wins/losses
df_clutch_games['home_clutch_win'] = (df_clutch_games['game_winner'] == df_clutch_games['hometeamId']).astype(int)
df_clutch_games['home_clutch_loss'] = (df_clutch_games['game_winner'] != df_clutch_games['hometeamId']).astype(int)

# Determine away team clutch wins/losses
df_clutch_games['away_clutch_win'] = (df_clutch_games['game_winner'] == df_clutch_games['awayteamId']).astype(int)
df_clutch_games['away_clutch_loss'] = (df_clutch_games['game_winner'] != df_clutch_games['awayteamId']).astype(int)

# Aggregate clutch stats by team
team_clutch_stats = pd.DataFrame()

# Home team stats
home_clutch_agg = df_clutch_games.groupby('hometeamId').agg(
    clutch_wins=('home_clutch_win', 'sum'),
    clutch_losses=('home_clutch_loss', 'sum'),
    clutch_games_played=('gameId', 'count')
).reset_index().rename(columns={'hometeamId': 'teamId'})

# Away team stats
away_clutch_agg = df_clutch_games.groupby('awayteamId').agg(
    clutch_wins=('away_clutch_win', 'sum'),
    clutch_losses=('away_clutch_loss', 'sum'),
    clutch_games_played=('gameId', 'count')
).reset_index().rename(columns={'awayteamId': 'teamId'})

# Combine home and away stats
team_clutch_stats = pd.concat([home_clutch_agg, away_clutch_agg]).groupby('teamId').sum().reset_index()

# Calculate clutch win rate
team_clutch_stats['clutch_win_rate'] = team_clutch_stats['clutch_wins'] / team_clutch_stats['clutch_games_played']
team_clutch_stats['clutch_win_rate'] = team_clutch_stats['clutch_win_rate'].fillna(0) # Handle division by zero if no clutch games

# Merge clutch stats back into df_games (for both home and away teams)
df_games = pd.merge(
    df_games,
    team_clutch_stats[['teamId', 'clutch_win_rate']].rename(columns={'clutch_win_rate': 'home_clutch_win_rate'}),
    left_on='hometeamId',
    right_on='teamId',
    how='left'
).drop(columns=['teamId'])

df_games = pd.merge(
    df_games,
    team_clutch_stats[['teamId', 'clutch_win_rate']].rename(columns={'clutch_win_rate': 'away_clutch_win_rate'}),
    left_on='awayteamId',
    right_on='teamId',
    how='left'
).drop(columns=['teamId'])
print("Team clutch win/loss rates calculated and merged.")

# --- Feature Engineering: Collapse Rate x Opponent Closing Strength (Proxy) ---
print("Calculating collapse rate x opponent closing strength proxy...")

df_games['home_collapse_x_opponent_closing_strength'] = (1 - df_games['home_clutch_win_rate']) * df_games['away_clutch_win_rate']
df_games['away_collapse_x_opponent_closing_strength'] = (1 - df_games['away_clutch_win_rate']) * df_games['home_clutch_win_rate']

print("Collapse rate x opponent closing strength proxy calculated.")

# --- Feature Engineering: Crowd Intensity ---
print("Calculating crowd intensity feature...")

# Load raw games data to get attendance
df_raw_games = pd.read_csv(os.path.join(raw_data_path, 'Games.csv'))
df_raw_games = df_raw_games[['gameId', 'hometeamId', 'attendance']]

# Load teams data to get arena capacity
# Assuming teams.csv is in data/metadata/lookup relative to the project root
df_teams = pd.read_csv(os.path.join(project_root, 'data/metadata/lookup/teams.csv'))
df_teams = df_teams[['team_id', 'arena']]

# Merge raw games with teams to get arena capacity for each game
df_games_with_capacity = pd.merge(
    df_raw_games,
    df_teams,
    left_on='hometeamId',
    right_on='team_id',
    how='left'
)

# Calculate crowd intensity
# Handle potential division by zero or missing capacity values
df_games_with_capacity['crowd_intensity'] = df_games_with_capacity.apply(
    lambda row: row['attendance'] / row['arena'] if pd.notna(row['attendance']) and pd.notna(row['arena']) and row['arena'] > 0 else 0,
    axis=1
)

# Merge crowd intensity into the main df_games DataFrame
df_games = pd.merge(
    df_games,
    df_games_with_capacity[['gameId', 'crowd_intensity']],
    left_on='gameId',
    right_on='gameId',
    how='left'
)
print("Crowd intensity feature calculated and merged.")

# --- Feature Engineering: Interaction Features ---
print("Calculating interaction features...")

df_games['home_lead_elasticity_x_away_comeback_rate'] = df_games['lead_elasticity_home_adv'] * df_games['comeback_rate_ht_last20_away_adv'] # Using halftime comeback rate as an example
df_games['home_comeback_rate_minus_away_comeback_rate'] = df_games['comeback_rate_ht_last20_home_adv'] - df_games['comeback_rate_ht_last20_away_adv'] # Using halftime comeback rate as an example

print("Interaction features calculated.")

# --- Save DataFrame with Aggregated Features ---
print("Saving DataFrame with aggregated features...")
df_games.to_parquet(os.path.join(processed_data_path, "games_with_team_stats.parquet"), index=False)
print("DataFrame saved to 'data/processed/games_with_team_stats.parquet'.")

