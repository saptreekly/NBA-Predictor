import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Define paths
current_dir = os.path.dirname(__file__)
processed_data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed'))
metadata_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'metadata', 'lookup'))

# Load configuration
with open(os.path.join(current_dir, '..', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

print("Loading processed data for closer model...")
df_games = pd.read_parquet(os.path.join(processed_data_path, "games_with_lineup_consistency.parquet"))
df_player_stats = pd.read_parquet(os.path.join(processed_data_path, "player_stats_processed.parquet"))
df_players_metadata = pd.read_csv(os.path.join(metadata_path, 'players.csv'))

# Ensure gameId and playerteamId are of consistent type for merging
df_player_stats['gameId'] = df_player_stats['gameId'].astype(str)
df_player_stats['playerteamId'] = df_player_stats['playerteamId'].astype(str)
df_games['gameId'] = df_games['gameId'].astype(str)
df_games['hometeamId'] = df_games['hometeamId'].astype(str)
df_games['awayteamId'] = df_games['awayteamId'].astype(str)

# Merge player metadata with player stats
df_players_metadata = df_players_metadata[['personId', 'position', 'height', 'weight']] # Add weight
df_player_stats = pd.merge(df_player_stats, df_players_metadata, on='personId', how='left')

# --- Tier 0: Labels + dataset assembly ---
print("Assembling player-game training table...")

# 1. Closer Label: A player is a closer if they are in the team’s top-5 minutes that game.
# We need df_top5_minutes which is created in 04_lineup_consistency_features.py
# For this script, we will re-create df_top5_minutes
df_top5_minutes = df_player_stats.set_index(['gameId', 'playerteamId']).groupby(level=[0, 1])['numMinutes'].nlargest(5).reset_index()
df_top5_minutes = pd.merge(df_top5_minutes, df_player_stats[['gameId', 'playerteamId', 'personId', 'numMinutes']], on=['gameId', 'playerteamId', 'numMinutes'], how='left')
df_top5_minutes = df_top5_minutes.groupby(['gameId', 'playerteamId'])['personId'].apply(list).reset_index(name='top5_players')

# Create a flattened DataFrame of top 5 players for easier merging
df_top5_players_flat = df_top5_minutes.explode('top5_players')
df_top5_players_flat.rename(columns={'top5_players': 'personId'}, inplace=True)
df_top5_players_flat = df_top5_players_flat[['gameId', 'playerteamId', 'personId']].drop_duplicates()

# Create the player-game training table
df_player_game = df_player_stats.copy()

# Label 'closer'
df_player_game['closer_label'] = 0
df_player_game.loc[df_player_game.set_index(['gameId', 'playerteamId', 'personId']).index.isin(df_top5_players_flat.set_index(['gameId', 'playerteamId', 'personId']).index), 'closer_label'] = 1

# Predictors (pregame or season-to-date)
# Sort by player and date for rolling calculations
df_player_game = df_player_game.sort_values(by=['personId', 'gameDate']).reset_index(drop=True)

# Recent minutes share (EWMA last 10)
# Calculate total team minutes for each game
team_minutes = df_player_game.groupby(['gameId', 'playerteamId'])['numMinutes'].sum().reset_index()
team_minutes.rename(columns={'numMinutes': 'team_total_minutes'}, inplace=True)
df_player_game = pd.merge(df_player_game, team_minutes, on=['gameId', 'playerteamId'], how='left')
df_player_game['minutes_share'] = df_player_game['numMinutes'] / df_player_game['team_total_minutes']
df_player_game['minutes_share'] = df_player_game['minutes_share'].fillna(0).replace([np.inf, -np.inf], 0)

# EWMA last 10 games for minutes share
df_player_game['recent_minutes_share_ewma'] = df_player_game.groupby('personId')['minutes_share'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

# Starter flag, bench role flag
# For simplicity, starter = closer_label for now. Bench role = 1 - starter.
# This is a simplification, as a starter might not be a closer.
# A more accurate starter flag would require external data.
df_player_game['is_starter'] = df_player_game['closer_label'] # Proxy for now
df_player_game['is_bench'] = 1 - df_player_game['is_starter']

# Role/size: position, height, weight (bucket into small/wing/big)
# Simplify position bucketing
def bucket_position(pos):
    if pd.isna(pos): return 'Unknown'
    if 'G' in pos: return 'Guard'
    if 'F' in pos: return 'Forward'
    if 'C' in pos: return 'Center'
    return 'Unknown'
df_player_game['position_bucket'] = df_player_game['position'].apply(bucket_position)

# Usage & shooting gravity proxies: last-10 usage% (or FGA share), 3PA rate.
# FGA share as usage proxy
team_fga = df_player_game.groupby(['gameId', 'playerteamId'])['fieldGoalsAttempted'].sum().reset_index()
team_fga.rename(columns={'fieldGoalsAttempted': 'team_total_fga'}, inplace=True)
df_player_game = pd.merge(df_player_game, team_fga, on=['gameId', 'playerteamId'], how='left')
df_player_game['fga_share'] = df_player_game['fieldGoalsAttempted'] / df_player_game['team_total_fga']
df_player_game['fga_share'] = df_player_game['fga_share'].fillna(0).replace([np.inf, -np.inf], 0)

df_player_game['recent_fga_share_ewma'] = df_player_game.groupby('personId')['fga_share'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

# 3PA rate
df_player_game['3pa_rate'] = df_player_game['threePointersAttempted'] / df_player_game['fieldGoalsAttempted']
df_player_game['3pa_rate'] = df_player_game['3pa_rate'].fillna(0).replace([np.inf, -np.inf], 0)
df_player_game['recent_3pa_rate_ewma'] = df_player_game.groupby('personId')['3pa_rate'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

# Foul rate (PF per 36), turnover rate.
df_player_game['fouls_per_36'] = (df_player_game['foulsPersonal'] / df_player_game['numMinutes']) * 36
df_player_game['fouls_per_36'] = df_player_game['fouls_per_36'].fillna(0).replace([np.inf, -np.inf], 0)
df_player_game['recent_fouls_per_36_ewma'] = df_player_game.groupby('personId')['fouls_per_36'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

df_player_game['turnovers_per_36'] = (df_player_game['turnovers'] / df_player_game['numMinutes']) * 36
df_player_game['turnovers_per_36'] = df_player_game['turnovers_per_36'].fillna(0).replace([np.inf, -np.inf], 0)
df_player_game['recent_turnovers_per_36_ewma'] = df_player_game.groupby('personId')['turnovers_per_36'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

# Team context: lineup entropy last 10, injury multipliers, churn score.
# Merge team-level features from df_games
df_player_game = pd.merge(
    df_player_game,
    df_games[['gameId', 'hometeamId', 'awayteamId', 'home_closing_entropy_last10', 'away_closing_entropy_last10',
              'home_delta_churn', 'away_delta_churn', 'home_weighted_injury_impact', 'away_weighted_injury_impact']],
    on='gameId',
    how='left'
)

# Assign correct team context features to each player
df_player_game['team_closing_entropy_last10'] = np.where(
    df_player_game['playerteamId'] == df_player_game['hometeamId'],
    df_player_game['home_closing_entropy_last10'],
    df_player_game['away_closing_entropy_last10']
)
df_player_game['team_delta_churn'] = np.where(
    df_player_game['playerteamId'] == df_player_game['hometeamId'],
    df_player_game['home_delta_churn'],
    df_player_game['away_delta_churn']
)
df_player_game['team_weighted_injury_impact'] = np.where(
    df_player_game['playerteamId'] == df_player_game['hometeamId'],
    df_player_game['home_weighted_injury_impact'],
    df_player_game['away_weighted_injury_impact']
)

# Clean up merged columns
df_player_game.drop(columns=[
    'hometeamId', 'awayteamId', 'home_closing_entropy_last10', 'away_closing_entropy_last10',
    'home_delta_churn', 'away_delta_churn', 'home_weighted_injury_impact', 'away_weighted_injury_impact'
], inplace=True)

# Season and team IDs; player ID as a key for partial pooling later.
df_player_game['season'] = pd.to_datetime(df_player_game['gameDate']).dt.year # Assuming gameDate is datetime

# --- "Plus defender" proxy ---
print("Calculating 'plus defender' proxy...")

# Standardize per-36 STL, BLK, DRB% (or DRB per 36), PF (negatively), TOV (negatively).
# DRB% needs total rebounds for opponent. For simplicity, use DRB per 36.
df_player_game['drb_per_36'] = (df_player_game['reboundsDefensive'] / df_player_game['numMinutes']) * 36
df_player_game['drb_per_36'] = df_player_game['drb_per_36'].fillna(0).replace([np.inf, -np.inf], 0)

# Calculate rolling averages for defensive stats
df_player_game['recent_stl_per_36'] = df_player_game.groupby('personId')['steals'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1)) * 36 / df_player_game['numMinutes'] # Placeholder for per 36
df_player_game['recent_blk_per_36'] = df_player_game.groupby('personId')['blocks'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1)) * 36 / df_player_game['numMinutes'] # Placeholder for per 36
df_player_game['recent_drb_per_36'] = df_player_game.groupby('personId')['drb_per_36'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))
df_player_game['recent_pf_per_36'] = df_player_game.groupby('personId')['fouls_per_36'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))
df_player_game['recent_tov_per_36'] = df_player_game.groupby('personId')['turnovers_per_36'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

# Standardize within position/season
def standardize_within_group(df, column, group_cols):
    df[f'z_{column}'] = df.groupby(group_cols)[column].transform(lambda x: (x - x.mean()) / x.std())
    return df

df_player_game = standardize_within_group(df_player_game, 'recent_stl_per_36', ['position_bucket', 'season'])
df_player_game = standardize_within_group(df_player_game, 'recent_blk_per_36', ['position_bucket', 'season'])
df_player_game = standardize_within_group(df_player_game, 'recent_drb_per_36', ['position_bucket', 'season'])
df_player_game = standardize_within_group(df_player_game, 'recent_pf_per_36', ['position_bucket', 'season'])
df_player_game = standardize_within_group(df_player_game, 'recent_tov_per_36', ['position_bucket', 'season'])

# Fill NaN from standardization (e.g., if std dev is 0)
for col in ['z_recent_stl_per_36', 'z_recent_blk_per_36', 'z_recent_drb_per_36', 'z_recent_pf_per_36', 'z_recent_tov_per_36']:
    df_player_game[col] = df_player_game[col].fillna(0)

df_player_game['def_plus_z'] = df_player_game['z_recent_stl_per_36'] + df_player_game['z_recent_blk_per_36'] + \
                               df_player_game['z_recent_drb_per_36'] - df_player_game['z_recent_pf_per_36'] - \
                               df_player_game['z_recent_tov_per_36']

# Convert to percentile within position/season
df_player_game['def_plus_percentile'] = df_player_game.groupby(['position_bucket', 'season'])['def_plus_z'].transform(lambda x: x.rank(pct=True) * 100)

print("Player-game training table assembled.")

# Save the assembled dataset
df_player_game.to_parquet(os.path.join(processed_data_path, "player_game_closer_data.parquet"), index=False)
print(f"Player-game closer data saved to {os.path.join(processed_data_path, 'player_game_closer_data.parquet')}")

# --- Tier 1: Frequentist mixed-effects GLMM (scikit-learn approximation) ---
print("Fitting Tier 1 closer model...")

# Define features (X) and target (y)
features = [
    'recent_minutes_share_ewma', 'is_starter', 'is_bench',
    'height', 'weight', # Weight will have NaNs, need to handle
    'recent_fga_share_ewma', 'recent_3pa_rate_ewma',
    'recent_fouls_per_36_ewma', 'recent_turnovers_per_36_ewma',
    'team_closing_entropy_last10', 'team_delta_churn', 'team_weighted_injury_impact'
]
categorical_features = ['position_bucket'] # For one-hot encoding

# Handle NaNs in features (e.g., fill with 0 or mean)
for col in ['height', 'weight', 'recent_minutes_share_ewma', 'recent_fga_share_ewma', 'recent_3pa_rate_ewma',
            'recent_fouls_per_36_ewma', 'recent_turnovers_per_36_ewma', 'team_closing_entropy_last10',
            'team_delta_churn', 'team_weighted_injury_impact']:
    if col in df_player_game.columns:
        df_player_game[col] = df_player_game[col].fillna(0) # Simple fill for now

# One-hot encode categorical features
df_player_game = pd.get_dummies(df_player_game, columns=categorical_features, prefix=categorical_features)
features.extend([col for col in df_player_game.columns if any(cat_feat in col for cat_feat in categorical_features)])
features = [f for f in features if f in df_player_game.columns] # Ensure all features exist

X = df_player_game[features]
y = df_player_game['closer_label']

# Target encoding for random effects (player, team, season)
# K-fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store target encoded features
df_player_game['re_player'] = 0.0
df_player_game['re_team'] = 0.0
df_player_game['re_season'] = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(df_player_game)):
    df_train = df_player_game.iloc[train_idx]
    df_val = df_player_game.iloc[val_idx]

    # Player target encoding
    player_means = df_train.groupby('personId')['closer_label'].mean()
    df_player_game.loc[val_idx, 're_player'] = df_val['personId'].map(player_means).fillna(player_means.mean())

    # Team target encoding
    team_means = df_train.groupby('playerteamId')['closer_label'].mean()
    df_player_game.loc[val_idx, 're_team'] = df_val['playerteamId'].map(team_means).fillna(team_means.mean())

    # Season target encoding
    season_means = df_train.groupby('season')['closer_label'].mean()
    df_player_game.loc[val_idx, 're_season'] = df_val['season'].map(season_means).fillna(season_means.mean())

# Add target encoded features to X
X['re_player'] = df_player_game['re_player']
X['re_team'] = df_player_game['re_team']
X['re_season'] = df_player_game['re_season']

# Fit Logistic Regression model
model = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=42) # C is inverse of regularization strength
model.fit(X, y)

# Predict p_closer_hat
df_player_game['p_closer_hat'] = model.predict_proba(X)[:, 1]

print("Tier 1 closer model fitted and predictions made.")

# --- Deliverables ---
# closer_probs.parquet
df_closer_probs = df_player_game[['gameId', 'playerteamId', 'personId', 'p_closer_hat']].copy()
df_closer_probs.rename(columns={'playerteamId': 'teamId'}, inplace=True)
df_closer_probs.to_parquet(os.path.join(processed_data_path, "closer_probs.parquet"), index=False)
print(f"Closer probabilities saved to {os.path.join(processed_data_path, 'closer_probs.parquet')}")

# closer_team_aggregates.parquet
print("Calculating team-level aggregates...")

# E[plus_defenders_closing] = Σ p_closer_i * 1[def_plus_i ≥ 70th pct].
df_player_game['is_plus_defender'] = (df_player_game['def_plus_percentile'] >= 70).astype(int)
df_player_game['expected_plus_defenders'] = df_player_game['p_closer_hat'] * df_player_game['is_plus_defender']

df_closer_team_aggregates = df_player_game.groupby(['gameId', 'playerteamId']).agg(
    E_plus_defenders_closing=('expected_plus_defenders', 'sum'),
    # Add other aggregates here as needed
).reset_index()
df_closer_team_aggregates.rename(columns={'playerteamId': 'teamId'}, inplace=True)

# Merge with df_games to get home/away team context
df_closer_team_aggregates_home = df_closer_team_aggregates.rename(columns={'teamId': 'hometeamId'})
df_closer_team_aggregates_away = df_closer_team_aggregates.rename(columns={'teamId': 'awayteamId'})

df_games = pd.merge(df_games, df_closer_team_aggregates_home, on=['gameId', 'hometeamId'], how='left', suffixes=('', '_home_agg'))
df_games = pd.merge(df_games, df_closer_team_aggregates_away, on=['gameId', 'awayteamId'], how='left', suffixes=('', '_away_agg'))

df_games.rename(columns={
    'E_plus_defenders_closing': 'home_E_plus_defenders_closing',
    'E_plus_defenders_closing_away_agg': 'away_E_plus_defenders_closing'
}, inplace=True)

# Placeholder for E_spacing_closing, uncertainty_closing
df_games['home_E_spacing_closing'] = np.nan
df_games['away_E_spacing_closing'] = np.nan
df_games['home_uncertainty_closing'] = np.nan
df_games['away_uncertainty_closing'] = np.nan

df_closer_team_aggregates = df_games[['gameId', 'hometeamId', 'awayteamId',
                                      'home_E_plus_defenders_closing', 'away_E_plus_defenders_closing',
                                      'home_E_spacing_closing', 'away_E_spacing_closing',
                                      'home_uncertainty_closing', 'away_uncertainty_closing']].copy()

df_closer_team_aggregates.to_parquet(os.path.join(processed_data_path, "closer_team_aggregates.parquet"), index=False)
print(f"Closer team aggregates saved to {os.path.join(processed_data_path, 'closer_team_aggregates.parquet')}")

print("Closer model script finished.")
