import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import sys
import yaml

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.data_helpers import asof_join, calculate_entropy


# Load configuration
with open('/Users/jackweekly/Desktop/NBA/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define paths from config
raw_data_path = os.path.join(project_root, config['data_paths']['raw'])
processed_data_path = os.path.join(project_root, config['data_paths']['processed'])

# --- Elo Rating Calculation Function ---
def calculate_elo_ratings(df_games, K=20, initial_elo=1500):
    elo_ratings = {team_id: initial_elo for team_id in pd.concat([df_games['hometeamId'], df_games['awayteamId']]).unique()}
    elo_history = []

    for index, row in df_games.iterrows():
        home_team = row['hometeamId']
        away_team = row['awayteamId']
        home_score = row['homeScore']
        away_score = row['awayScore']
        winner = row['game_winner']

        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        elo_history.append({
            'gameId': row['gameId'],
            'hometeamId': home_team,
            'awayteamId': away_team,
            'home_elo_pre': home_elo,
            'away_elo_pre': away_elo
        })

        # Calculate expected scores
        expected_home_score = 1 / (1 + 10**((away_elo - home_elo) / 400))
        expected_away_score = 1 / (1 + 10**((home_elo - away_elo) / 400))

        # Determine actual score (win/loss/draw)
        if winner == home_team:
            actual_home_score = 1
            actual_away_score = 0
        elif winner == away_team:
            actual_home_score = 0
            actual_away_score = 1
        else:
            # This case should ideally not happen in NBA (no draws)
            actual_home_score = 0.5
            actual_away_score = 0.5

        # Update Elo ratings
        elo_ratings[home_team] += K * (actual_home_score - expected_home_score)
        elo_ratings[away_team] += K * (actual_away_score - expected_away_score)

    return pd.DataFrame(elo_history)

# --- Strength-of-Schedule Adjustments ---
def calculate_strength_of_schedule(df_games):
    df_games_copy = df_games.copy()
    print(f"df_games columns in calculate_strength_of_schedule: {df_games_copy.columns.tolist()}")

    # Calculate opponent elo for each game
    df_games_copy['home_opponent_elo'] = df_games_copy['away_elo_pre']
    df_games_copy['away_opponent_elo'] = df_games_copy['home_elo_pre']

    # Create a dataframe with all games for each team
    df_home = df_games_copy[['gameId', 'gameDate', 'hometeamId', 'home_opponent_elo']].rename(columns={'hometeamId': 'teamId', 'gameDate': 'gameDate', 'home_opponent_elo': 'opponent_elo'})
    df_away = df_games_copy[['gameId', 'gameDate', 'awayteamId', 'away_opponent_elo']].rename(columns={'awayteamId': 'teamId', 'gameDate': 'gameDate', 'away_opponent_elo': 'opponent_elo'})
    df_all_games = pd.concat([df_home, df_away]).sort_values(by='gameDate').reset_index(drop=True)

    # Calculate the average opponent elo for each team
    df_all_games['avg_opponent_elo'] = df_all_games.groupby('teamId')['opponent_elo'].transform(lambda x: x.expanding().mean())

    return df_all_games[['gameId', 'teamId', 'avg_opponent_elo']]


# Load processed data with aggregated team stats
# print("Loading processed data with aggregated team stats...")
df = pd.read_parquet(os.path.join(processed_data_path, "games_with_team_stats.parquet"))
df_team_stats = pd.read_csv(os.path.join(raw_data_path, "TeamStatistics.csv"))
df_games = pd.read_csv(os.path.join(raw_data_path, "Games.csv"))
df_player_stats = pd.read_parquet(os.path.join(processed_data_path, "player_stats_processed.parquet"))
players_df = pd.read_csv(os.path.join(raw_data_path, "Players.csv"))

print("Data loaded.")

# Calculate Elo ratings
# print("Calculating Elo ratings...")
elo_df = calculate_elo_ratings(df.sort_values(by='gameDate'))
# print("Elo ratings calculated.")

# Merge Elo ratings into the main DataFrame
df = pd.merge(
    df,
    elo_df[['gameId', 'hometeamId', 'home_elo_pre']],
    on=['gameId', 'hometeamId'],
    how='left'
)
df = pd.merge(
    df,
    elo_df[['gameId', 'awayteamId', 'away_elo_pre']],
    on=['gameId', 'awayteamId'],
    how='left'
)

# Calculate strength of schedule
print("Attempting to calculate strength of schedule...")
sos_df = calculate_strength_of_schedule(df[['gameId', 'gameDate', 'hometeamId', 'awayteamId', 'home_elo_pre', 'away_elo_pre']])
print(f"sos_df calculated. Type: {type(sos_df)}")

# Merge sos_df into the main DataFrame
df = pd.merge(df, sos_df, left_on=['gameId', 'hometeamId'], right_on=['gameId', 'teamId'], how='left', suffixes=('', '_home_sos'))
df = pd.merge(df, sos_df, left_on=['gameId', 'awayteamId'], right_on=['gameId', 'teamId'], how='left', suffixes=('', '_away_sos'))

# Calculate Elo difference
df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre']

# --- Placeholder for Strength-of-Schedule Adjustments ---
# This would involve adjusting team performance metrics (e.g., points scored/allowed)
# by the opponent's quality (e.g., average Elo rating of opponents).
# This could be calculated here and added to the lagged features.

# --- Travel Fatigue Features ---
# TODO: Implement travel fatigue features.
# This requires external data for stadium locations (latitude, longitude), timezones, and altitudes.
# print("Placeholder: Travel fatigue features would be calculated here if the required data was available.")


# Ensure gameDate is datetime and sort for lagged features
df['gameDate'] = pd.to_datetime(df['gameDate'])
df = df.sort_values(by=['gameDate', 'gameId']).reset_index(drop=True)
# print("Data sorted by gameDate and gameId.")

# --- Feature Engineering: Lagged Features ---
# print("Starting lagged feature engineering...")

# Identify all unique team IDs
all_team_ids = pd.concat([df['hometeamId'], df['awayteamId']]).unique()

# Prepare a list to store dataframes with lagged features
lagged_features_list = []

# Define features to lag (e.g., scores, aggregated stats)
# We need to consider both home and away scores/stats
features_to_lag_home = [col for col in df.columns if col.endswith('_home_agg')] + ['home_teamScore']
features_to_lag_away = [col for col in df.columns if col.endswith('_away_agg')] + ['away_teamScore']

# Iterate through each team to calculate lagged features
for team_id in all_team_ids:
    # Get games where the current team was the home team
    df_home_team = df[df['hometeamId'] == team_id].copy()
    print(f"Columns in df_home_team for team {team_id}: {df_home_team.columns.tolist()}")
    # Get games where the current team was the away team
    df_away_team = df[df['awayteamId'] == team_id].copy()
    print(f"Columns in df_away_team for team {team_id}: {df_away_team.columns.tolist()}")

    # Combine and sort all games for this team chronologically
    # This is crucial for calculating rolling averages correctly
    df_team_games = pd.concat([
        df_home_team.assign(teamId=team_id, is_home=1),
        df_away_team.assign(teamId=team_id, is_home=0)
    ]).sort_values(by=['gameDate', 'gameId']).reset_index(drop=True)
    print(f"Columns in df_team_games after concat for team {team_id}: {df_team_games.columns.tolist()}")

    # Calculate rest days for 'teamId'
    df_team_games['days_since_last_game'] = df_team_games['gameDate'].diff().dt.days.fillna(0)

    # Derive rest topology features
    df_team_games = df_team_games.set_index('gameDate').sort_index()
    df_team_games['b2b'] = (df_team_games['days_since_last_game'] == 1).astype(int)
    df_team_games['temp_ones'] = 1 # Temporary column of ones
    df_team_games['3in4'] = df_team_games['temp_ones'].rolling(window=pd.Timedelta(days=4), closed='right').count().shift(1).apply(lambda x: 1 if x >= 3 else 0)
    df_team_games['4in6'] = df_team_games['temp_ones'].rolling(window=pd.Timedelta(days=6), closed='right').count().shift(1).apply(lambda x: 1 if x >= 4 else 0)
    df_team_games = df_team_games.reset_index() # Reset index after rolling calculations

    # Calculate rolling averages for relevant stats
    # For simplicity, let's start with average points scored and allowed
    # We need to calculate these based on the perspective of 'teamId'

    # Points scored by 'teamId'
    df_team_games['team_points_scored'] = df_team_games.apply(
        lambda row: row['homeScore'] if row['is_home'] == 1 else row['awayScore'], axis=1
    )
    df_team_games['team_points_scored_adj'] = df_team_games.apply(
        lambda row: (row['homeScore'] * (row['avg_opponent_elo'] / 1500)) if row['is_home'] == 1 else (row['awayScore'] * (row['avg_opponent_elo_away_sos'] / 1500)), axis=1
    )
    # Points allowed by 'teamId'
    df_team_games['team_points_allowed'] = df_team_games.apply(
        lambda row: row['awayScore'] if row['is_home'] == 1 else row['homeScore'], axis=1
    )
    df_team_games['margin'] = df_team_games['team_points_scored'] - df_team_games['team_points_allowed']
    df_team_games['plusMinusPoints'] = df_team_games.apply(
        lambda row: row['plusMinusPoints_home_agg'] if row['is_home'] == 1 else row['plusMinusPoints_away_agg'], axis=1
    )
    df_team_games['threePointersAttempted'] = df_team_games.apply(
        lambda row: row['threePointersAttempted_home_agg'] if row['is_home'] == 1 else row['threePointersAttempted_away_agg'], axis=1
    )
    df_team_games['threePointersMade'] = df_team_games.apply(
        lambda row: row['threePointersMade_home_agg'] if row['is_home'] == 1 else row['threePointersMade_away_agg'], axis=1
    )
    df_team_games['team_points_allowed_adj'] = df_team_games.apply(
        lambda row: (row['awayScore'] * (row['avg_opponent_elo'] / 1500)) if row['is_home'] == 1 else (row['homeScore'] * (row['avg_opponent_elo_away_sos'] / 1500)), axis=1
    )

    # Define stats to calculate rolling averages for
    stats_for_rolling_avg = [
        'team_points_scored', 'team_points_allowed', 'margin', 'plusMinusPoints', 'threePointersAttempted', 'threePointersMade'
    ]

    # Define window sizes
    window_sizes = [5, 10, 20]

    for window_size in window_sizes:
        for stat in stats_for_rolling_avg:
            df_team_games[f'avg_{stat}_last_{window_size}'] = df_team_games[stat].shift(1).rolling(window=window_size).mean()

    # Calculate win/loss streak
    df_team_games['team_won'] = df_team_games.apply(
        lambda row: 1 if (row['is_home'] == 1 and row['game_winner'] == row['hometeamId']) or \
                        (row['is_home'] == 0 and row['game_winner'] == row['awayteamId']) else 0, axis=1
    )
    df_team_games['win_streak'] = df_team_games.groupby((df_team_games['team_won'] != df_team_games['team_won'].shift()).cumsum()).cumcount() + 1
    df_team_games.loc[df_team_games['team_won'] == 0, 'win_streak'] = 0 # Reset streak for losses

    # Calculate signed win/loss streak
    df_team_games['signed_win_streak'] = df_team_games.groupby((df_team_games['team_won'] != df_team_games['team_won'].shift()).cumsum()).cumcount() + 1
    df_team_games.loc[df_team_games['team_won'] == 0, 'signed_win_streak'] *= -1

    # Exponential decay averages (EWMA)
    ewma_alpha = 0.2 # Example alpha, can be tuned
    for stat in stats_for_rolling_avg:
        df_team_games[f'ewma_{stat}'] = df_team_games[stat].shift(1).ewm(alpha=ewma_alpha).mean()

    # Variance features (std dev of points scored/allowed)
    for window_size in [5, 10]: # Use smaller windows for variance
        for stat in stats_for_rolling_avg:
            df_team_games[f'std_{stat}_last_{window_size}'] = df_team_games[stat].shift(1).rolling(window=window_size).std()

    # Calculate skew, kurtosis, and quantile gaps for margin and plusMinusPoints
    for stat in ['margin', 'plusMinusPoints']:
        df_team_games[f'{stat}_skew'] = df_team_games[stat].shift(1).rolling(window=20).skew()
        df_team_games[f'{stat}_kurtosis'] = df_team_games[stat].shift(1).rolling(window=20).kurt()
        df_team_games[f'{stat}_quantile_gap'] = df_team_games[stat].shift(1).rolling(window=20).quantile(0.9) - df_team_games[stat].shift(1).rolling(window=20).quantile(0.1)

    # Calculate the volatility of 3P attempts and makes
    for stat in ['threePointersAttempted', 'threePointersMade']:
        df_team_games[f'{stat}_volatility'] = df_team_games[stat].shift(1).rolling(window=20).std()

    df_team_games['biggestLead'] = df_team_games.apply(
        lambda row: row['biggestLead_home_adv'] if row['is_home'] == 1 else row['biggestLead_away_adv'], axis=1
    )
    df_team_games['leadChanges'] = df_team_games.apply(
        lambda row: row['leadChanges_home_adv'] if row['is_home'] == 1 else row['leadChanges_away_adv'], axis=1
    )
    df_team_games['timesTied'] = df_team_games.apply(
        lambda row: row['timesTied_home_adv'] if row['is_home'] == 1 else row['timesTied_away_adv'], axis=1
    )

    # Calculate collapse and comeback rates
    df_team_games['collapse'] = ((df_team_games['biggestLead'] >= 10) & (df_team_games['team_won'] == 0)).astype(int)
    df_team_games['comeback'] = ((df_team_games['biggestLead'] <= -10) & (df_team_games['team_won'] == 1)).astype(int)

    df_team_games['collapse_rate'] = df_team_games['collapse'].shift(1).rolling(window=20).mean()
    df_team_games['comeback_rate'] = df_team_games['comeback'].shift(1).rolling(window=20).mean()

    # Create an interaction feature between the collapse rate and the team's Elo rating
    df_team_games['elo_x_collapse_rate'] = df_team_games['collapse_rate'] * df_team_games['home_elo_pre']

    # Create a 'style' feature for each team
    df_team_games['style'] = df_team_games.apply(
        lambda row: row[['pointsInThePaint_home_adv', 'pointsFastBreak_home_adv', 'pointsFromTurnovers_home_adv', 'pointsSecondChance_home_adv']].idxmax() if row['is_home'] == 1 else row[['pointsInThePaint_away_adv', 'pointsFastBreak_away_adv', 'pointsFromTurnovers_away_adv', 'pointsSecondChance_away_adv']].idxmax(), axis=1
    )

    # Calculate the rolling average of blocks per game for each team
    df_team_games['avg_blocks'] = df_team_games.apply(
        lambda row: row['blocks_home_agg'] if row['is_home'] == 1 else row['blocks_away_agg'], axis=1
    ).shift(1).rolling(window=20).mean()

    # Create the cross-style matchup feature
    df_team_games['paint_vs_rim_protection'] = ((df_team_games['style'] == 'pointsInThePaint') & (df_team_games['avg_blocks'] >= df_team_games['avg_blocks'].quantile(0.8))).astype(int)

    # Calculate star dependency
    df_player_stats_game = df_player_stats[df_player_stats['gameId'].isin(df_team_games['gameId'])]
    df_player_stats_game = df_player_stats_game.groupby(['gameId', 'playerteamId'])['points'].apply(lambda x: x.nlargest(3).sum()).reset_index()
    df_player_stats_game.rename(columns={'points': 'star_points'}, inplace=True)
    df_team_games = pd.merge(df_team_games, df_player_stats_game, left_on=['gameId', 'teamId'], right_on=['gameId', 'playerteamId'], how='left')
    df_team_games['star_dependency'] = df_team_games['star_points'] / df_team_games['team_points_scored']

    # Calculate bench depth
    df_bench_points = df_player_stats[df_player_stats['gameId'].isin(df_team_games['gameId'])]
    df_bench_points = df_bench_points[df_bench_points['numMinutes'] < 20]
    df_bench_points = df_bench_points.groupby(['gameId', 'playerteamId'])['points'].sum().reset_index()
    df_bench_points.rename(columns={'points': 'bench_points'}, inplace=True)
    df_team_games = pd.merge(df_team_games, df_bench_points, left_on=['gameId', 'teamId'], right_on=['gameId', 'playerteamId'], how='left')
    df_team_games['bench_depth'] = df_team_games['bench_points'] / df_team_games['team_points_scored']

    # Calculate the rolling average of minutes played for each player
    df_player_stats['avg_minutes'] = df_player_stats.groupby('personId')['numMinutes'].transform(lambda x: x.shift(1).rolling(window=5).mean())

    # Identify sudden dips in minutes for core players
    df_player_stats['minutes_dip'] = (df_player_stats['numMinutes'] < (df_player_stats['avg_minutes'] * 0.5)).astype(int)
    df_minutes_dip_agg = df_player_stats.groupby(['gameId', 'playerteamId'])['minutes_dip'].sum().reset_index()
    df_minutes_dip_agg.rename(columns={'minutes_dip': 'core_player_minutes_dip'}, inplace=True)
    df_team_games = pd.merge(df_team_games, df_minutes_dip_agg, left_on=['gameId', 'teamId'], right_on=['gameId', 'playerteamId'], how='left')

    # Calculate the entropy of the quarterly points
    df_team_games['scoring_entropy'] = df_team_games.apply(
        lambda row: calculate_entropy(row[['q1Points_home_adv', 'q2Points_home_adv', 'q3Points_home_adv', 'q4Points_home_adv']]) if row['is_home'] == 1 else calculate_entropy(row[['q1Points_away_adv', 'q2Points_away_adv', 'q3Points_away_adv', 'q4Points_away_adv']]), axis=1
    )

    # Calculate first-half and second-half points
    df_team_games['first_half_points'] = df_team_games.apply(
            lambda row: row['q1Points_home_adv'] + row['q2Points_home_adv'] if row['is_home'] == 1 else row['q1Points_away_adv'] + row['q2Points_away_adv'], axis=1
        )
    df_team_games['second_half_points'] = df_team_games.apply(
            lambda row: row['q3Points_home_adv'] + row['q4Points_home_adv'] if row['is_home'] == 1 else row['q3Points_away_adv'] + row['q4Points_away_adv'], axis=1
        )
    # Calculate first-half performance volatility
    df_team_games['first_half_volatility'] = abs(df_team_games['first_half_points'] - df_team_games['second_half_points']) / (df_team_games['first_half_points'] + df_team_games['second_half_points'])
    df_team_games['first_half_volatility_rolling'] = df_team_games['first_half_volatility'].shift(1).rolling(window=20).mean()

    # Calculate rolling coach win rates
    df_team_games['coachId'] = df_team_games.apply(
        lambda row: row['coachId_home_adv'] if row['is_home'] == 1 else row['coachId_away_adv'], axis=1
    )
    df_team_games['coach_win'] = df_team_games.apply(lambda row: 1 if row['game_winner'] == row['teamId'] else 0, axis=1)
    df_team_games['rolling_coach_win_rate'] = df_team_games.groupby('coachId')['coach_win'].transform(lambda x: x.shift(1).rolling(window=20).mean())

    # Identify coaching changes
    df_team_games['coach_change'] = (df_team_games.groupby('teamId')['coachId'].shift(1) != df_team_games['coachId']).astype(int)
    df_team_games['games_since_coach_change'] = df_team_games.groupby('teamId')['coach_change'].cumsum()
    df_team_games['new_coach_bump'] = ((df_team_games['games_since_coach_change'] > 0) & (df_team_games['games_since_coach_change'] <= 10)).astype(int)

    # Calculate talent density
    df_player_draft = pd.merge(df_player_stats, players_df[['personId', 'draftNumber']], left_on='personId', right_on='personId', how='left')
    df_player_draft = df_player_draft[df_player_draft['gameId'].isin(df_team_games['gameId'])]
    df_player_draft = df_player_draft[df_player_draft['numMinutes'] >= 20] # Top rotation players
    df_talent_density = df_player_draft.groupby(['gameId', 'playerteamId'])['draftNumber'].mean().reset_index()
    df_talent_density.rename(columns={'draftNumber': 'talent_density', 'playerteamId': 'talent_teamId'}, inplace=True)
    df_team_games = pd.merge(df_team_games, df_talent_density, left_on=['gameId', 'teamId'], right_on=['gameId', 'talent_teamId'], how='left')

    lagged_features_list.append(df_team_games[[
        'gameId', 'teamId', 'is_home',
        *[f'avg_{stat}_last_{ws}' for ws in window_sizes for stat in stats_for_rolling_avg],
        'days_since_last_game', 'win_streak', 'b2b', '3in4', '4in6',
        'signed_win_streak',
        *[f'ewma_{stat}' for stat in stats_for_rolling_avg],
        *[f'std_{stat}_last_{ws}' for ws in [5, 10] for stat in stats_for_rolling_avg],
        *[f'{stat}_skew' for stat in ['margin', 'plusMinusPoints']],
        *[f'{stat}_kurtosis' for stat in ['margin', 'plusMinusPoints']],
        *[f'{stat}_quantile_gap' for stat in ['margin', 'plusMinusPoints']],
        'scoring_entropy',
        *[f'{stat}_volatility' for stat in ['threePointersAttempted', 'threePointersMade']],
        'collapse_rate',
        'comeback_rate',
        'elo_x_collapse_rate',
        'style',
        'paint_vs_rim_protection',
        'star_dependency',
        'bench_depth',
        'core_player_minutes_dip',
        'first_half_volatility_rolling',
        'rolling_coach_win_rate',
        'new_coach_bump',
        'talent_density'
    ]])

# Concatenate all lagged features
df_lagged_features = pd.concat(lagged_features_list)

# Merge lagged features back into the main DataFrame
# print("Merging lagged features into main DataFrame...")

# Define all new lagged feature columns to merge
new_lagged_cols = [
    *[f'avg_{stat}_last_{ws}' for ws in window_sizes for stat in stats_for_rolling_avg],
    'days_since_last_game', 'win_streak', 'b2b', '3in4', '4in6',
    'signed_win_streak',
    *[f'ewma_{stat}' for stat in stats_for_rolling_avg],
    *[f'std_{stat}_last_{ws}' for ws in [5, 10] for stat in stats_for_rolling_avg],
    *[f'{stat}_skew' for stat in ['margin', 'plusMinusPoints']],
    *[f'{stat}_kurtosis' for stat in ['margin', 'plusMinusPoints']],
    *[f'{stat}_quantile_gap' for stat in ['margin', 'plusMinusPoints']],
    'scoring_entropy',
    *[f'{stat}_volatility' for stat in ['threePointersAttempted', 'threePointersMade']],
    'collapse_rate',
    'comeback_rate',
    'elo_x_collapse_rate',
    'style',
    'paint_vs_rim_protection',
    'star_dependency',
    'bench_depth',
    'core_player_minutes_dip',
    'first_half_volatility_rolling',
    'rolling_coach_win_rate',
    'new_coach_bump',
    'talent_density'
]

# Merge home team lagged features
df = pd.merge(
    df,
    df_lagged_features[df_lagged_features['is_home'] == 1][['gameId', 'teamId'] + new_lagged_cols].rename(columns={'teamId': 'hometeamId'}),
    on=['gameId', 'hometeamId'],
    how='left',
    suffixes=('', '_home_lag')
)

# Merge away team lagged features
df = pd.merge(
    df,
    df_lagged_features[df_lagged_features['is_home'] == 0][['gameId', 'teamId'] + new_lagged_cols].rename(columns={'teamId': 'awayteamId'}),
    on=['gameId', 'awayteamId'],
    how='left',
    suffixes=('', '_away_lag')
)

# print("Lagged features merged.")

# --- Save DataFrame with all Engineered Features ---
# print("Saving DataFrame with all engineered features...")
df.to_parquet(os.path.join(processed_data_path, "games_with_all_features.parquet"), index=False)
# print("DataFrame saved to 'data/processed/games_with_all_features.parquet'.")