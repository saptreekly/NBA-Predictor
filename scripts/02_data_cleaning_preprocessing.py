import pandas as pd
import os
import yaml
import re
import unicodedata

# Load configuration
with open('/Users/jackweekly/Desktop/NBA/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths from config, using absolute paths
raw_data_path = os.path.join(project_root, config['data_paths']['raw'])
processed_data_path = os.path.join(project_root, config['data_paths']['processed'])

# --- Helper Function: Name Normalization ---
def normalize_name(name):
    if pd.isna(name): return name
    name = str(name).upper() # Convert to uppercase
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8') # Remove accents
    name = re.sub(r'[^\w\s]', '', name) # Remove punctuation
    name = re.sub(r'\s+', ' ', name).strip() # Collapse whitespace
    return name

# --- Helper Function: Stale Data Check ---
def check_if_stale(filepath, max_staleness_minutes=1440):
    """
    Checks if a file is stale based on its last modification time.
    
    Args:
        filepath (str): The path to the file.
        max_staleness_minutes (int): The maximum allowed age of the file in minutes.
        
    Returns:
        bool: True if the file is stale, False otherwise.
    """
    if not os.path.exists(filepath):
        return True # File doesn't exist, so it's stale
        
    last_modified_time = pd.to_datetime(os.path.getmtime(filepath), unit='s')
    if (pd.Timestamp.now() - last_modified_time).total_seconds() / 60 > max_staleness_minutes:
        return True
    return False


# Load raw data
print("Loading raw data...")
games_df = pd.read_csv(os.path.join(raw_data_path, "Games.csv"))
players_df = pd.read_csv(os.path.join(raw_data_path, "Players.csv"))
player_stats_df = pd.read_csv(os.path.join(raw_data_path, "PlayerStatistics.csv"))

# Load teams data for team ID lookup
teams_lookup_df = pd.read_csv(os.path.join(project_root, 'data/metadata/lookup/teams.csv'))
teams_lookup_df = teams_lookup_df[['team_id', 'full_name']]

# Merge player stats with teams lookup to get playerteamId
player_stats_df = pd.merge(
    player_stats_df,
    teams_lookup_df,
    left_on='playerteamName',
    right_on='full_name',
    how='left'
)
player_stats_df.rename(columns={'team_id': 'playerteamId'}, inplace=True)
team_stats_df = pd.read_csv(os.path.join(raw_data_path, "TeamStatistics.csv"))
print("Raw data loaded.")

# --- Data Quality Checks (Pre-processing) ---
# Freshness Checks:
raw_files_to_check = {
    "Games.csv": 1440,
    "Players.csv": 1440,
    "PlayerStatistics.csv": 1440,
    "TeamStatistics.csv": 1440,
}

for filename, staleness_threshold in raw_files_to_check.items():
    filepath = os.path.join(raw_data_path, filename)
    if check_if_stale(filepath, staleness_threshold):
        print(f"Warning: {filename} is stale (older than {staleness_threshold} minutes).")

# Source Drift Sentinels:
# TODO: Implement schema hash comparison or checksum validation.
print("Placeholder: Source drift sentinel checks would run here.")

# --- Data Cleaning and Preprocessing ---
print("Starting data cleaning and preprocessing...")

# Apply name normalization
players_df['display_name'] = players_df['firstName'] + ' ' + players_df['lastName']
players_df['display_name_normalized'] = players_df['display_name'].apply(normalize_name)
player_stats_df['playerteamName_normalized'] = player_stats_df['playerteamName'].apply(normalize_name)
# Assuming player_stats_df also has a player_name column that needs normalization
if 'player_name' in player_stats_df.columns:
    player_stats_df['player_name_normalized'] = player_stats_df['player_name'].apply(normalize_name)
print("Applied name normalization to player data.")

# --- Helper Function: Resolve Team Aliases ---
def resolve_team_alias(team_name, game_date, team_alias_df):
    if team_alias_df is None or team_alias_df.empty: # Handle case where alias table is empty
        return team_name # Return original name if no alias table

    # Filter aliases valid for the given game_date
    valid_aliases = team_alias_df[
        (team_alias_df['alias_name'] == team_name) &
        (team_alias_df['valid_from'] <= game_date) &
        (team_alias_df['valid_to'] >= game_date)
    ]

    if not valid_aliases.empty:
        return valid_aliases.iloc[0]['team_id']
    return team_name # Return original name if no alias found

# Load team alias data (if available)
team_alias_df = None
team_alias_path = os.path.join(processed_data_path, "..", "metadata", "lookup", "team_alias.csv")
if os.path.exists(team_alias_path) and os.path.getsize(team_alias_path) > 0:
    team_alias_df = pd.read_csv(team_alias_path)
    team_alias_df['valid_from'] = pd.to_datetime(team_alias_df['valid_from'])
    team_alias_df['valid_to'] = pd.to_datetime(team_alias_df['valid_to'])
    print("Loaded team alias data.")
else:
    print("Team alias file not found or is empty. Skipping team alias resolution.")

# Convert gameDate to datetime
games_df['gameDate'] = pd.to_datetime(games_df['gameDate'], utc=True)
player_stats_df['gameDate'] = pd.to_datetime(player_stats_df['gameDate'], utc=True)
print("Converted 'gameDate' columns to datetime (UTC).")

# Apply team alias resolution
if team_alias_df is not None and not team_alias_df.empty:
    games_df['hometeamId_canonical'] = games_df.apply(lambda row: resolve_team_alias(row['hometeamName'], row['gameDate'], team_alias_df), axis=1)
    games_df['awayteamId_canonical'] = games_df.apply(lambda row: resolve_team_alias(row['awayteamName'], row['gameDate'], team_alias_df), axis=1)
    print("Applied team alias resolution.")
else:
    games_df['hometeamId_canonical'] = games_df['hometeamId'] # Use original if no alias resolution
    games_df['awayteamId_canonical'] = games_df['awayteamId'] # Use original if no alias resolution


# --- Constraint Test: IDs unique per date ---
# Ensure gameId is unique for each gameDate
if not (games_df.groupby('gameDate')['gameId'].nunique() == games_df.groupby('gameDate')['gameId'].size()).all():
    raise ValueError("Constraint Violation: gameId is not unique for all gameDates in games_df.")
print("Constraint Test Passed: gameId is unique per gameDate.")


# Handle missing values in PlayerStatistics.csv (fill with 0 for stats)
stats_columns = [
    'numMinutes', 'points', 'assists', 'blocks', 'steals',
    'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalsPercentage',
    'threePointersAttempted', 'threePointersMade', 'threePointersPercentage',
    'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowsPercentage',
    'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
    'foulsPersonal', 'turnovers', 'plusMinusPoints'
]
for col in stats_columns:
    if col in player_stats_df.columns:
        player_stats_df[col] = player_stats_df[col].fillna(0)
print("Filled missing player statistics with 0.")

# For Games.csv, attendance, gameLabel, gameSubLabel, seriesGameNumber
# will be handled later if needed, or during feature engineering.
# For now, we'll leave them as is.

print("Data cleaning and preprocessing complete.")

# --- Define Second-Order Event Labels (Postgame Data Only) ---
# These labels capture specific game dynamics and are derived from postgame data.
# They should NEVER be used as pregame features directly.

# TODO: Implement Collapse, Comeback, Volatility Spike, Whistle Tilt, Three-point Swing, and Tempo Shock labels.
# These require play-by-play or detailed quarter-by-quarter scores, which are not currently available.
print("Placeholder: Second-order event labels would be calculated here if the required data was available.")


# --- Save Processed Data ---
# Idempotency: For true idempotency, we would need to read the existing data,
# merge with the new data, and then write back. For now, we are overwriting the files.
print("Saving processed games data to 'games_processed.parquet'...")
games_df.to_parquet(os.path.join(processed_data_path, "games_processed.parquet"), index=False)
print("Processed games data saved.")

print("Saving processed player stats data to 'player_stats_processed.parquet'...")
player_stats_df.to_parquet(os.path.join(processed_data_path, "player_stats_processed.parquet"), index=False)
print("Processed player stats data saved.")


# --- Row-Count Sanity Check ---
# TODO: Implement a check to compare the number of games in the schedule with the number of predictions.
print("Placeholder: Row-count sanity check would run here.")


# --- Dead-Letter Queue ---
# TODO: Implement a dead-letter queue for games that fail processing.
print("Placeholder: Dead-letter queue logic would be implemented here.")


# --- Time Zone Single-Source-of-Truth ---
# Ensure all timestamps are stored in UTC. Convert to local time zones only for UI display.
# For example, 'gameDate' should be treated as UTC internally.

# --- Drift Alert Integration Point ---
# TODO: Implement a drift alert mechanism to compare new data against historical baselines.
print("Placeholder: Drift alert integration point would be here.")

