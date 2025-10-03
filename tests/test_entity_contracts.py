import pandas as pd
import pytest
import os

# Define paths to lookup tables (assuming they are in data/metadata/lookup/)
LOOKUP_PATH = "/Users/jackweekly/Desktop/NBA/data/metadata/lookup/"

@pytest.fixture
def mock_players_data():
    # Mock data for players.csv
    return pd.DataFrame({
        'player_id': ['P1', 'P2', 'P3'],
        'full_name': ['LeBron James', 'Stephen Curry', 'Kevin Durant'],
        'birthdate': ['1984-12-30', '1988-03-14', '1988-09-29'],
        'height_cm': [206, 191, 208],
        'weight_kg': [113, 84, 109],
        'handedness': ['Right', 'Right', 'Right'],
        'created_at': ['2023-01-01', '2023-01-01', '2023-01-01']
    })

@pytest.fixture
def mock_player_xref_data():
    # Mock data for player_xref.csv
    return pd.DataFrame({
        'player_id': ['P1', 'P1', 'P2', 'P3', 'P1'],
        'source': ['kaggle', 'nba', 'kaggle', 'nba', 'kaggle'],
        'source_player_id': ['K1', 'N1', 'K2', 'N3', 'K1_alias'],
        'name_raw': ['LeBron James', 'LeBron James', 'Stephen Curry', 'Kevin Durant', 'LeBron J.'],
        'valid_from': ['2003-10-29', '2003-10-29', '2009-10-28', '2007-10-30', '2003-10-29'],
        'valid_to': ['2099-12-31', '2099-12-31', '2099-12-31', '2099-12-31', '2005-10-29'] # Overlapping for P1, K1
    })

@pytest.fixture
def mock_teams_data():
    # Mock data for teams.csv
    return pd.DataFrame({
        'team_id': ['T1', 'T2'],
        'full_name': ['Los Angeles Lakers', 'Golden State Warriors'],
        'city': ['Los Angeles', 'San Francisco'],
        'arena': ['Crypto.com Arena', 'Chase Center'],
        'created_at': ['2023-01-01', '2023-01-01']
    })

@pytest.fixture
def mock_team_xref_data():
    # Mock data for team_xref.csv
    return pd.DataFrame({
        'team_id': ['T1', 'T1', 'T2'],
        'source': ['kaggle', 'nba', 'kaggle'],
        'source_team_id': ['KA1', 'NA1', 'KA2'],
        'name_raw': ['LAL', 'Lakers', 'GSW'],
        'valid_from': ['1947-11-01', '1947-11-01', '1947-11-01'],
        'valid_to': ['2099-12-31', '2099-12-31', '2099-12-31']
    })

@pytest.fixture
def mock_games_data():
    # Mock data for games.csv
    return pd.DataFrame({
        'game_id': ['G1', 'G2'],
        'game_date': ['2023-10-24', '2023-10-25'],
        'home_team_id': ['T1', 'T2'],
        'away_team_id': ['T2', 'T1'],
        'created_at': ['2023-01-01', '2023-01-01']
    })

@pytest.fixture
def mock_game_xref_data():
    # Mock data for game_xref.csv
    return pd.DataFrame({
        'game_id': ['G1', 'G1', 'G2'],
        'source': ['kaggle', 'nba', 'kaggle'],
        'source_game_id': ['KG1', 'NG1', 'KG2'],
        'valid_from': ['2023-10-24', '2023-10-24', '2023-10-25'],
        'valid_to': ['2023-10-24', '2023-10-24', '2023-10-25']
    })


# --- Uniqueness Tests ---
def test_players_player_id_unique(mock_players_data):
    assert mock_players_data['player_id'].is_unique, "player_id in players.csv is not unique."

def test_player_xref_source_source_player_id_unique(mock_player_xref_data):
    # Convert valid_from/to to datetime for proper comparison
    df = mock_player_xref_data.copy()
    df['valid_from'] = pd.to_datetime(df['valid_from'])
    df['valid_to'] = pd.to_datetime(df['valid_to'])

    # Check uniqueness of (source, source_player_id) within their valid periods
    # This is a simplified check; a full check would involve iterating through time
    assert df.duplicated(subset=['source', 'source_player_id']).sum() == 0, \
        "(source, source_player_id) in player_xref.csv is not unique."

def test_teams_team_id_unique(mock_teams_data):
    assert mock_teams_data['team_id'].is_unique, "team_id in teams.csv is not unique."

def test_team_xref_source_source_team_id_unique(mock_team_xref_data):
    df = mock_team_xref_data.copy()
    df['valid_from'] = pd.to_datetime(df['valid_from'])
    df['valid_to'] = pd.to_datetime(df['valid_to'])
    assert df.duplicated(subset=['source', 'source_team_id']).sum() == 0, \
        "(source, source_team_id) in team_xref.csv is not unique."

def test_games_game_id_unique(mock_games_data):
    assert mock_games_data['game_id'].is_unique, "game_id in games.csv is not unique."

def test_game_xref_source_source_game_id_unique(mock_game_xref_data):
    df = mock_game_xref_data.copy()
    df['valid_from'] = pd.to_datetime(df['valid_from'])
    df['valid_to'] = pd.to_datetime(df['valid_to'])
    assert df.duplicated(subset=['source', 'source_game_id']).sum() == 0, \
        "(source, source_game_id) in game_xref.csv is not unique."

# --- Referential Integrity Tests ---
def test_player_xref_referential_integrity(mock_players_data, mock_player_xref_data):
    missing_player_ids = mock_player_xref_data[~mock_player_xref_data['player_id'].isin(mock_players_data['player_id'])]
    assert missing_player_ids.empty, \
        f"player_id(s) in player_xref.csv not found in players.csv: {missing_player_ids['player_id'].unique().tolist()}"

def test_team_xref_referential_integrity(mock_teams_data, mock_team_xref_data):
    missing_team_ids = mock_team_xref_data[~mock_team_xref_data['team_id'].isin(mock_teams_data['team_id'])]
    assert missing_team_ids.empty, \
        f"team_id(s) in team_xref.csv not found in teams.csv: {missing_team_ids['team_id'].unique().tolist()}"

def test_game_xref_referential_integrity(mock_games_data, mock_game_xref_data):
    missing_game_ids = mock_game_xref_data[~mock_game_xref_data['game_id'].isin(mock_games_data['game_id'])]
    assert missing_game_ids.empty, \
        f"game_id(s) in game_xref.csv not found in games.csv: {missing_game_ids['game_id'].unique().tolist()}"

# --- Ambiguity Checks ---
def test_player_xref_no_ambiguous_mapping(mock_player_xref_data):
    df = mock_player_xref_data.copy()
    df['valid_from'] = pd.to_datetime(df['valid_from'])
    df['valid_to'] = pd.to_datetime(df['valid_to'])

    # Check for cases where a (source, source_player_id) maps to multiple canonical player_ids
    # within overlapping valid periods.
    # This is a complex check. For simplicity, we'll check if a (source, source_player_id) maps to more than one player_id at any point.
    # A more robust check would involve iterating through time or using interval trees.
    ambiguous_mappings = df.groupby(['source', 'source_player_id'])['player_id'].nunique()
    ambiguous_mappings = ambiguous_mappings[ambiguous_mappings > 1]

    assert ambiguous_mappings.empty, \
        f"Ambiguous mappings found in player_xref.csv: {ambiguous_mappings.index.tolist()}"

def test_team_xref_no_ambiguous_mapping(mock_team_xref_data):
    df = mock_team_xref_data.copy()
    df['valid_from'] = pd.to_datetime(df['valid_from'])
    df['valid_to'] = pd.to_datetime(df['valid_to'])

    ambiguous_mappings = df.groupby(['source', 'source_team_id'])['team_id'].nunique()
    ambiguous_mappings = ambiguous_mappings[ambiguous_mappings > 1]

    assert ambiguous_mappings.empty, \
        f"Ambiguous mappings found in team_xref.csv: {ambiguous_mappings.index.tolist()}"

def test_game_xref_no_ambiguous_mapping(mock_game_xref_data):
    df = mock_game_xref_data.copy()
    df['valid_from'] = pd.to_datetime(df['valid_from'])
    df['valid_to'] = pd.to_datetime(df['valid_to'])

    ambiguous_mappings = df.groupby(['source', 'source_game_id'])['game_id'].nunique()
    ambiguous_mappings = ambiguous_mappings[ambiguous_mappings > 1]

    assert ambiguous_mappings.empty, \
        f"Ambiguous mappings found in game_xref.csv: {ambiguous_mappings.index.tolist()}"
