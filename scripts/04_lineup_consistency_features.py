import pandas as pd
import os
import yaml
from scipy.stats import entropy
import numpy as np
import multiprocessing
import sys

# Global variables for multiprocessing
_df_games_global = None
_df_top5_minutes_global = None
_df_players_metadata_global = None
_df_player_stats_global = None

def init_worker(df_games_shared, df_top5_minutes_shared, df_players_metadata_shared, df_player_stats_shared):
    global _df_games_global
    global _df_top5_minutes_global
    global _df_players_metadata_global
    global _df_player_stats_global
    _df_games_global = df_games_shared
    _df_top5_minutes_global = df_top5_minutes_shared
    _df_players_metadata_global = df_players_metadata_shared
    _df_player_stats_global = df_player_stats_shared

# Function to calculate entropy of a list of lists (lineups)
def calculate_lineup_entropy(lineups_list):
    if not lineups_list:
        return 0.0 # No lineups, no entropy

    # Convert list of player IDs to sorted tuples for consistent hashing
    lineups_tuples = [tuple(sorted(l)) for l in lineups_list if l is not None]

    if not lineups_tuples:
        return 0.0

    # Calculate frequency distribution
    lineup_counts = pd.Series(lineups_tuples).value_counts()
    probabilities = lineup_counts / lineup_counts.sum()

    # Calculate Shannon entropy
    return entropy(probabilities)

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if not s1 and not s2:
        return 1.0 # Both sets are empty, consider them 100% similar
    if not s1 or not s2:
        return 0.0 # One set is empty, the other is not
    return len(s1.intersection(s2)) / len(s1.union(s2))

def get_player_positions(player_ids):
    df_meta = _df_players_metadata_global
    positions = df_meta[df_meta['personId'].isin(player_ids)]['position'].tolist()
    return positions

def get_player_heights(player_ids):
    df_meta = _df_players_metadata_global
    heights = df_meta[df_meta['personId'].isin(player_ids)]['height'].tolist()
    return heights

N_GAMES = 10 # Number of games to consider for rolling entropy

def process_game(game_idx_and_data):
    i, game = game_idx_and_data
    home_team_id = game['hometeamId']
    away_team_id = game['awayteamId']
    current_game_date = game['gameDate']

    # Access global DataFrames
    df_games_local = _df_games_global
    df_top5_minutes_local = _df_top5_minutes_global

    # Get historical games for home team
    home_historical_games = df_games_local[
        (df_games_local['gameDate'] < current_game_date) &
        ((df_games_local['hometeamId'] == home_team_id) | (df_games_local['awayteamId'] == home_team_id))
    ].sort_values(by='gameDate', ascending=False).head(N_GAMES)

    # Extract top5 players for home team from historical games
    home_lineups = []
    for _, hist_game in home_historical_games.iterrows():
        if hist_game['hometeamId'] == home_team_id:
            top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)].empty else None
        else: # away team in historical game was the current home team
            top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)].empty else None
        if top5:
            home_lineups.append(top5)

    home_entropy = calculate_lineup_entropy(home_lineups)

    # Calculate rolling Jaccard overlap for home team
    home_jaccard_overlaps = []
    current_home_top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)].empty else []

    for _, hist_game in home_historical_games.iterrows():
        if hist_game['hometeamId'] == home_team_id:
            hist_top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)].empty else []
        else: # away team in historical game was the current home team
            hist_top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == home_team_id)].empty else []
        home_jaccard_overlaps.append(jaccard_similarity(current_home_top5, hist_top5))
    home_jaccard_overlap_avg = np.mean(home_jaccard_overlaps) if home_jaccard_overlaps else 0.0

    # Get historical games for away team
    away_historical_games = df_games_local[
        (df_games_local['gameDate'] < current_game_date) &
        ((df_games_local['hometeamId'] == away_team_id) | (df_games_local['awayteamId'] == away_team_id))
    ].sort_values(by='gameDate', ascending=False).head(N_GAMES)

    # Extract top5 players for away team from historical games
    away_lineups = []
    for _, hist_game in away_historical_games.iterrows():
        if hist_game['hometeamId'] == away_team_id:
            top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)].empty else None
        else: # home team in historical game was the current away team
            top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)].empty else None
        if top5:
            away_lineups.append(top5)

    away_entropy = calculate_lineup_entropy(away_lineups)

    # Calculate rolling Jaccard overlap for away team
    away_jaccard_overlaps = []
    current_away_top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)].empty else []

    for _, hist_game in away_historical_games.iterrows():
        if hist_game['hometeamId'] == away_team_id:
            hist_top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)].empty else []
        else: # home team in historical game was the current away team
            hist_top5 = df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)]['top5_players'].iloc[0] if not df_top5_minutes_local[(df_top5_minutes_local['gameId'] == hist_game['gameId']) & (df_top5_minutes_local['playerteamId'] == away_team_id)].empty else []
        away_jaccard_overlaps.append(jaccard_similarity(current_away_top5, hist_top5))
    away_jaccard_overlap_avg = np.mean(away_jaccard_overlaps) if away_jaccard_overlaps else 0.0

    # Positional Balance and Switchability for Home Team
    home_top5_positions = get_player_positions(current_home_top5)
    home_top5_heights = get_player_heights(current_home_top5)

    # Positional Balance (simplified: count G, F, C)
    home_pos_counts = {'G': 0, 'F': 0, 'C': 0}
    for pos in home_top5_positions:
        if 'G' in pos: home_pos_counts['G'] += 1
        elif 'F' in pos or 'W' in pos: home_pos_counts['F'] += 1 # Assuming W maps to F
        elif 'C' in pos: home_pos_counts['C'] += 1
    home_pos_variance = np.var(list(home_pos_counts.values())) if home_pos_counts else 0.0

    # Switchability Index (std dev of heights)
    home_switchability_index = np.std(home_top5_heights) if len(home_top5_heights) > 1 else 0.0

    # Positional Balance and Switchability for Away Team
    away_top5_positions = get_player_positions(current_away_top5)
    away_top5_heights = get_player_heights(current_away_top5)

    # Positional Balance (simplified: count G, F, C)
    away_pos_counts = {'G': 0, 'F': 0, 'C': 0}
    for pos in away_top5_positions:
        if 'G' in pos: away_pos_counts['G'] += 1
        elif 'F' in pos or 'W' in pos: away_pos_counts['F'] += 1
        elif 'C' in pos: away_pos_counts['C'] += 1
    away_pos_variance = np.var(list(away_pos_counts.values())) if away_pos_counts else 0.0

    # Switchability Index (std dev of heights)
    away_switchability_index = np.std(away_top5_heights) if len(away_top5_heights) > 1 else 0.0

    # Access global DataFrames
    df_games_local = _df_games_global
    df_top5_minutes_local = _df_top5_minutes_global
    df_player_stats_local = _df_player_stats_global

    # --- No-Closer Counterfactual MC ---
    D = 1000 # Number of Monte Carlo simulations
    kappa = 50 # Dirichlet concentration parameter
    MINUTES_CAP = 42.0 # Max minutes for a player in a simulated draw

    # Helper to run No-Closer MC for a single team
    def run_no_closer_mc(team_id, current_game_id, current_top5_players, is_home_team):
        # Get player stats for the current game and team
        game_players = df_player_stats_local[
            (df_player_stats_local['gameId'] == current_game_id) &
            (df_player_stats_local['playerteamId'] == team_id)
        ].copy()

        if game_players.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # Get baseline minutes share and pts/min
        game_players['baseline_minutes_share'] = game_players['minutes_share_l10'].fillna(0)
        game_players['baseline_pts_per_min'] = game_players['pts_per_min_l10'].fillna(0)

        # Identify closers for this game
        closers_in_game = [p for p in current_top5_players if p in game_players['personId'].tolist()]

        # Baseline total points for the team
        baseline_pts_team = (game_players['baseline_minutes_share'] * 240 * game_players['baseline_pts_per_min']).sum()

        # Prepare for counterfactual: zero closer minutes
        game_players['cf_minutes_share'] = game_players['baseline_minutes_share'].copy()
        game_players.loc[game_players['personId'].isin(closers_in_game), 'cf_minutes_share'] = 0

        # Redistribute minutes to remaining rotation
        mask_keep = game_players['personId'].isin(game_players[~game_players['personId'].isin(closers_in_game)]['personId'].tolist())
        
        # Ensure sum of cf_minutes_share for non-closers is not zero before normalizing
        if game_players.loc[mask_keep, 'cf_minutes_share'].sum() == 0:
            # If no minutes left for non-closers, distribute equally or based on some other logic
            # For simplicity, if no one else played, just return baseline
            if len(game_players.loc[mask_keep]) > 0:
                game_players.loc[mask_keep, 'cf_minutes_share'] = 1 / len(game_players.loc[mask_keep])
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan

        w = game_players.loc[mask_keep, 'cf_minutes_share'] / game_players.loc[mask_keep, 'cf_minutes_share'].sum()

        mc_pts = []
        for d in range(D):
            # Draw from Dirichlet for minute redistribution
            r = np.random.dirichlet(kappa * w) * 240.0 # Scale to 240 total minutes

            m_draw = np.zeros(len(game_players))
            m_draw[mask_keep.values] = np.minimum(r, MINUTES_CAP) # Cap minutes

            # Optional: shrink rates when overextended (simplified)
            # For now, use baseline pts_per_min
            pts = (m_draw * game_players['baseline_pts_per_min']).sum()
            mc_pts.append(pts)

        delta_pts_off_mean = np.mean(mc_pts) - baseline_pts_team
        delta_pts_off_sd = np.std(mc_pts)
        delta_pts_off_p10 = np.percentile(mc_pts, 10) - baseline_pts_team
        delta_pts_off_p90 = np.percentile(mc_pts, 90) - baseline_pts_team

        # Risk tail: share of simulations where points drop significantly (e.g., > 5 points)
        risk_tail = np.mean(np.array(mc_pts) < (baseline_pts_team - 5))

        return delta_pts_off_mean, delta_pts_off_sd, delta_pts_off_p10, delta_pts_off_p90, risk_tail

    # Run for home team
    home_no_closer_delta_pts_off_mean, home_no_closer_delta_pts_off_sd, home_no_closer_delta_pts_off_p10, home_no_closer_delta_pts_off_p90, home_no_closer_risk_tail = \
        run_no_closer_mc(home_team_id, game['gameId'], current_home_top5, True)

    # Run for away team
    away_no_closer_delta_pts_off_mean, away_no_closer_delta_pts_off_sd, away_no_closer_delta_pts_off_p10, away_no_closer_delta_pts_off_p90, away_no_closer_risk_tail = \
        run_no_closer_mc(away_team_id, game['gameId'], current_away_top5, False)

    # Calculate delta margin features
    no_closer_delta_margin_mean = home_no_closer_delta_pts_off_mean - away_no_closer_delta_pts_off_mean
    no_closer_delta_margin_sd = np.sqrt(home_no_closer_delta_pts_off_sd**2 + away_no_closer_delta_pts_off_sd**2) # Simple sum of variances
    # For p10/p90 of margin, a full simulation of margin is needed, for now use mean deltas
    no_closer_delta_margin_p10 = home_no_closer_delta_pts_off_p10 - away_no_closer_delta_pts_off_p90 # Approximation
    no_closer_delta_margin_p90 = home_no_closer_delta_pts_off_p90 - away_no_closer_delta_pts_off_p10 # Approximation

    # --- Pace-Stress Test MC ---
    def run_pace_stress_mc(team_prefix, opponent_prefix, current_game_id):
        # Get baseline team stats
        baseline_pace = game[f'{team_prefix}_rolling_pace_l10']
        baseline_ORtg = game[f'{team_prefix}_rolling_ORtg_l10']
        baseline_DRtg = game[f'{team_prefix}_rolling_DRtg_l10']
        baseline_3PA = game[f'{team_prefix}_threePointersAttempted_agg'] # Current game 3PA
        baseline_3PM = game[f'{team_prefix}_threePointersMade_agg'] # Current game 3PM
        baseline_total_points = game[f'{team_prefix}Score']

        # Get opponent stats
        opp_p90_pace = game[f'{opponent_prefix}_rolling_pace_p90_l10']

        # Get Beta parameters for shooting perturbation
        alpha_3P = game[f'{team_prefix}_alpha_3P']
        beta_3P = game[f'{team_prefix}_beta_3P']
        team_3p_mean = game[f'{team_prefix}_season_3P_pct']

        # Handle NaNs
        if pd.isna(baseline_pace) or pd.isna(baseline_ORtg) or pd.isna(baseline_DRtg) or pd.isna(opp_p90_pace):
            return np.nan, np.nan, np.nan, np.nan

        # Target pace: max of current team pace and opponent's p90 pace
        target_pace = max(baseline_pace, opp_p90_pace)

        mc_totals = []
        for d in range(D):
            # Simulate possessions (simple Normal around target pace)
            sim_pace = np.random.normal(target_pace, target_pace * 0.05) # 5% std dev for pace
            sim_pace = max(50, min(120, sim_pace)) # Clip to realistic range
            sim_poss = (sim_pace / 48) * 240 # Convert pace to possessions for a game

            # Expected points under stress
            # Using average of team's ORtg and opponent's DRtg for expected points per 100 possessions
            expected_pts_per_100_poss = (baseline_ORtg + baseline_DRtg) / 2
            pts_d = (sim_poss / 100.0) * expected_pts_per_100_poss

            # Shooting perturbation (if Beta params are valid)
            if not pd.isna(alpha_3P) and not pd.isna(beta_3P) and alpha_3P > 0 and beta_3P > 0:
                th3 = np.random.beta(alpha_3P, beta_3P)
                # Approximate expected 3PA under higher pace (scale with possessions)
                expected_3PA = baseline_3PA * (sim_poss / game[f'{team_prefix}_possessions']) if game[f'{team_prefix}_possessions'] > 0 else baseline_3PA
                pts_d += 3 * (th3 - team_3p_mean) * expected_3PA

            mc_totals.append(pts_d)

        # Summarize deltas vs baseline total points
        pace_stress_delta_total_mean = np.mean(mc_totals) - baseline_total_points
        pace_stress_delta_total_sd = np.std(mc_totals)
        pace_stress_high_total_risk = np.mean(np.array(mc_totals) > (baseline_total_points + 10)) # Example threshold

        return pace_stress_delta_total_mean, pace_stress_delta_total_sd, pace_stress_high_total_risk

    # Run for home team
    home_pace_stress_delta_total_mean, home_pace_stress_delta_total_sd, home_pace_stress_high_total_risk = \
        run_pace_stress_mc('home', 'away', game['gameId'])

    # Run for away team
    away_pace_stress_delta_total_mean, away_pace_stress_delta_total_sd, away_pace_stress_high_total_risk = \
        run_pace_stress_mc('away', 'home', game['gameId'])

    # For pace-stress delta margin, we need to simulate both teams simultaneously
    # This is more complex, for now, we'll just return the total deltas
    # Placeholder for now, or can be derived from home/away total deltas
    pace_stress_delta_margin_mean = home_pace_stress_delta_total_mean - away_pace_stress_delta_total_mean

    return i, home_entropy, away_entropy, home_jaccard_overlap_avg, away_jaccard_overlap_avg, home_pos_counts['G'], home_pos_counts['F'], home_pos_counts['C'], home_pos_variance, home_switchability_index, away_pos_counts['G'], away_pos_counts['F'], away_pos_counts['C'], away_pos_variance, away_switchability_index, \
           home_no_closer_delta_pts_off_mean, home_no_closer_delta_pts_off_sd, home_no_closer_delta_pts_off_p10, home_no_closer_delta_pts_off_p90, home_no_closer_risk_tail, \
           away_no_closer_delta_pts_off_mean, away_no_closer_delta_pts_off_sd, away_no_closer_delta_pts_off_p10, away_no_closer_delta_pts_off_p90, away_no_closer_risk_tail, \
           no_closer_delta_margin_mean, no_closer_delta_margin_sd, no_closer_delta_margin_p10, no_closer_delta_margin_p90, \
           home_pace_stress_delta_total_mean, home_pace_stress_delta_total_sd, home_pace_stress_high_total_risk, \
           away_pace_stress_delta_total_mean, away_pace_stress_delta_total_sd, away_pace_stress_high_total_risk, \
           pace_stress_delta_margin_mean

if __name__ == '__main__':
    # Load configuration
    with open('/Users/jackweekly/Desktop/NBA/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Define paths from config
    current_dir = os.path.dirname(__file__)
    processed_data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed'))

    # Load processed data from the previous step
    print("Loading processed data for lineup consistency features...")
    df_games = pd.read_parquet(os.path.join(processed_data_path, "games_with_team_stats.parquet"))
    df_player_stats = pd.read_parquet(os.path.join(processed_data_path, "player_stats_processed.parquet"))
    # Filter out negative numMinutes values
    df_player_stats = df_player_stats[df_player_stats['numMinutes'] >= 0]

    # Load player metadata for positions and heights
    df_players_metadata = pd.read_csv(os.path.join(current_dir, '..', 'data', 'metadata', 'lookup', 'players.csv'))
    df_players_metadata = df_players_metadata[['personId', 'position', 'height']]

    # Merge player metadata with player stats
    df_player_stats = pd.merge(df_player_stats, df_players_metadata, on='personId', how='left')

    print("Processed data loaded.")

    # Revert lambda to original form
    df_top5_minutes = df_player_stats.set_index(['gameId', 'playerteamId']).groupby(level=[0, 1])['numMinutes'].nlargest(5).reset_index()
    df_top5_minutes = pd.merge(df_top5_minutes, df_player_stats[['gameId', 'playerteamId', 'personId', 'numMinutes']], on=['gameId', 'playerteamId', 'numMinutes'], how='left')
    df_top5_minutes = df_top5_minutes.groupby(['gameId', 'playerteamId'])['personId'].apply(list).reset_index(name='top5_players')



    df_player_stats = df_player_stats[df_player_stats['numMinutes'] >= 0]
    print("Processed data loaded.")

    # Ensure gameId and playerteamId are of consistent type for merging
    df_top5_minutes['gameId'] = df_top5_minutes['gameId'].astype(str)
    df_top5_minutes['playerteamId'] = df_top5_minutes['playerteamId'].astype(str)
    df_games['gameId'] = df_games['gameId'].astype(str)
    df_games['hometeamId'] = df_games['hometeamId'].astype(str)
    df_games['awayteamId'] = df_games['awayteamId'].astype(str)

    print("Top 5 minutes players identified.")

    # --- Pre-calculate Player-level Rolling Stats for Counterfactuals ---
    print("Pre-calculating player-level rolling stats...")

    # Calculate minutes share and points per minute for each player in each game
    df_player_stats['minutes_share'] = df_player_stats.groupby(['gameId', 'playerteamId'])['numMinutes'].transform(lambda x: x / x.sum())
    df_player_stats['pts_per_min'] = df_player_stats['points'] / df_player_stats['numMinutes']
    df_player_stats['pts_per_min'] = df_player_stats['pts_per_min'].replace([np.inf, -np.inf], 0).fillna(0)

    # Sort by player and date for rolling calculations
    df_player_stats = df_player_stats.sort_values(by=['personId', 'gameDate']).reset_index(drop=True)

    # EWMA last 10 games for minutes share and points per minute
    df_player_stats['minutes_share_l10'] = df_player_stats.groupby('personId')['minutes_share'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))
    df_player_stats['pts_per_min_l10'] = df_player_stats.groupby('personId')['pts_per_min'].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

    # --- Pre-calculate Team-level Rolling Stats (Pace, ORtg, DRtg) for Counterfactuals ---
    print("Pre-calculating team-level rolling stats...")

    # Helper function to estimate possessions (re-implement from 03_feature_engineering.py)
    def estimate_possessions(df, team_prefix):
        fga = df[f'fieldGoalsAttempted_{team_prefix}_agg']
        fta = df[f'freeThrowsAttempted_{team_prefix}_agg']
        oreb = df[f'reboundsOffensive_{team_prefix}_agg']
        tov = df[f'turnovers_{team_prefix}_agg']
        possessions = fga + 0.44 * fta - oreb + tov
        return possessions

    # Calculate possessions for home and away teams
    df_games['home_possessions'] = estimate_possessions(df_games, 'home')
    df_games['away_possessions'] = estimate_possessions(df_games, 'away')

    # Calculate pace (possessions per 48 minutes). Assuming 240 team minutes per game.
    df_games['home_pace'] = (df_games['home_possessions'] / 240) * 48
    df_games['away_pace'] = (df_games['away_possessions'] / 240) * 48

    # Calculate ORtg and DRtg (points per 100 possessions)
    df_games['home_ORtg'] = (df_games['homeScore'] / df_games['home_possessions']) * 100
    df_games['away_ORtg'] = (df_games['awayScore'] / df_games['away_possessions']) * 100
    df_games['home_DRtg'] = (df_games['awayScore'] / df_games['home_possessions']) * 100 # Opponent points / team possessions
    df_games['away_DRtg'] = (df_games['homeScore'] / df_games['away_possessions']) * 100 # Opponent points / team possessions

    # Fill any inf/NaN values
    for col in ['home_pace', 'away_pace', 'home_ORtg', 'away_ORtg', 'home_DRtg', 'away_DRtg']:
        df_games[col] = df_games[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Prepare data for rolling team stats
    team_game_stats = []
    for team_id in pd.concat([df_games['hometeamId'], df_games['awayteamId']]).unique():
        team_games_stats = df_games[
            (df_games['hometeamId'] == team_id) | (df_games['awayteamId'] == team_id)
        ].copy()
        team_games_stats['teamId'] = team_id
        team_games_stats['pace'] = np.where(
            team_games_stats['hometeamId'] == team_id, team_games_stats['home_pace'], team_games_stats['away_pace']
        )
        team_games_stats['ORtg'] = np.where(
            team_games_stats['hometeamId'] == team_id, team_games_stats['home_ORtg'], team_games_stats['away_ORtg']
        )
        team_games_stats['DRtg'] = np.where(
            team_games_stats['hometeamId'] == team_id, team_games_stats['home_DRtg'], team_games_stats['away_DRtg']
        )
        team_game_stats.append(team_games_stats[['gameId', 'gameDate', 'teamId', 'pace', 'ORtg', 'DRtg']])

    df_team_rolling_stats = pd.concat(team_game_stats).sort_values(by=['teamId', 'gameDate']).reset_index(drop=True)

    # Calculate rolling averages for pace, ORtg, DRtg
    df_team_rolling_stats['rolling_pace_l10'] = df_team_rolling_stats.groupby('teamId')['pace']\
        .transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))
    df_team_rolling_stats['rolling_ORtg_l10'] = df_team_rolling_stats.groupby('teamId')['ORtg']\
        .transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))
    df_team_rolling_stats['rolling_DRtg_l10'] = df_team_rolling_stats.groupby('teamId')['DRtg']\
        .transform(lambda x: x.ewm(span=10, adjust=False, min_periods=1).mean().shift(1))

    # Calculate rolling p90 pace for pace-stress test
    df_team_rolling_stats['rolling_pace_p90_l10'] = df_team_rolling_stats.groupby('teamId')['pace']\
        .transform(lambda x: x.rolling(window=10, min_periods=1).quantile(0.9).shift(1))

    # Merge rolling team stats back into df_games
    df_games = pd.merge(
        df_games,
        df_team_rolling_stats[['gameId', 'teamId', 'rolling_pace_l10', 'rolling_ORtg_l10', 'rolling_DRtg_l10', 'rolling_pace_p90_l10']].rename(columns={'teamId': 'hometeamId'}),
        on=['gameId', 'hometeamId'],
        how='left',
        suffixes=('', '_home_rolling')
    )
    df_games.rename(columns={
        'rolling_pace_l10': 'home_rolling_pace_l10',
        'rolling_ORtg_l10': 'home_rolling_ORtg_l10',
        'rolling_DRtg_l10': 'home_rolling_DRtg_l10',
        'rolling_pace_p90_l10': 'home_rolling_pace_p90_l10'
    }, inplace=True)

    df_games = pd.merge(
        df_games,
        df_team_rolling_stats[['gameId', 'teamId', 'rolling_pace_l10', 'rolling_ORtg_l10', 'rolling_DRtg_l10', 'rolling_pace_p90_l10']].rename(columns={'teamId': 'awayteamId'}),
        on=['gameId', 'awayteamId'],
        how='left',
        suffixes=('', '_away_rolling')
    )
    df_games.rename(columns={
        'rolling_pace_l10': 'away_rolling_pace_l10',
        'rolling_ORtg_l10': 'away_rolling_ORtg_l10',
        'rolling_DRtg_l10': 'away_rolling_DRtg_l10',
        'rolling_pace_p90_l10': 'away_rolling_pace_p90_l10'
    }, inplace=True)

    print("Player and team rolling stats pre-calculated.")

    # --- Task 2 & 3: Compute closing lineup frequency distribution and calculate entropy ---
    print("Calculating rolling closing lineup entropy for each team...")

    # Merge home team top5 players
    df_top5_home_renamed = df_top5_minutes.copy()
    df_top5_home_renamed.rename(columns={'playerteamId': 'hometeamId', 'top5_players': 'home_top5_players'}, inplace=True)
    df_games = pd.merge(
        df_games,
        df_top5_home_renamed,
        on=['gameId', 'hometeamId'],
        how='left'
    )

    # Merge away team top5 players
    df_top5_away_renamed = df_top5_minutes.copy()
    df_top5_away_renamed.rename(columns={'playerteamId': 'awayteamId', 'top5_players': 'away_top5_players'}, inplace=True)
    df_games = pd.merge(
        df_games,
        df_top5_away_renamed,
        on=['gameId', 'awayteamId'],
        how='left'
    )

    # Sort games by date to ensure correct rolling window
    df_games = df_games.sort_values(by='gameDate').reset_index(drop=True)

    # Initialize columns for entropy
    df_games['home_closing_entropy_last10'] = np.nan
    df_games['away_closing_entropy_last10'] = np.nan

    # Prepare data for multiprocessing
    games_for_processing = [(i, row) for i, row in df_games.iterrows()]

    print("Starting parallel calculation of rolling entropy...")

    # Use multiprocessing Pool
    num_processes = multiprocessing.cpu_count() # Use all available CPU cores
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(df_games, df_top5_minutes, df_players_metadata, df_player_stats)) as pool:
        total_games = len(games_for_processing)
        print(f"Processing {total_games} games in parallel...")
        sys.stdout.flush()

        for idx, (i, home_entropy, away_entropy, home_jaccard_overlap_avg, away_jaccard_overlap_avg, home_g_count, home_f_count, home_c_count, home_pos_variance, home_switchability_index, away_g_count, away_f_count, away_c_count, away_pos_variance, away_switchability_index, \
                    home_no_closer_delta_pts_off_mean, home_no_closer_delta_pts_off_sd, home_no_closer_delta_pts_off_p10, home_no_closer_delta_pts_off_p90, home_no_closer_risk_tail, \
                    away_no_closer_delta_pts_off_mean, away_no_closer_delta_pts_off_sd, away_no_closer_delta_pts_off_p10, away_no_closer_delta_pts_off_p90, away_no_closer_risk_tail, \
                    no_closer_delta_margin_mean, no_closer_delta_margin_sd, no_closer_delta_margin_p10, no_closer_delta_margin_p90, \
                    home_pace_stress_delta_total_mean, home_pace_stress_delta_total_sd, home_pace_stress_high_total_risk, \
                    away_pace_stress_delta_total_mean, away_pace_stress_delta_total_sd, away_pace_stress_high_total_risk, \
                    pace_stress_delta_margin_mean) in enumerate(pool.imap_unordered(process_game, games_for_processing)):
            if (idx + 1) % 100 == 0 or idx == 0 or idx == total_games - 1:
                print(f"Collected results for {idx + 1}/{total_games} games...")
                sys.stdout.flush()
            df_games.loc[i, 'home_closing_entropy_last10'] = home_entropy
            df_games.loc[i, 'away_closing_entropy_last10'] = away_entropy
            df_games.loc[i, 'home_jaccard_overlap_last10'] = home_jaccard_overlap_avg
            df_games.loc[i, 'away_jaccard_overlap_last10'] = away_jaccard_overlap_avg
            df_games.loc[i, 'home_pos_G_count'] = home_g_count
            df_games.loc[i, 'home_pos_F_count'] = home_f_count
            df_games.loc[i, 'home_pos_C_count'] = home_c_count
            df_games.loc[i, 'home_pos_variance'] = home_pos_variance
            df_games.loc[i, 'home_switchability_index'] = home_switchability_index
            df_games.loc[i, 'away_pos_G_count'] = away_g_count
            df_games.loc[i, 'away_pos_F_count'] = away_f_count
            df_games.loc[i, 'away_pos_C_count'] = away_c_count
            df_games.loc[i, 'away_pos_variance'] = away_pos_variance
            df_games.loc[i, 'away_switchability_index'] = away_switchability_index

            df_games.loc[i, 'home_no_closer_delta_pts_off_mean'] = home_no_closer_delta_pts_off_mean
            df_games.loc[i, 'home_no_closer_delta_pts_off_sd'] = home_no_closer_delta_pts_off_sd
            df_games.loc[i, 'home_no_closer_delta_pts_off_p10'] = home_no_closer_delta_pts_off_p10
            df_games.loc[i, 'home_no_closer_delta_pts_off_p90'] = home_no_closer_delta_pts_off_p90
            df_games.loc[i, 'home_no_closer_risk_tail'] = home_no_closer_risk_tail
            df_games.loc[i, 'away_no_closer_delta_pts_off_mean'] = away_no_closer_delta_pts_off_mean
            df_games.loc[i, 'away_no_closer_delta_pts_off_sd'] = away_no_closer_delta_pts_off_sd
            df_games.loc[i, 'away_no_closer_delta_pts_off_p10'] = away_no_closer_delta_pts_off_p10
            df_games.loc[i, 'away_no_closer_delta_pts_off_p90'] = away_no_closer_delta_pts_off_p90
            df_games.loc[i, 'away_no_closer_risk_tail'] = away_no_closer_risk_tail
            df_games.loc[i, 'no_closer_delta_margin_mean'] = no_closer_delta_margin_mean
            df_games.loc[i, 'no_closer_delta_margin_sd'] = no_closer_delta_margin_sd
            df_games.loc[i, 'no_closer_delta_margin_p10'] = no_closer_delta_margin_p10
            df_games.loc[i, 'no_closer_delta_margin_p90'] = no_closer_delta_margin_p90

            df_games.loc[i, 'home_pace_stress_delta_total_mean'] = home_pace_stress_delta_total_mean
            df_games.loc[i, 'home_pace_stress_delta_total_sd'] = home_pace_stress_delta_total_sd
            df_games.loc[i, 'home_pace_stress_high_total_risk'] = home_pace_stress_high_total_risk
            df_games.loc[i, 'away_pace_stress_delta_total_mean'] = away_pace_stress_delta_total_mean
            df_games.loc[i, 'away_pace_stress_delta_total_sd'] = away_pace_stress_delta_total_sd
            df_games.loc[i, 'away_pace_stress_high_total_risk'] = away_pace_stress_high_total_risk
            df_games.loc[i, 'pace_stress_delta_margin_mean'] = pace_stress_delta_margin_mean

    print("Rolling closing lineup entropy, Jaccard overlap, positional balance, switchability, No-Closer counterfactuals, and Pace-Stress counterfactuals calculated.")

    # --- Task 5: Cross with outcome: entropy x clutch_delta ---
    print("Calculating entropy x clutch_delta interaction feature...")

    # Define clutch_delta - using difference in clutch points per 100 possessions as a proxy
    # Assuming 'home_clutch_pp100' and 'away_clutch_pp100' are already in df_games from previous script
    df_games['clutch_pp100_delta'] = df_games['clutch_pp100'] - df_games['clutch_pp100_away_clutch_eff']

    # Interaction term
    df_games['home_entropy_x_clutch_delta'] = df_games['home_closing_entropy_last10'] * df_games['clutch_pp100']
    df_games['away_entropy_x_clutch_delta'] = df_games['away_closing_entropy_last10'] * (-df_games['clutch_pp100_away_clutch_eff']) # Away team's perspective

    print("Entropy x clutch_delta interaction feature calculated.")

    # --- Feature Engineering: Bench Finisher Impact ---
    print("Calculating bench finisher impact...")

    # Define bench players as those not in the top 5 minutes for a game
    # First, get all players who played in each game
    players_in_game = df_player_stats[['gameId', 'personId', 'playerteamId', 'points', 'numMinutes']].copy()

    # Create a flattened DataFrame of top 5 players for easier merging
    df_top5_players_flat = df_top5_minutes.explode('top5_players')
    df_top5_players_flat.rename(columns={'top5_players': 'personId'}, inplace=True)
    df_top5_players_flat = df_top5_players_flat[['gameId', 'playerteamId', 'personId']].drop_duplicates()

    # Merge to identify players who are NOT in the top 5 (i.e., bench players)
    df_merged_players = pd.merge(
        players_in_game,
        df_top5_players_flat,
        on=['gameId', 'playerteamId', 'personId'],
        how='left',
        indicator=True
    )

    # Bench players are those not in top5_players
    df_bench_players = df_merged_players[df_merged_players['_merge'] == 'left_only'].copy()

    # Calculate points per minute for bench players
    df_bench_players['points_per_minute'] = df_bench_players['points'] / df_bench_players['numMinutes']
    df_bench_players['points_per_minute'] = df_bench_players['points_per_minute'].replace([np.inf, -np.inf], 0).fillna(0)

    # Aggregate bench player points per minute for each team in each game
    bench_impact = df_bench_players.groupby(['gameId', 'playerteamId'])['points_per_minute'].sum().reset_index()
    bench_impact.rename(columns={'points_per_minute': 'bench_finisher_impact'}, inplace=True)

    # Merge bench impact into df_games
    home_bench_impact_df = bench_impact.rename(columns={
        'playerteamId': 'hometeamId',
        'bench_finisher_impact': 'home_bench_finisher_impact'
    })
    df_games = pd.merge(
        df_games,
        home_bench_impact_df[['gameId', 'hometeamId', 'home_bench_finisher_impact']],
        on=['gameId', 'hometeamId'],
        how='left'
    )

    away_bench_impact_df = bench_impact.rename(columns={
        'playerteamId': 'awayteamId',
        'bench_finisher_impact': 'away_bench_finisher_impact'
    })
    df_games = pd.merge(
        df_games,
        away_bench_impact_df[['gameId', 'awayteamId', 'away_bench_finisher_impact']],
        on=['gameId', 'awayteamId'],
        how='left'
    )

    df_games['home_bench_finisher_impact'] = df_games['home_bench_finisher_impact'].fillna(0)
    df_games['away_bench_finisher_impact'] = df_games['away_bench_finisher_impact'].fillna(0)

    print("Bench finisher impact calculated.")

    # --- Feature Engineering: Rolling Margin Volatility ---
    print("Calculating rolling margin volatility features...")

    ROLLING_WINDOW_SIZE = 20

    # Calculate game margin for each team
    df_games['home_margin'] = df_games['homeScore'] - df_games['awayScore']
    df_games['away_margin'] = df_games['awayScore'] - df_games['homeScore']

    # Prepare data for rolling calculations
    team_margins = []
    for team_id in pd.concat([df_games['hometeamId'], df_games['awayteamId']]).unique():
        team_games = df_games[
            (df_games['hometeamId'] == team_id) | (df_games['awayteamId'] == team_id)
        ].copy()
        team_games['teamId'] = team_id
        team_games['margin'] = np.where(
            team_games['hometeamId'] == team_id, team_games['home_margin'], team_games['away_margin']
        )
        team_games['is_win'] = np.where(team_games['winner'] == team_id, 1, 0)
        team_margins.append(team_games[['gameId', 'gameDate', 'teamId', 'margin', 'is_win']])

    df_team_margins = pd.concat(team_margins).sort_values(by=['teamId', 'gameDate']).reset_index(drop=True)

    # Calculate rolling skew, kurtosis, p10, p90
    df_team_margins['rolling_margin_skew'] = df_team_margins.groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).skew())
    df_team_margins['rolling_margin_kurtosis'] = df_team_margins.groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).kurt())
    df_team_margins['rolling_margin_p10'] = df_team_margins.groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).quantile(0.1))
    df_team_margins['rolling_margin_p90'] = df_team_margins.groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).quantile(0.9))

    df_team_margins['rolling_quantile_spread'] = df_team_margins['rolling_margin_p90'] - df_team_margins['rolling_margin_p10']

    # Calculate rolling skew and kurtosis for wins only
    df_team_margins['rolling_win_margin_skew'] = df_team_margins[df_team_margins['is_win'] == 1].groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).skew())
    df_team_margins['rolling_win_margin_kurtosis'] = df_team_margins[df_team_margins['is_win'] == 1].groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).kurt())

    # Calculate rolling skew and kurtosis for losses only
    df_team_margins['rolling_loss_margin_skew'] = df_team_margins[df_team_margins['is_win'] == 0].groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).skew())
    df_team_margins['rolling_loss_margin_kurtosis'] = df_team_margins[df_team_margins['is_win'] == 0].groupby('teamId')['margin']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).kurt())

    # Merge rolling features back into df_games
    df_games = pd.merge(
        df_games,
        df_team_margins[['gameId', 'teamId', 'rolling_margin_skew', 'rolling_margin_kurtosis', 'rolling_quantile_spread', 'rolling_win_margin_skew', 'rolling_win_margin_kurtosis', 'rolling_loss_margin_skew', 'rolling_loss_margin_kurtosis']].rename(columns={'teamId': 'hometeamId'}),
        on=['gameId', 'hometeamId'],
        how='left',
        suffixes=('', '_home')
    )
    df_games.rename(columns={
        'rolling_margin_skew_home': 'home_rolling_margin_skew',
        'rolling_margin_kurtosis_home': 'home_rolling_margin_kurtosis',
        'rolling_quantile_spread_home': 'home_rolling_quantile_spread',
        'rolling_win_margin_skew_home': 'home_rolling_win_margin_skew',
        'rolling_win_margin_kurtosis_home': 'home_rolling_win_margin_kurtosis',
        'rolling_loss_margin_skew_home': 'home_rolling_loss_margin_skew',
        'rolling_loss_margin_kurtosis_home': 'home_rolling_loss_margin_kurtosis'
    }, inplace=True)

    df_games = pd.merge(
        df_games,
        df_team_margins[['gameId', 'teamId', 'rolling_margin_skew', 'rolling_margin_kurtosis', 'rolling_quantile_spread', 'rolling_win_margin_skew', 'rolling_win_margin_kurtosis', 'rolling_loss_margin_skew', 'rolling_loss_margin_kurtosis']].rename(columns={'teamId': 'awayteamId'}),
        on=['gameId', 'awayteamId'],
        how='left',
        suffixes=('', '_away')
    )
    df_games.rename(columns={
        'rolling_margin_skew_away': 'away_rolling_margin_skew',
        'rolling_margin_kurtosis_away': 'away_rolling_margin_kurtosis',
        'rolling_quantile_spread_away': 'away_rolling_quantile_spread',
        'rolling_win_margin_skew_away': 'away_rolling_win_margin_skew',
        'rolling_win_margin_kurtosis_away': 'away_rolling_win_margin_kurtosis',
        'rolling_loss_margin_skew_away': 'away_rolling_loss_margin_skew',
        'rolling_loss_margin_kurtosis_away': 'away_rolling_loss_margin_kurtosis'
    }, inplace=True)

    print("Rolling margin volatility features calculated.")

    # --- Placeholder for Lead Volatility (requires more detailed data) ---
    print("Placeholder for Lead Volatility: Requires play-by-play data for biggestLead, timesTied, leadChanges.")
    df_games['home_biggest_lead_normalized'] = np.nan # Placeholder
    df_games['away_biggest_lead_normalized'] = np.nan # Placeholder
    df_games['home_times_tied_normalized'] = np.nan # Placeholder
    df_games['away_times_tied_normalized'] = np.nan # Placeholder
    df_games['home_lead_changes_normalized'] = np.nan # Placeholder
    df_games['away_lead_changes_normalized'] = np.nan # Placeholder

    # --- Feature Engineering: Chaos Index ---
    print("Calculating Chaos Index...")

    # For now, using available volatility measures. Assign equal weights.
    # Fill NaN with 0 for calculation, as missing values would indicate no volatility data
    df_games['home_chaos_index'] = df_games['home_rolling_margin_skew'].fillna(0).abs() + \
                                   df_games['home_rolling_margin_kurtosis'].fillna(0).abs() + \
                                   df_games['home_rolling_quantile_spread'].fillna(0)

    df_games['away_chaos_index'] = df_games['away_rolling_margin_skew'].fillna(0).abs() + \
                                   df_games['away_rolling_margin_kurtosis'].fillna(0).abs() + \
                                   df_games['away_rolling_quantile_spread'].fillna(0)

    # Engineer differential features
    df_games['chaos_index_differential'] = df_games['home_chaos_index'] - df_games['away_chaos_index']

    print("Chaos Index calculated.")

    # --- Placeholder for Lead Elasticity (requires biggestLead data) ---
    print("Placeholder for Lead Elasticity: Requires biggestLead data.")
    df_games['home_lead_elasticity'] = np.nan # Placeholder
    df_games['away_lead_elasticity'] = np.nan # Placeholder

    # --- Placeholder for Comeback Propensity Proxy (requires half/Q3 scores) ---
    print("Placeholder for Comeback Propensity Proxy: Requires halftime or Q3 scores.")
    df_games['home_comeback_propensity'] = np.nan # Placeholder
    df_games['away_comeback_propensity'] = np.nan # Placeholder

    # --- Placeholder for Clutch Specificity (requires quarter-specific player stats) ---
    print("Placeholder for Clutch Specificity: Requires quarter-specific player statistics.")
    df_games['home_q4_usage_skew'] = np.nan # Placeholder
    df_games['away_q4_usage_skew'] = np.nan # Placeholder
    df_games['home_q4_pace_shift'] = np.nan # Placeholder
    df_games['away_q4_pace_shift'] = np.nan # Placeholder
    df_games['home_clutch_foul_sensitivity'] = np.nan # Placeholder
    df_games['away_clutch_foul_sensitivity'] = np.nan # Placeholder

    # --- Feature Engineering: 3P Reliance ---
    print("Calculating 3P reliance...")

    # Assuming 'threePointersMade_home_agg' and 'points_home_agg' are available from 03_feature_engineering.py
    df_games['home_3P_reliance'] = (df_games['threePointersMade_home_agg'] * 3) / df_games['points_home_agg']
    df_games['away_3P_reliance'] = (df_games['threePointersMade_away_agg'] * 3) / df_games['points_away_agg']

    df_games['home_3P_reliance'] = df_games['home_3P_reliance'].replace([np.inf, -np.inf], 0).fillna(0)
    df_games['away_3P_reliance'] = df_games['away_3P_reliance'].replace([np.inf, -np.inf], 0).fillna(0)

    # Winsorize and standardize 3P reliance
    df_games['home_3P_reliance'] = winsorize_series(df_games['home_3P_reliance'])
    df_games['away_3P_reliance'] = winsorize_series(df_games['away_3P_reliance'])
    df_games = standardize_series_by_season(df_games, 'home_3P_reliance')
    df_games = standardize_series_by_season(df_games, 'away_3P_reliance')

    print("3P reliance calculated.")

    # Helper functions for winsorization and standardization
    def winsorize_series(series, lower_bound=0.01, upper_bound=0.99):
        lower_val = series.quantile(lower_bound)
        upper_val = series.quantile(upper_bound)
        return series.clip(lower=lower_val, upper=upper_val)

    def standardize_series_by_season(df, column):
        df[f'{column}_z'] = df.groupby('season')[column].transform(lambda x: (x - x.mean()) / x.std())
        return df

    # --- Feature Engineering: Style-Matchup Grid ---
    print("Calculating style-matchup grid features...")

    # Calculate OREB% (opportunity-adjusted)
    # Home OREB%
    df_games['home_OREB_pct'] = df_games['reboundsOffensive_home_agg'] / (df_games['reboundsOffensive_home_agg'] + (df_games['reboundsTotal_away_agg'] - df_games['reboundsOffensive_away_agg']))
    df_games['home_OREB_pct'] = df_games['home_OREB_pct'].replace([np.inf, -np.inf], 0).fillna(0)

    # Away OREB%
    df_games['away_OREB_pct'] = df_games['reboundsOffensive_away_agg'] / (df_games['reboundsOffensive_away_agg'] + (df_games['reboundsTotal_home_agg'] - df_games['reboundsOffensive_home_agg']))
    df_games['away_OREB_pct'] = df_games['away_OREB_pct'].replace([np.inf, -np.inf], 0).fillna(0)

    # Winsorize and standardize OREB%
    df_games['home_OREB_pct'] = winsorize_series(df_games['home_OREB_pct'])
    df_games['away_OREB_pct'] = winsorize_series(df_games['away_OREB_pct'])
    df_games = standardize_series_by_season(df_games, 'home_OREB_pct')
    df_games = standardize_series_by_season(df_games, 'away_OREB_pct')

    # Home DREB% weakness (1 - DREB%)
    df_games['home_DREB_pct_weakness'] = 1 - ( (df_games['reboundsTotal_home_agg'] - df_games['reboundsOffensive_home_agg']) / ((df_games['reboundsTotal_home_agg'] - df_games['reboundsOffensive_home_agg']) + df_games['reboundsOffensive_away_agg']) )
    df_games['home_DREB_pct_weakness'] = df_games['home_DREB_pct_weakness'].replace([np.inf, -np.inf], 0).fillna(0)

    # Away DREB% weakness (1 - DREB%)
    df_games['away_DREB_pct_weakness'] = 1 - ( (df_games['reboundsTotal_away_agg'] - df_games['reboundsOffensive_away_agg']) / ((df_games['reboundsTotal_away_agg'] - df_games['reboundsOffensive_away_agg']) + df_games['reboundsOffensive_home_agg']) )
    df_games['away_DREB_pct_weakness'] = df_games['away_DREB_pct_weakness'].replace([np.inf, -np.inf], 0).fillna(0)

    # Winsorize and standardize DREB% weakness
    df_games['home_DREB_pct_weakness'] = winsorize_series(df_games['home_DREB_pct_weakness'])
    df_games['away_DREB_pct_weakness'] = winsorize_series(df_games['away_DREB_pct_weakness'])
    df_games = standardize_series_by_season(df_games, 'home_DREB_pct_weakness')
    df_games = standardize_series_by_season(df_games, 'away_DREB_pct_weakness')

    # Interaction: (home 3P reliance - away 3P defense quality)
    df_games['home_3P_reliance_minus_away_3P_def_quality'] = df_games['home_3P_reliance_z'] - df_games['away_opp_3P_def_volatility'].fillna(0) # Using volatility as proxy for quality

    # Interaction: (home 3P reliance x away 3P-def variance)
    df_games['home_3P_reliance_x_away_3P_def_var'] = df_games['home_3P_reliance_z'] * df_games['away_opp_3P_def_volatility'].fillna(0)

    # Interaction: (away 3P reliance - home 3P defense quality)
    df_games['away_3P_reliance_minus_home_3P_def_quality'] = df_games['away_3P_reliance_z'] - df_games['home_opp_3P_def_volatility'].fillna(0)

    # Interaction: (away 3P reliance x home 3P-def variance)
    df_games['away_3P_reliance_x_home_3P_def_var'] = df_games['away_3P_reliance_z'] * df_games['home_opp_3P_def_volatility'].fillna(0)

    # Interaction: (home OREB% x away DREB% weakness)
    df_games['home_OREB_x_away_DREB_weakness'] = df_games['home_OREB_pct_z'] * df_games['away_DREB_pct_weakness_z']

    # Interaction: (away OREB% x home DREB% weakness)
    df_games['away_OREB_x_home_DREB_weakness'] = df_games['away_OREB_pct_z'] * df_games['home_DREB_pct_weakness_z']

    # --- Placeholder for (home pace tendency x away fatigue flags) ---
    print("Placeholder for (home pace tendency x away fatigue flags): Requires pace tendency and fatigue flags data.")
    df_games['home_pace_x_away_fatigue'] = np.nan # Placeholder
    df_games['away_pace_x_home_fatigue'] = np.nan # Placeholder

    print("Style-matchup grid features calculated.")

    # --- Feature Engineering: Four-Factors Grid ---
    print("Calculating Four-Factors grid features...")

    # Calculate eFG%
    df_games['home_eFG_pct'] = (df_games['fieldGoalsMade_home_agg'] + 0.5 * df_games['threePointersMade_home_agg']) / df_games['fieldGoalsAttempted_home_agg']
    df_games['away_eFG_pct'] = (df_games['fieldGoalsMade_away_agg'] + 0.5 * df_games['threePointersMade_away_agg']) / df_games['fieldGoalsAttempted_away_agg']
    df_games['home_eFG_pct'] = df_games['home_eFG_pct'].replace([np.inf, -np.inf], 0).fillna(0)
    df_games['away_eFG_pct'] = df_games['away_eFG_pct'].replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate TOV% (using possessions as denominator)
    df_games['home_TOV_pct'] = df_games['turnovers_home_agg'] / (df_games['fieldGoalsAttempted_home_agg'] + 0.44 * df_games['freeThrowsAttempted_home_agg'] + df_games['turnovers_home_agg'])
    df_games['away_TOV_pct'] = df_games['turnovers_away_agg'] / (df_games['fieldGoalsAttempted_away_agg'] + 0.44 * df_games['freeThrowsAttempted_away_agg'] + df_games['turnovers_away_agg'])
    df_games['home_TOV_pct'] = df_games['home_TOV_pct'].replace([np.inf, -np.inf], 0).fillna(0)
    df_games['away_TOV_pct'] = df_games['away_TOV_pct'].replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate FTr (Free Throw Rate)
    df_games['home_FTr'] = df_games['freeThrowsAttempted_home_agg'] / df_games['fieldGoalsAttempted_home_agg']
    df_games['away_FTr'] = df_games['freeThrowsAttempted_away_agg'] / df_games['fieldGoalsAttempted_away_agg']
    df_games['home_FTr'] = df_games['home_FTr'].replace([np.inf, -np.inf], 0).fillna(0)
    df_games['away_FTr'] = df_games['away_FTr'].replace([np.inf, -np.inf], 0).fillna(0)

    # Winsorize and standardize Four Factors
    for col in ['home_eFG_pct', 'away_eFG_pct', 'home_TOV_pct', 'away_TOV_pct', 'home_FTr', 'away_FTr']:
        df_games[col] = winsorize_series(df_games[col])
        df_games = standardize_series_by_season(df_games, col)

    # Interactions: eFG%_off vs eFG%_def_opp
    df_games['home_eFG_pct_off_minus_away_eFG_pct_def'] = df_games['home_eFG_pct_z'] - df_games['away_eFG_pct_z']
    df_games['home_eFG_pct_off_x_away_eFG_pct_def'] = df_games['home_eFG_pct_z'] * df_games['away_eFG_pct_z']

    df_games['away_eFG_pct_off_minus_home_eFG_pct_def'] = df_games['away_eFG_pct_z'] - df_games['home_eFG_pct_z']
    df_games['away_eFG_pct_off_x_home_eFG_pct_def'] = df_games['away_eFG_pct_z'] * df_games['home_eFG_pct_z']

    # Interactions: TOV%_off vs STL%_opp (using opp TOV% as proxy for STL% allowed)
    df_games['home_TOV_pct_off_minus_away_TOV_pct_def'] = df_games['home_TOV_pct_z'] - df_games['away_TOV_pct_z']
    df_games['home_TOV_pct_off_x_away_TOV_pct_def'] = df_games['home_TOV_pct_z'] * df_games['away_TOV_pct_z']

    df_games['away_TOV_pct_off_minus_home_TOV_pct_def'] = df_games['away_TOV_pct_z'] - df_games['home_TOV_pct_z']
    df_games['away_TOV_pct_off_x_home_TOV_pct_def'] = df_games['away_TOV_pct_z'] * df_games['home_TOV_pct_z']

    # Interactions: OREB%_off vs DREB%_opp
    df_games['home_OREB_pct_off_minus_away_DREB_pct_def'] = df_games['home_OREB_pct_z'] - df_games['away_DREB_pct_weakness_z'] # DREB% weakness is 1-DREB%
    df_games['home_OREB_pct_off_x_away_DREB_pct_def'] = df_games['home_OREB_pct_z'] * df_games['away_DREB_pct_weakness_z']

    df_games['away_OREB_pct_off_minus_home_DREB_pct_def'] = df_games['away_OREB_pct_z'] - df_games['home_DREB_pct_weakness_z']
    df_games['away_OREB_pct_off_x_home_DREB_pct_def'] = df_games['away_OREB_pct_z'] * df_games['home_DREB_pct_weakness_z']

    # Interactions: FTr_off vs PF rate_opp (using opp FTr as proxy for PF rate allowed)
    df_games['home_FTr_off_minus_away_FTr_def'] = df_games['home_FTr_z'] - df_games['away_FTr_z']
    df_games['home_FTr_off_x_away_FTr_def'] = df_games['home_FTr_z'] * df_games['away_FTr_z']

    df_games['away_FTr_off_minus_home_FTr_def'] = df_games['away_FTr_z'] - df_games['home_FTr_z']
    df_games['away_FTr_off_x_home_FTr_def'] = df_games['away_FTr_z'] * df_games['home_FTr_z']

    print("Four-Factors grid features calculated.")

    # --- Feature Engineering: Lineup Churn Score ---
    print("Calculating lineup churn score...")

    CHURN_SHORT_WINDOW = 5
    CHURN_LONG_WINDOW = 20

    # Prepare data for rolling churn calculations
    team_top5_lineups = []
    for team_id in pd.concat([df_games['hometeamId'], df_games['awayteamId']]).unique():
        team_games_top5 = df_games[
            (df_games['hometeamId'] == team_id) | (df_games['awayteamId'] == team_id)
        ].copy()
        team_games_top5['teamId'] = team_id
        team_games_top5['top5_players_current'] = np.where(
            team_games_top5['hometeamId'] == team_id, team_games_top5['home_top5_players'], team_games_top5['away_top5_players']
        )
        team_top5_lineups.append(team_games_top5[['gameId', 'gameDate', 'teamId', 'top5_players_current']])

    df_team_top5_lineups = pd.concat(team_top5_lineups).sort_values(by=['teamId', 'gameDate']).reset_index(drop=True)

    # Function to count distinct lineups in a rolling window
    def count_distinct_lineups(lineup_series):
        distinct_lineups = []
        for lineup_list in lineup_series:
            if lineup_list is not None:
                distinct_lineups.append(tuple(sorted(lineup_list)))
        return len(set(distinct_lineups))

    # Calculate rolling churn for short and long windows
    df_team_top5_lineups['rolling_churn_short'] = df_team_top5_lineups.groupby('teamId')['top5_players_current']\
        .transform(lambda x: x.rolling(window=CHURN_SHORT_WINDOW, min_periods=1).apply(count_distinct_lineups, raw=False))
    df_team_top5_lineups['rolling_churn_long'] = df_team_top5_lineups.groupby('teamId')['top5_players_current']\
        .transform(lambda x: x.rolling(window=CHURN_LONG_WINDOW, min_periods=1).apply(count_distinct_lineups, raw=False))

    # Calculate delta churn
    df_team_top5_lineups['delta_churn'] = df_team_top5_lineups['rolling_churn_short'] - df_team_top5_lineups['rolling_churn_long']

    # Merge churn features back into df_games
    df_games = pd.merge(
        df_games,
        df_team_top5_lineups[['gameId', 'teamId', 'rolling_churn_short', 'rolling_churn_long', 'delta_churn']].rename(columns={'teamId': 'hometeamId'}),
        on=['gameId', 'hometeamId'],
        how='left',
        suffixes=('', '_home')
    )
    df_games.rename(columns={
        'rolling_churn_short_home': 'home_rolling_churn_short',
        'rolling_churn_long_home': 'home_rolling_churn_long',
        'delta_churn_home': 'home_delta_churn'
    }, inplace=True)

    df_games = pd.merge(
        df_games,
        df_team_top5_lineups[['gameId', 'teamId', 'rolling_churn_short', 'rolling_churn_long', 'delta_churn']].rename(columns={'teamId': 'awayteamId'}),
        on=['gameId', 'awayteamId'],
        how='left',
        suffixes=('', '_away')
    )
    df_games.rename(columns={
        'rolling_churn_short_away': 'away_rolling_churn_short',
        'rolling_churn_long_away': 'away_rolling_churn_long',
        'delta_churn_away': 'away_delta_churn'
    }, inplace=True)

    print("Lineup churn score calculated.")

    print("Lineup churn score calculated.")

    # --- Feature Engineering: Shooting Variance (Beta Posteriors) ---
    print("Calculating shooting variance features...")

    # Prepare data for rolling shooting calculations
    team_shooting_stats = []
    for team_id in pd.concat([df_games['hometeamId'], df_games['awayteamId']]).unique():
        team_games_shooting = df_games[
            (df_games['hometeamId'] == team_id) | (df_games['awayteamId'] == team_id)
        ].copy()
        team_games_shooting['teamId'] = team_id
        team_games_shooting['3PM'] = np.where(
            team_games_shooting['hometeamId'] == team_id, team_games_shooting['threePointersMade_home_agg'], team_games_shooting['threePointersMade_away_agg']
        )
        team_games_shooting['3PA'] = np.where(
            team_games_shooting['hometeamId'] == team_id, team_games_shooting['threePointersAttempted_home_agg'], team_games_shooting['threePointersAttempted_away_agg']
        )
        team_games_shooting['FTM'] = np.where(
            team_games_shooting['hometeamId'] == team_id, team_games_shooting['freeThrowsMade_home_agg'], team_games_shooting['freeThrowsMade_away_agg']
        )
        team_games_shooting['FTA'] = np.where(
            team_games_shooting['hometeamId'] == team_id, team_games_shooting['freeThrowsAttempted_home_agg'], team_games_shooting['freeThrowsAttempted_away_agg']
        )
        team_shooting_stats.append(team_games_shooting[['gameId', 'gameDate', 'teamId', '3PM', '3PA', 'FTM', 'FTA']])

    df_team_shooting = pd.concat(team_shooting_stats).sort_values(by=['teamId', 'gameDate']).reset_index(drop=True)

    # Calculate rolling sums for makes and attempts
    df_team_shooting['rolling_3PM'] = df_team_shooting.groupby('teamId')['3PM']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).sum())
    df_team_shooting['rolling_3PA'] = df_team_shooting.groupby('teamId')['3PA']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).sum())
    df_team_shooting['rolling_FTM'] = df_team_shooting.groupby('teamId')['FTM']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).sum())
    df_team_shooting['rolling_FTA'] = df_team_shooting.groupby('teamId')['FTA']\
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).sum())

    # Calculate Beta posterior parameters (alpha, beta) for 3P% and FT%
    # Using a uniform prior Beta(1,1)
    df_team_shooting['alpha_3P'] = df_team_shooting['rolling_3PM'] + 1
    df_team_shooting['beta_3P'] = df_team_shooting['rolling_3PA'] - df_team_shooting['rolling_3PM'] + 1
    df_team_shooting['alpha_FT'] = df_team_shooting['rolling_FTM'] + 1
    df_team_shooting['beta_FT'] = df_team_shooting['rolling_FTA'] - df_team_shooting['rolling_FTM'] + 1

    # Calculate team season average 3P% and FT% (for comparison)
    df_team_shooting['season_3P_pct'] = df_team_shooting.groupby('teamId')['3PM'].transform('cumsum') / df_team_shooting.groupby('teamId')['3PA'].transform('cumsum')
    df_team_shooting['season_FT_pct'] = df_team_shooting.groupby('teamId')['FTM'].transform('cumsum') / df_team_shooting.groupby('teamId')['FTA'].transform('cumsum')

    # Fill any NaN or inf values that might result from division by zero
    df_team_shooting['season_3P_pct'] = df_team_shooting['season_3P_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df_team_shooting['season_FT_pct'] = df_team_shooting['season_FT_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Merge shooting variance features back into df_games
    df_games = pd.merge(
        df_games,
        df_team_shooting[['gameId', 'teamId', 'alpha_3P', 'beta_3P', 'alpha_FT', 'beta_FT', 'season_3P_pct', 'season_FT_pct']].rename(columns={'teamId': 'hometeamId'}),
        on=['gameId', 'hometeamId'],
        how='left',
        suffixes=('', '_home')
    )
    df_games.rename(columns={
        'alpha_3P_home': 'home_alpha_3P',
        'beta_3P_home': 'home_beta_3P',
        'alpha_FT_home': 'home_alpha_FT',
        'beta_FT_home': 'home_beta_FT',
        'season_3P_pct_home': 'home_season_3P_pct',
        'season_FT_pct_home': 'home_season_FT_pct'
    }, inplace=True)

    df_games = pd.merge(
        df_games,
        df_team_shooting[['gameId', 'teamId', 'alpha_3P', 'beta_3P', 'alpha_FT', 'beta_FT', 'season_3P_pct', 'season_FT_pct']].rename(columns={'teamId': 'awayteamId'}),
        on=['gameId', 'awayteamId'],
        how='left',
        suffixes=('', '_away')
    )
    df_games.rename(columns={
        'alpha_3P_away': 'away_alpha_3P',
        'beta_3P_away': 'away_beta_3P',
        'alpha_FT_away': 'away_alpha_FT',
        'beta_FT_away': 'away_beta_FT',
        'season_3P_pct_away': 'away_season_3P_pct',
        'season_FT_pct_away': 'away_season_FT_pct'
    }, inplace=True)

    print("Shooting variance features calculated.")

    print("Shooting variance features calculated.")

    # --- Feature Engineering: Simulated Shooting Variance Features ---
    print("Calculating simulated shooting variance features...")

    NUM_SIMULATIONS = 1000

    # Function to simulate and calculate features for a given row
    def calculate_simulated_shooting_features(row, team_prefix):
        alpha_3P = row[f'{team_prefix}_alpha_3P']
        beta_3P = row[f'{team_prefix}_beta_3P']
        season_3P_pct = row[f'{team_prefix}_season_3P_pct']

        if pd.isna(alpha_3P) or pd.isna(beta_3P) or alpha_3P <= 0 or beta_3P <= 0:
            return np.nan, np.nan

        # Simulate 3P% from Beta posterior
        simulated_3P_pcts = np.random.beta(alpha_3P, beta_3P, NUM_SIMULATIONS)

        # P(3P% > season average)
        P_3P_gt_season_avg = (simulated_3P_pcts > season_3P_pct).mean()

        # SD(simulated 3P%)
        SD_simulated_3P_pct = np.std(simulated_3P_pcts)

        return P_3P_gt_season_avg, SD_simulated_3P_pct

    # Apply the function for home team
    df_games[['home_P_3P_gt_season_avg', 'home_SD_simulated_3P_pct']] = df_games.apply(
        lambda row: calculate_simulated_shooting_features(row, 'home'), axis=1, result_type='expand'
    )

    # Apply the function for away team
    df_games[['away_P_3P_gt_season_avg', 'away_SD_simulated_3P_pct']] = df_games.apply(
        lambda row: calculate_simulated_shooting_features(row, 'away'), axis=1, result_type='expand'
    )

    print("Simulated shooting variance features calculated.")

    print("Simulated shooting variance features calculated.")

    # --- Placeholder for Scenario Features (Counterfactual Monte Carlo) ---
    print("Placeholder for Scenario Features: Requires a full game simulation engine and advanced statistical models.")

    # "No-Closer" counterfactual
    df_games['home_no_closer_delta_win_prob'] = np.nan # Placeholder
    df_games['away_no_closer_delta_win_prob'] = np.nan # Placeholder
    df_games['home_no_closer_delta_margin'] = np.nan # Placeholder
    df_games['away_no_closer_delta_margin'] = np.nan # Placeholder

    # Pace-stress test
    df_games['home_pace_stress_turnover_delta'] = np.nan # Placeholder
    df_games['away_pace_stress_turnover_delta'] = np.nan # Placeholder
    df_games['home_pace_stress_3P_volume_delta'] = np.nan # Placeholder
    df_games['away_pace_stress_3P_volume_delta'] = np.nan # Placeholder

    # --- Placeholder for Uncertainty Features (posterior predictive) ---
    print("Placeholder for Uncertainty Features: Requires advanced statistical modeling and simulation.")

    # Minutes uncertainty
    df_games['home_minutes_entropy_simulated'] = np.nan # Placeholder
    df_games['away_minutes_entropy_simulated'] = np.nan # Placeholder
    df_games['home_P_starter_lt_24min'] = np.nan # Placeholder
    df_games['away_P_starter_lt_24min'] = np.nan # Placeholder
    df_games['home_P_bench_closes'] = np.nan # Placeholder
    df_games['away_P_bench_closes'] = np.nan # Placeholder

    # Shooting variance
    df_games['home_P_3P_gt_season_p75'] = np.nan # Placeholder
    df_games['away_P_3P_gt_season_p75'] = np.nan # Placeholder
    df_games['home_expected_3P_made_spread'] = np.nan # Placeholder
    df_games['away_expected_3P_made_spread'] = np.nan # Placeholder

    # Possessions
    df_games['home_P_total_gt_threshold'] = np.nan # Placeholder
    df_games['away_P_total_gt_threshold'] = np.nan # Placeholder
    df_games['home_SD_total_possessions'] = np.nan # Placeholder
    df_games['away_SD_total_possessions'] = np.nan # Placeholder
    df_games['home_tail_risk_possessions'] = np.nan # Placeholder
    df_games['away_tail_risk_possessions'] = np.nan # Placeholder

    # Outcome micro-risk
    df_games['home_P_upset_high_variance'] = np.nan # Placeholder
    df_games['away_P_upset_high_variance'] = np.nan # Placeholder
    df_games['home_P_margin_gt_X'] = np.nan # Placeholder
    df_games['away_P_margin_gt_X'] = np.nan # Placeholder

    # --- Placeholder for Injury Compounding v2 ---
    print("Placeholder for Injury Compounding v2: Requires injury multiplier, lost minutes share, and player usage data.")
    df_games['home_weighted_injury_impact'] = np.nan # Placeholder
    df_games['away_weighted_injury_impact'] = np.nan # Placeholder

    # --- Placeholder for Collapse Rate (requires more detailed data) ---
    print("Placeholder for Collapse Rate: Requires quarter-by-quarter scores or play-by-play data.")
    df_games['home_collapse_rate'] = np.nan # Placeholder
    df_games['away_collapse_rate'] = np.nan # Placeholder

    # --- Placeholder for Opponent Collapse Rate (requires more detailed data) ---
    print("Placeholder for Opponent Collapse Rate: Requires quarter-by-quarter scores or play-by-play data.")
    df_games['opp_home_collapse_rate'] = np.nan # Placeholder
    df_games['opp_away_collapse_rate'] = np.nan # Placeholder

    # --- Placeholder for bench_finisher_points x opp_collapse_rate ---
    print("Placeholder for bench_finisher_points x opp_collapse_rate.")
    df_games['home_bench_finisher_x_opp_collapse'] = np.nan # Placeholder
    df_games['away_bench_finisher_x_opp_collapse'] = np.nan # Placeholder

    # --- Placeholder for 3P defense volatility ---
    print("Placeholder for 3P defense volatility: Requires historical opponent 3P% data.")
    df_games['home_opp_3P_def_volatility'] = np.nan # Placeholder
    df_games['away_opp_3P_def_volatility'] = np.nan # Placeholder

    # --- Placeholder for team_3P_reliance x opp_3P_def_var ---
    print("Placeholder for team_3P_reliance x opp_3P_def_var.")
    df_games['home_3P_reliance_x_opp_3P_def_var'] = np.nan # Placeholder
    df_games['away_3P_reliance_x_opp_3P_def_var'] = np.nan # Placeholder

    # --- Save DataFrame with new features ---
    print("Saving DataFrame with lineup consistency features...")
    df_games.to_parquet(os.path.join(processed_data_path, "games_with_lineup_consistency.parquet"), index=False)
    print("DataFrame saved to 'data/processed/games_with_lineup_consistency.parquet'.")
