import pandas as pd
import numpy as np
import os

# Define paths
PROCESSED_DATA_PATH = "/Users/jackweekly/Desktop/NBA/data/processed"
RAW_DATA_PATH = "/Users/jackweekly/Desktop/NBA/data/raw"

def calculate_advanced_features(df):
    # Ensure gameDate is datetime for sorting
    df['gameDate'] = pd.to_datetime(df['gameDate'])

    # Calculate final margin
    df['finalMargin'] = df['teamScore'] - df['opponentScore']

    # Merge to get opponent's quarter scores
    # Create a unique game_team_id for merging
    df['game_team_id'] = df['gameId'].astype(str) + '_' + df['teamId'].astype(str)
    df['game_opponent_id'] = df['gameId'].astype(str) + '_' + df['opponentTeamId'].astype(str)

    # Prepare opponent data
    opponent_df = df[['game_team_id', 'q1Points', 'q2Points', 'q3Points', 'q4Points']].copy()
    opponent_df.columns = ['game_opponent_id', 'opponent_q1Points', 'opponent_q2Points', 'opponent_q3Points', 'opponent_q4Points']

    df = pd.merge(df, opponent_df, left_on='game_team_id', right_on='game_opponent_id', how='left', suffixes=('', '_opp'))
    df.drop(columns=['game_opponent_id_opp'], inplace=True) # Drop the redundant column

    # Calculate halftime and Q3 margins
    df['halftime_score_team'] = df['q1Points'] + df['q2Points']
    df['halftime_score_opponent'] = df['opponent_q1Points'] + df['opponent_q2Points']
    df['halftime_margin'] = df['halftime_score_team'] - df['halftime_score_opponent']

    df['q3_score_team'] = df['q1Points'] + df['q2Points'] + df['q3Points']
    df['q3_score_opponent'] = df['opponent_q1Points'] + df['opponent_q2Points'] + df['opponent_q3Points']
    df['q3_margin'] = df['q3_score_team'] - df['q3_score_opponent']

    # 1. Lead Elasticity (proxy version)
    # Using biggestLead
    df['lead_elasticity'] = np.where(
        df['biggestLead'].notna(),
        np.abs(df['biggestLead'] - df['finalMargin']) / np.maximum(1, df['biggestLead']),
        np.nan
    )

    # Using halftime/Q3 margin vs final margin
    df['elasticity_ht'] = np.where(
        df['halftime_margin'].notna(),
        np.abs(df['halftime_margin'] - df['finalMargin']) / np.maximum(1, np.abs(df['halftime_margin'])),
        np.nan
    )
    df['elasticity_q3'] = np.where(
        df['q3_margin'].notna(),
        np.abs(df['q3_margin'] - df['finalMargin']) / np.maximum(1, np.abs(df['q3_margin'])),
        np.nan
    )

    # 2. Comeback Propensity Proxy
    # Indicator: comeback = 1 if (halftime_margin < -5 and final_margin > 0)
    df['comeback_ht'] = ((df['halftime_margin'] < -5) & (df['finalMargin'] > 0)).astype(int)
    df['comeback_q3'] = ((df['q3_margin'] < -5) & (df['finalMargin'] > 0)).astype(int)

    # Extra twist: half_deficit_avg
    df['half_deficit'] = np.where(df['halftime_margin'] < 0, df['halftime_margin'], np.nan)

    # Sort by team and date for rolling calculations
    df = df.sort_values(by=['teamId', 'gameDate'])

    # Rolling averages for Lead Elasticity
    df['lead_elasticity_last10'] = df.groupby('teamId')['lead_elasticity'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['lead_elasticity_last20'] = df.groupby('teamId')['lead_elasticity'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['elasticity_ht_last10'] = df.groupby('teamId')['elasticity_ht'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['elasticity_ht_last20'] = df.groupby('teamId')['elasticity_ht'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['elasticity_q3_last10'] = df.groupby('teamId')['elasticity_q3'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['elasticity_q3_last20'] = df.groupby('teamId')['elasticity_q3'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    # Rolling rates for Comeback Propensity
    df['comeback_rate_ht_last10'] = df.groupby('teamId')['comeback_ht'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['comeback_rate_ht_last20'] = df.groupby('teamId')['comeback_ht'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['comeback_rate_q3_last10'] = df.groupby('teamId')['comeback_q3'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['comeback_rate_q3_last20'] = df.groupby('teamId')['comeback_q3'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    # Rolling average for half_deficit_avg
    df['half_deficit_avg_last10'] = df.groupby('teamId')['half_deficit'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['half_deficit_avg_last20'] = df.groupby('teamId')['half_deficit'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    # Drop temporary columns
    df.drop(columns=['game_team_id', 'game_opponent_id', 'opponent_q1Points', 'opponent_q2Points', 'opponent_q3Points', 'opponent_q4Points',
                     'halftime_score_team', 'halftime_score_opponent', 'q3_score_team', 'q3_score_opponent', 'half_deficit'], inplace=True)

    return df

if __name__ == "__main__":
    # Load the TeamStatistics data
    team_stats_file = os.path.join(RAW_DATA_PATH, "TeamStatistics.csv")
    team_stats_df = pd.read_csv(team_stats_file)

    # Calculate advanced features
    team_stats_df_processed = calculate_advanced_features(team_stats_df.copy())

    # Save the processed data
    output_file_path = os.path.join(PROCESSED_DATA_PATH, "TeamStatistics_AdvancedFeatures.csv")
    team_stats_df_processed.to_csv(output_file_path, index=False)
    print(f"Advanced features calculated and saved to {output_file_path}")
