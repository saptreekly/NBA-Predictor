import pandas as pd
import os

# Define the path to the raw data directory
raw_data_path = "./data/raw"

# --- Explore Games.csv ---
print("\n--- Exploring Games.csv ---")
games_csv_path = os.path.join(raw_data_path, "Games.csv")
try:
    df_games = pd.read_csv(games_csv_path)
    print("\nHead of Games.csv:")
    print(df_games.head())
    print("\nInfo of Games.csv:")
    df_games.info()
except FileNotFoundError:
    print(f"Error: {games_csv_path} not found. Please ensure the data is downloaded.")

# --- Explore PlayerStatistics.csv ---
print("\n--- Exploring PlayerStatistics.csv ---")
player_stats_csv_path = os.path.join(raw_data_path, "PlayerStatistics.csv")
try:
    df_player_stats = pd.read_csv(player_stats_csv_path)
    print("\nHead of PlayerStatistics.csv:")
    print(df_player_stats.head())
    print("\nInfo of PlayerStatistics.csv:")
    df_player_stats.info()
except FileNotFoundError:
    print(f"Error: {player_stats_csv_path} not found. Please ensure the data is downloaded.")
