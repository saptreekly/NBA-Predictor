import pandas as pd
import os

processed_data_path = "/Users/jackweekly/Desktop/NBA/data/processed"
file_path = os.path.join(processed_data_path, "games_processed.parquet")

df = pd.read_parquet(file_path)
print(df.columns)