import pandas as pd
import os

processed_data_path = "/Users/jackweekly/Desktop/NBA/data/processed"
file_path = os.path.join(processed_data_path, "TeamStatistics_AdvancedFeatures.csv")

df = pd.read_csv(file_path)
print(df.columns)