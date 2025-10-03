import pandas as pd
import os

pd.set_option('display.max_columns', None)


# Define the path to the raw data directory
raw_data_path = "/Users/jackweekly/Desktop/NBA/data/raw"

# Define the path to the summary file
summary_file_path = "summary_raw_data.txt"

# Open the summary file in write mode
with open(summary_file_path, "w") as f:
    # Loop through all the files in the raw data directory
    for filename in os.listdir(raw_data_path):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            # Create the full path to the file
            file_path = os.path.join(raw_data_path, filename)
            
            # Write the filename to the summary file
            f.write(f"--- {filename} ---")
            f.write("\n\n")
            
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            
            # Write the number of rows and columns to the summary file
            f.write(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            f.write("\n\n")
            
            # Write the column data types to the summary file
            f.write("Column Data Types:")
            f.write("\n")
            f.write(str(df.dtypes))
            f.write("\n\n")
            
            # Write the first 5 rows of the DataFrame to the summary file
            f.write("Head:")
            f.write("\n")
            f.write(str(df.head()))
            f.write("\n\n")

print(f"Summary of raw data files saved to {summary_file_path}")