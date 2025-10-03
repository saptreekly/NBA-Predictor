import pandas as pd
import os
import glob

def summarize_parquet_files(processed_data_path, output_file):
    with open(output_file, 'w') as f:
        parquet_files = glob.glob(os.path.join(processed_data_path, '*.parquet'))
        csv_files = glob.glob(os.path.join(processed_data_path, '*.csv'))
        all_files = sorted(parquet_files + csv_files)
        
        if not all_files:
            f.write("No Parquet or CSV files found in the processed data directory.\n")
            return

        for file_path in all_files:
            file_name = os.path.basename(file_path)
            f.write(f"Summary for: {file_name}\n")
            f.write("=" * (len(file_name) + 13) + "\n\n")

            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    f.write(f"Skipping unsupported file type: {file_name}\n\n")
                    continue

                f.write("Column Names:\n")
                for col in df.columns:
                    f.write(f"- {col}\n")
                f.write("\n")

                f.write("First 5 Rows:\n")
                f.write(df.head().to_string() + "\n\n")

                f.write("Last 5 Rows:\n")
                f.write(df.tail().to_string() + "\n\n")

            except Exception as e:
                f.write(f"Error processing {file_name}: {e}\n\n")

if __name__ == "__main__":
    # Define paths
    processed_data_path = 'data/processed'  # Relative to project root
    output_summary_file = 'processed_data_summary.txt'

    summarize_parquet_files(processed_data_path, output_summary_file)
    print(f"Summary written to {output_summary_file}")
