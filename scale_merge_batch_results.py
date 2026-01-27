import os
import glob
import pandas as pd
import re

def main():
    # Define paths
    results_dir = "scale_batch/results"
    output_file = "scale_results_aggregated.csv"
    
    # Pattern to match your batch files
    # Matches: results_n50000_k4_bs512.csv
    file_pattern = os.path.join(results_dir, "results_n*_k*_bs*.csv")
    all_files = glob.glob(file_pattern)
    
    print(f"Found {len(all_files)} result files in {results_dir}...")
    
    if not all_files:
        print("No files found. Check your directory.")
        return

    combined_data = []

    # Regex to extract metadata from filename
    # expecting format: results_n{n}_k{k}_bs{bs}.csv
    filename_regex = re.compile(r"results_n(\d+)_k(\d+)_bs(\d+)\.csv")

    for filename in all_files:
        try:
            # Read the CSV
            df = pd.read_csv(filename)
            
            # Extract metadata from filename
            basename = os.path.basename(filename)
            match = filename_regex.search(basename)
            
            if match:
                # We trust the filename metadata for n, k, and batch_size
                # specifically to capture batch_size which isn't in the CSV content
                meta_n = int(match.group(1))
                meta_k = int(match.group(2))
                meta_bs = int(match.group(3))
                
                # Add batch_size column (and overwrite n/k to be safe/consistent)
                df['batch_size'] = meta_bs
                df['n'] = meta_n
                df['k'] = meta_k
            
            combined_data.append(df)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if combined_data:
        # Concatenate all dataframes
        master_df = pd.concat(combined_data, ignore_index=True)
        
        # Sort for cleaner reading
        master_df.sort_values(by=['k', 'batch_size', 'n'], inplace=True)
        
        # Save to disk
        master_df.to_csv(output_file, index=False)
        print(f"Successfully merged data into '{output_file}'")
        print(master_df.head())
        print(f"Total rows: {len(master_df)}")
    else:
        print("No valid data found to merge.")

if __name__ == "__main__":
    main()