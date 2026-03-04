import pandas as pd
import argparse
import os

def process_results(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Tip: Did you run 'merge_batch_results.py' first?")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Filter out failures
    if 'status' in df.columns:
        success_count = len(df[df['status'] == 'success'])
        total_count = len(df)
        print(f"Processing {success_count}/{total_count} successful trials.")
        df = df[df['status'] == 'success'].copy()

    # 2. Rename 'k-Level' to '4-Level' for publication standard
    if 'algo' in df.columns:
        df['algo'] = df['algo'].replace({'k-Level': '4-Level'})

    # 3. Ensure numeric columns are actually numeric
    numeric_cols = ['total_time_s', 'peak_gpu_mem_mb', 'cost', 'abs_error', 'rel_error']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Group by N and Algo to get statistics (Mean & Std)
    if 'n' in df.columns and 'algo' in df.columns:
        grouped = df.groupby(['n', 'algo'])
        
        # We explicitly aggregate 'abs_error' (Actual Error) as requested
        agg_funcs = {
            'cost': ['mean', 'std'],
            'abs_error': ['mean', 'std'],
            'total_time_s': ['mean', 'std'],
            'peak_gpu_mem_mb': ['mean', 'std']
        }
        
        # Only use columns that actually exist in the CSV
        existing_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
        
        summary = grouped.agg(existing_agg_funcs).reset_index()
        
        # Flatten MultiIndex columns (e.g., cost_mean, cost_std)
        summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]

        print(f"Writing aggregated results to {output_file}...")
        summary.to_csv(output_file, index=False)
        print("Done. Preview of data:")
        print(summary.head())
    else:
        print("Error: Required columns 'n' or 'algo' missing from CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated Default: Looks for the merged file in the CURRENT (root) directory
    parser.add_argument("--input", default="results_e1_mnist.csv", help="Merged results CSV")
    # Default Output: Saves the summary to the CURRENT (root) directory
    parser.add_argument("--output", default="aggregated_e1_mnist.csv", help="Processed summary CSV")
    args = parser.parse_args()
    
    process_results(args.input, args.output)