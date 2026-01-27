import pandas as pd
import glob
import os

# 1. Find all result files
all_files = glob.glob("batch/results/results_n*.csv")
all_files.sort()

print(f"Found {len(all_files)} result files. Merging...")

if not all_files:
    print("No result files found! Check batch/results/ directory.")
    exit()

# 2. Read and concatenate
df_list = []
for filename in all_files:
    df = pd.read_csv(filename)
    df_list.append(df)

final_df = pd.concat(df_list, ignore_index=True)

# 3. Save to the single master file you wanted
output_filename = "results_e1_mnist.csv"
final_df.to_csv(output_filename, index=False)

print(f"Successfully merged all jobs into {output_filename}")
print(f"Total rows: {len(final_df)}")