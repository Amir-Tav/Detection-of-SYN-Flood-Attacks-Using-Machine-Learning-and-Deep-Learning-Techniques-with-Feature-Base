import pandas as pd
import math

# Load the dataset
df = pd.read_csv(r"C:\Users\44776\Downloads\Kfold\K5_Dataset.csv")

# Number of rows and size of each split
total_rows = len(df)
chunk_size = math.ceil(total_rows / 5)

# Split into 5 groups and output line numbers
print("Line numbers (0-indexed) for each of the 5 equal groups:\n")
for i in range(5):
    start = i * chunk_size
    end = min(start + chunk_size, total_rows)
    line_numbers = list(range(start, end))
    print(f"Group {i+1}: Lines {start} to {end - 1} ({len(line_numbers)} lines)")
