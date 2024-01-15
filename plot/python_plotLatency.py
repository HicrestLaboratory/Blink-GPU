import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract data from a file
def extract_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Use regular expressions to extract relevant information
    transfer_sizes = [int(match.group(1)) for match in re.finditer(r'Transfer size \(B\):\s+(\d+)', data)]
    transfer_times = [float(match.group(1)) for match in re.finditer(r'Transfer Time \(s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Transfer Time (s)': transfer_times})

# Function to split camel case string
def split_camel_case(s):
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', re.sub('([a-z])([0-9])|([0-9])([a-z])', r'\1\3 \2\4', s))

# Function to plot transfer times as a histogram
def plot_transfer_time_histogram(file_paths):
    plt.figure(figsize=(12, 6))

    # Extract data from the first file to get the smallest transfer size
    first_file_path = file_paths[0]
    first_df = extract_data(first_file_path)
    smallest_size = first_df.loc[first_df['Transfer Size (B)'].idxmin(), 'Transfer Size (B)']

    # Plot transfer times as a histogram for the common smallest transfer size
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        df = extract_data(file_path)
        transfer_time = df.loc[df['Transfer Size (B)'] == smallest_size, 'Transfer Time (s)'].iloc[0]
        label = split_camel_case(file_name.split('_')[0])
        plt.bar(label, transfer_time)

    plt.xlabel('File')
    plt.ylabel('Transfer Time (s)')
    plt.title(f'Transfer Time for Smallest Size ({smallest_size} B) Comparison')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.grid(axis='y')
    plt.show()

# Specify the directory containing the files
directory_path = 'sout/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot the transfer time histogram for the common smallest transfer size
plot_transfer_time_histogram(file_paths)
