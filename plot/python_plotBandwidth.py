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
    bandwidths = [float(match.group(1)) for match in re.finditer(r'Bandwidth \(GB/s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Bandwidth (GB/s)': bandwidths})

# Function to plot performance comparison
def plot_performance(file_paths):
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        linestyle = '-' if "Internode" in file_name else '--'
        label = file_name.split('_')[0]  # Use the file name until the first '_'

        df = extract_data(file_path)
        plt.plot(df['Transfer Size (B)'], df['Bandwidth (GB/s)'], label=label, linestyle=linestyle)

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Transfer Size (B)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Leonardo Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Specify the directory containing the files
directory_path = 'sout/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot performance comparison
plot_performance(file_paths)
