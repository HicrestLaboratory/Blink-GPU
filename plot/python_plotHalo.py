import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to extract information from a line
def extract_info(line):
    parts = line.split(',')
    iteration = int(parts[-1].strip().split()[-1])
    size = int(parts[0].split(':')[1].strip())
    time = float(parts[1].split(':')[1].strip())
    bandwidth = float(parts[2].split(':')[1].strip())
    return {'Iteration': iteration, 'Transfer size (B)': size, 'Transfer Time (s)': time, 'Bandwidth (GB/s)': bandwidth}

# Function to read data from a file and add exp_name column
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Create a list of dictionaries, considering only lines with iteration > 0
    my_min = 15
    my_max = 19
    results = []
    for line in data:
        if 'Iteration ' in line:
            info = extract_info(line)
            #if info['Iteration'] > 0 and info['Transfer size (B)'] in range(2**my_min, 2**my_max +1): # for debug
            if info['Iteration'] > 0:
                results.append(info)

    # Create a DataFrame
    df = pd.DataFrame(results)

    # Extract experiment name from the file name
    data_name = file_path.split('_')
    mac_name  = data_name[0]
    exp_name  = data_name[1]
    exp_type  = data_name[2]
    exp_topo  = data_name[3]

    # Add exp_name as a new column
    df['mac_name'] = mac_name
    df['exp_name'] = exp_name
    df['exp_type'] = exp_type
    df['exp_topo'] = exp_topo

    return df

# Folder path
folder_path = 'sout'

# Get a list of files in the folder
files = [file for file in os.listdir(folder_path) if file.endswith('.out')]

# Concatenate DataFrames for all files into a single DataFrame
complete_data = pd.concat([read_data(os.path.join(folder_path, file_name)) for file_name in files])
print(complete_data)

# Define label colors
label_colors = {'Baseline': 'blue', 'CudaAware': 'red', 'Nccl': 'green', 'Nvlink': 'gray'}

# Create custom color palette
palette = [label_colors[label] for label in label_colors]

# Group the data by the specified columns
grouped_data = complete_data.groupby(['mac_name', 'exp_name', 'exp_topo'])

# Create the line plot with confidence intervals for each group
plt.figure(figsize=(10, 6))  # Set the figure size
for name, group in grouped_data:
    print("Name: ", name)
    print(group)
    sns.lineplot(data=group, x='Transfer size (B)', y='Bandwidth (GB/s)', hue='exp_type', errorbar=('ci', 95), palette=label_colors)

    # Add labels and title
    plt.xlabel('Transfer size (Bytes)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('%s %s %s' % (name[0], name[1], name[2]) )

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.grid(True)


    # Show plot
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
