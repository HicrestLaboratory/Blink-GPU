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
        if 'Iteration' in line:
            info = extract_info(line)
            if info['Iteration'] > 0 and info['Transfer size (B)'] in range(2**my_min, 2**my_max +1): # for debug
            #if info['Iteration'] > 0:
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

lable_colors = { 'Baseline': 'blue', 'CudaAware': 'red', 'Nccl': 'green', 'Nvlink': 'gray'}

# Folder path
folder_path = 'sout'

# Get a list of files in the folder
files = [file for file in os.listdir(folder_path) if file.endswith('.out')]

# Concatenate DataFrames for all files into a single DataFrame
complete_data = pd.concat([read_data(os.path.join(folder_path, file_name)) for file_name in files])
print(complete_data)

for exp_name in complete_data['exp_name'].unique():
    exp_data = complete_data[complete_data['exp_name'] == exp_name]
    for exp_topo in complete_data['exp_topo'].unique():
        all_data = exp_data[exp_data['exp_topo'] == exp_topo]

        # Identify the baseline avg time
        baseline_data = {}
        baseline_avg_time = {}
        baseline_exp = 'Baseline'

        for size in all_data['Transfer size (B)'].unique():
            baseline_data[size] = all_data[(all_data['exp_type'] == baseline_exp) & (all_data['Transfer size (B)'] == size)]
            baseline_avg_time[size] = baseline_data[size]['Transfer Time (s)'].mean()

        print('baseline_avg_time[]: ')
        print(baseline_avg_time)

        # Calculate relative times by dividing transfer time by baseline average
        all_data['Power of Two (B)'] = all_data['Transfer size (B)'].apply(lambda x: int(math.log2(x)) )
        all_data['Baseline Avg Time'] = all_data['Transfer size (B)'].map(baseline_avg_time)
        all_data['Relative Time'] = all_data['Transfer Time (s)'] / all_data['Baseline Avg Time']


        # Set up a color palette for experiments
        palette = sns.color_palette([lable_colors[exp] for exp in all_data['exp_type'].unique()])

        # Plot violin plots for each experiment
        plt.figure(figsize=(14, 8))

        # Plot violin plot with unique color and label in legend
        ax = sns.violinplot(x='Power of Two (B)', y='Relative Time', hue='exp_type', data=all_data, inner='quartile', palette=palette)

        # Set vertical labels on x-axis
        limit=2
        plt.ylim(-limit/4, limit)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        #plt.xscale('log', base=2)
        plt.xticks(rotation=90)
        plt.title('Violin Plot of Transfer Time vs. Transfer Size for %s %s' % (exp_name, exp_topo))
        plt.legend(title='Experiment Name', bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()
