import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def unpack(file_paths):
    files=[]
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        machines, experiments, label, _ = file_name.split('_')
        # linestyle = '-' if "Internode" in label else '--'
        df = extract_data(file_path)
        files.append([machines,experiments,label,df])
        #print(files)
    return files


# Function to extract data from a file
substring = "[Average]"
def extract_data(file_path):
    with open(file_path, 'r') as file:
        #data = file.read()
        data = '\n'.join(line.strip() for line in file if substring in line)

    # Use regular expressions to extract relevant information
    transfer_sizes = [int(match.group(1)) for match in re.finditer(r'Transfer size \(B\):\s+(\d+)', data)]
    transfer_times = [float(match.group(1)) for match in re.finditer(r'Transfer Time \(s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Transfer Time (s)': transfer_times})

lable_colors = { 'baseline': 'blue', 'Internode': 'blue', 'CudaAware': 'red','InternodeCudaAware': 'red', 'Nccl': 'green', 'InternodeNccl': 'green', 'Nvlink': 'gray', 'InternodeNvlink': 'gray' }

# Function to plot performance comparison
def plot_performance(file_paths):
    all_machines = []
    all_experiments = []
    files = unpack(file_paths)
    for i in files:
        if i[0] not in all_machines:
            all_machines.append(i[0])
        if i[1] not in all_experiments:
            all_experiments.append(i[1])

    print('all_machines: ' + str(all_machines))
    print('all_experiments: ' + str(all_experiments))

    plots={}
    for m in all_machines:
        for e in all_experiments:
            plots[m+e]={}

    for f in files:
        plots[f[0]+f[1]][f[2]] = f[3]

    bar_order = list(lable_colors.keys())

    for key in plots.keys():

        plt.figure(figsize=(10, 6))
        for bar in bar_order:
            if bar in plots[key]:
                print('Key: ', key)
                print('Bar: ', bar)

                smallest_size = plots[key][bar]['Transfer Size (B)'][0]
                transfer_time = plots[key][bar]['Transfer Time (s)'][0]

                print('smallest_size: ', smallest_size)
                print('transfer_time: ', transfer_time)
                print('color: ', lable_colors[bar])

                plt.bar(bar, transfer_time, color=lable_colors[bar])

        plt.xlabel('File')
        plt.ylabel('Transfer Time (s)')
        plt.title(f'Transfer Time for Smallest Size ({smallest_size} B) Comparison')
        e = 'ping-pong' if 'pp' in key else 'all2all'
        m = 'Leonardo' if 'leonardo' in key else 'Marzola'
        plt.title(m + ' ' + e + f' Transfer Time for Smallest Size ({smallest_size} B) Comparison')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.grid(axis='y')
        plt.show()

        plt.show()

# Specify the directory containing the files
directory_path = 'sout/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot performance comparison
plot_performance(file_paths)
