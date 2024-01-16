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
def extract_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Use regular expressions to extract relevant information
    transfer_sizes = [int(match.group(1)) for match in re.finditer(r'Transfer size \(B\):\s+(\d+)', data)]
    bandwidths = [float(match.group(1)) for match in re.finditer(r'Bandwidth \(GB/s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Bandwidth (GB/s)': bandwidths})



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
            plots[m+e]=[]

    for f in files:
        plots[f[0]+f[1]].append([f[2],f[3]])

    for key in plots:

        plt.figure(figsize=(10, 6))
        for lines in plots[key]:
            linestyle = '-' if 'Internode' in lines[0] else '--'
            plt.plot(lines[1]['Transfer Size (B)'], lines[1]['Bandwidth (GB/s)'], label=lines[0], linestyle=linestyle)
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.xlabel('Transfer Size (B)')
        plt.ylabel('Bandwidth (GB/s)')
        e = 'ping-pong' if 'pp' in key else 'all2all'
        m = 'Leonardo' if 'leonardo' in key else 'Marzola'
        plt.title(m + ' ' + e + ' Performance Comparison')
        plt.legend()
        plt.grid(True)

        plt.show()

# Specify the directory containing the files
directory_path = 'sout/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot performance comparison
plot_performance(file_paths)
