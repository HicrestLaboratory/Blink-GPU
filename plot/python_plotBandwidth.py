import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def unpack(file_paths):
    files=[]
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        machines, experiments, types, topology, _ = file_name.split('_')
        # linestyle = '-' if "Internode" in label else '--'
        df = extract_data(file_path)
        files.append([machines,experiments,types,topology,df])
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
    bandwidths = [float(match.group(1)) for match in re.finditer(r'Bandwidth \(GB/s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Bandwidth (GB/s)': bandwidths})


lable_colors = { 'Baseline': 'blue', 'CudaAware': 'red', 'Nccl': 'green', 'Nvlink': 'gray'}

lable_machines = { 'leonardo': 'Leonardo', 'marzola': 'Marzola'}
lable_topologyes = { 'singlenode': 'Single node', 'multinode': 'Multi nodes'}
lable_experiments = { '-pp-': 'Ping-pong', '-a2a-': 'AllToAll', '-ar-': 'AllReduce'}

# Function to plot performance comparison
def plot_performance(file_paths):

    all_types = []
    all_machines = []
    all_topologyes = []
    all_experiments = []
    files = unpack(file_paths)
    for i in files:
        if i[0] not in all_machines:
            all_machines.append(i[0])
        if i[1] not in all_experiments:
            all_experiments.append(i[1])
        if i[2] not in all_types:
            all_types.append(i[2])
        if i[3] not in all_topologyes:
            all_topologyes.append(i[3])

    print('all_types: ' + str(all_types))
    print('all_machines: ' + str(all_machines))
    print('all_topologyes: ' + str(all_topologyes))
    print('all_experiments: ' + str(all_experiments))

    plots={}
    for m in all_machines:
        for e in all_experiments:
            for t in all_topologyes:
                plots[m+'-'+e+'-'+t]={}

    for f in files:
        plots[f[0]+'-'+f[1]+'-'+f[3]][f[2]] = f[4]

    line_order = list(lable_colors.keys())

    for topology in all_topologyes:
        for key in plots:
            if topology in key:
                plt.figure(figsize=(10, 6))
                for line in line_order:
                    if line in plots[key]:
                        print('Key: ', key)
                        print('Line: ', line)

                        linestyle = '--' if 'multinode' in line else '-'
                        transfer_size = plots[key][line]['Transfer Size (B)']
                        bandwidth = plots[key][line]['Bandwidth (GB/s)']

                        print('transfer_size: ', transfer_size)
                        print('bandwidth: ', bandwidth)
                        print('linestyle: ', linestyle)
                        print('color: ', lable_colors[line])

                        plt.plot(transfer_size, bandwidth, label=line, linestyle=linestyle, color=lable_colors[line])

                #e = 'ping-pong' if 'pp' in key else 'all2all'
                for k in lable_experiments:
                    if k in key:
                        e = lable_experiments[k]
                #m = 'Leonardo' if 'leonardo' in key else 'Marzola'
                for k in lable_machines:
                    if k in key:
                        m = lable_machines[k]
                #t = 'SingleNode' if 'singlenode' in key else 'MultiNode'
                for k in lable_topologyes:
                    if k in key:
                        t = lable_topologyes[k]

                if m == 'Leonardo':
                    if t == 'Single node':
                        if e == 'Ping-pong':
                            peak = 100
                        else:
                            peak = 300
                    else:
                        peak = 25
                plt.axhline(y=peak, color='red', linestyle='--', label='Theoretical peak')

                plt.xscale('log', base=2)
                plt.yscale('log')
                plt.xlabel('Transfer Size (B)')
                plt.ylabel('Bandwidth (GB/s)')

                plt.title(m + ' ' + e + ' ' + t + ' Performance Comparison')
                print(line_order)
                #plt.legend(legend_order)
                plt.legend()
                plt.grid(True)

                plt.show()

# Specify the directory containing the files
directory_path = 'sout/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot performance comparison
plot_performance(file_paths)
