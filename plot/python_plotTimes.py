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
    transfer_sizes = [int(match.group(1)) for match in re.finditer(r'Cycle:\s+(\d+)', data)]
    transfer_times = [float(match.group(1)) for match in re.finditer(r'Elapsed Time \(s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Cycle': transfer_sizes, 'Elapsed Time (s)': transfer_times})

def get_position(a, b, c):
    return (c//a, c%b)

lable_colors = { 'baseline': 'blue', 'Internode': 'blue', 'CudaAware': 'red','InternodeCudaAware': 'red', 'Nccl': 'green', 'InternodeNccl': 'green', 'Nvlink': 'black', 'InternodeNvlink': 'black' }


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

    line_order = list(lable_colors.keys())
    print('line_order: ' + str(line_order))
    print('all_machines: ' + str(all_machines))
    print('all_experiments: ' + str(all_experiments))

    plots={}
    for m in all_machines:
        for e in all_experiments:
            plots[m+e]={}

    for f in files:
        plots[f[0]+f[1]][f[2]] = f[3]

    for key in plots:
        print("key: ", key)

        fig,axs = plt.subplots(4, 4, sharex='all')
        fig.tight_layout()
        plt.xlabel('Iteration')
        plt.ylabel('Elapsed Time (s)')

        for line in line_order:
            if line in plots[key]:
                print("line: ", line)

                BSize = plots[key][line]['Cycle']
                all_bs = []
                for bs in BSize:
                    if bs not in all_bs:
                        all_bs.append(bs)
                print('all_bs: ', all_bs)


                for bs in all_bs:
                    if bs % 2 == 0:
                        subset = plots[key][line][plots[key][line]['Cycle'] == bs]
                        print("subset of %d and %s:" % (bs, line))
                        #print(subset)

                        transfer_time = list(subset['Elapsed Time (s)'])
                        print("transfer_time: ", transfer_time)

                        linestyle = '--' if 'Internode' in line else '-'

                        bs_x = int(bs//2)//4
                        bs_y = int(bs//2)%4
                        print("bs = ", bs, ", ", bs_x, bs_y)
                        #axs[bs_x, bs_y].bar(j, transfer_time)
                        axs[bs_x, bs_y].plot(range(0,50), transfer_time, label=line, linestyle=linestyle, color=lable_colors[line])

            #for bs in all_bs:
                #if bs % 2 == 0:
                    #j = 0
                    #bs_x = int(bs//2)//4
                    #bs_y = int(bs//2)%4
                    #print("bs = ", bs, ", ", bs_x, bs_y)
                    #for i in range(0,len(BSize)):
                        #if BSize[i] == bs:
                            #axs[bs_x, bs_y].bar(j, transfer_time[i])
                            #j += 1

                    #axs[bs_x, bs_y].set_title('Byte Size ' + str(2**bs) + ' B')

        e = 'ping-pong' if 'pp' in key else 'all2all'
        m = 'Leonardo' if 'leonardo' in key else 'Marzola'
        fig.suptitle(m + ' ' + e + ' ' + key, y = 0.05)
        plt.show()

# Specify the directory containing the files
directory_path = 'sout/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot performance comparison
plot_performance(file_paths)
