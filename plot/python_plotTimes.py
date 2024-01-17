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


# Function to plot performance comparison
def plot_performance(file_paths):
    all_machines = []
    all_experiments = []
    all_communications = []
    files = unpack(file_paths)
    for i in files:
        if i[0] not in all_machines:
            all_machines.append(i[0])
        if i[1] not in all_experiments:
            all_experiments.append(i[1])
        if i[2] not in all_communications:
            all_communications.append(i[2])

    print('all_machines: ' + str(all_machines))
    print('all_experiments: ' + str(all_experiments))
    print('all_communications: ' + str(all_communications))

    plots={}
    for m in all_machines:
        for e in all_experiments:
            for c in all_communications:
                plots[m+e+c]=[]

    for f in files:
        plots[f[0]+f[1]+f[2]].append(f[3])

    for key in plots:

        for lines in plots[key]:
            transfer_time = lines['Elapsed Time (s)']
            BSize = lines['Cycle']
            all_bs = []
            for bs in BSize:
                if bs not in all_bs:
                    all_bs.append(bs)
            print('all_bs: ', all_bs)

            fig,axs = plt.subplots(4, 4, sharex='all')
            fig.tight_layout()
            plt.xlabel('Iteration')
            plt.ylabel('Elapsed Time (s)')


            for bs in all_bs:
                if bs % 2 == 0:
                    j = 0
                    bs_x = int(bs//2)//4
                    bs_y = int(bs//2)%4
                    print("bs = ", bs, ", ", bs_x, bs_y)
                    for i in range(0,len(BSize)):
                        if BSize[i] == bs:
                            axs[bs_x, bs_y].bar(j, transfer_time[i])
                            j += 1

                    axs[bs_x, bs_y].set_title('Byte Size ' + str(2**bs) + ' B')

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
