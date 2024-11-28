import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

mypalette={0: 'b', 1: 'y', 2: 'g', 3: 'r'}

Occupancy_grp = ['SMOCC', 'GPUTL']
Resources_grp = ['GPUTL', 'TMPTR', 'POWER']
ALU_grp = ['SMACT', 'TENSO', 'FP64A', 'FP32A', 'FP16A']
DataTransfer_grp = ['PCITX', 'PCIRX', 'NVLTX', 'NVLRX']

MetricsGroups = {'OccupancyGrp': Occupancy_grp, 'ResourcesGrp': Resources_grp, 'ALUGrp': ALU_grp, 'DataTransferGrp': DataTransfer_grp}

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Check if the script is run with the correct number of arguments
if len(sys.argv) < 2:
    print("Usage: python script_name.py <input_file.csv>")
    sys.exit(1)

# Read and parse input file
gpus = []
files = []
datas = []
for i in range(1, len(sys.argv)):
    file = sys.argv[i]
    print('file: ', file)
    data = pd.read_csv(file)
    data['Occurrence'] = data.groupby('#Entity').cumcount()
    print(str(file), data)

    files.append(file)
    datas.append(data)

    filegpus = data['#Entity'].unique()
    print('%s GPUs: ' % file, filegpus)
    if len(gpus) == 0:
        gpus = filegpus
    else:
        if str(filegpus) != str(gpus):
            print("ERROR: profiled GPUs on the different files does not metch")
            exit()

print('GPUs: ', gpus)

for gpu in gpus:

    subDatas = []
    for data in datas:
        subData = data.loc[data['#Entity'] == gpu]
        print(subData)
        subDatas.append(subData)

    for group in MetricsGroups.items():
        output_file = os.path.splitext(files[0])[0] + '_' + group[0] + '_' + str(gpu) + ".png"
        print('output_file: ', output_file)

        if group[0] in {'ALUGrp', 'DataTransferGrp'}:
            mysharedy = True
        else:
            mysharedy = 'row'

        print('len(group[1]): ', len(group[1]))
        fig, axes = plt.subplots(len(group[1]), len(datas), figsize=(20, 32), sharex=True, sharey=mysharedy)
        fig.suptitle("Line Plots for %s" %  group[0], y=0.93)
        print('axes: ', axes)

        # Loop over each metric and create a line plot in a separate subplot
        for i, metric in enumerate(group[1]):
            print("    i: ", i, ", metric: ", metric)
            for j, subData in enumerate(subDatas):
                print("    j: ", j, ", data: ", files[j])
                if i == 0:
                    axes[0,j].set_title(files[j])
                sns.lineplot(data=subData, x='Occurrence', y=metric, hue='#Entity', ax=axes[i,j], linewidth=2, palette=mypalette)
                axes[i,j].legend(title="#Entity", loc="upper right")
            axes[i,0].set_ylabel(metric)

        axes[-1,0].set_xlabel("Occurrence")

        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved as {output_file}")
