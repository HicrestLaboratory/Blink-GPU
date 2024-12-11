import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

mypalette={0: 'b', 1: 'y', 2: 'g', 3: 'r'}

Resources_grp = ['InstantPower(mW)', 'InstantTemperature(C)', 'TotalEnergy(mJ)']

MetricsGroups = {'ResourcesGrp': Resources_grp}

Implementations = {'Baseline', 'CudaAware', 'Nccl', 'Nvlink'}

def findImplFromFilename ( filename ):
    for imp in Implementations:
        if '_' + imp + '_' in filename:
            return imp
    return 'Unknown'

def checkpointfileFromFile ( filename ):
    return filename.replace("nvmlMesures", "nvmlCheckpoints")

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
checkpointdatas = []
for i in range(1, len(sys.argv)):
    file = sys.argv[i]
    checkpointfile = checkpointfileFromFile(file)
    print('file: ', file)
    print("checkpointfile: ", checkpointfile)
    
    data = pd.read_csv(file)
    checkpointdata = pd.read_csv(checkpointfile)
    data['Occurrence'] = data.groupby('#deviceId').cumcount()
    
    print(str(file), data)
    print(str(checkpointfile), checkpointdata)

    files.append(file)
    datas.append(data)
    checkpointdatas.append(checkpointdata)

    filegpus = data['#deviceId'].unique()
    checkpointgpus = checkpointdata['#deviceId'].unique()
    print('%s GPUs: ' % file, filegpus)
    print('%s GPUs: ' % checkpointfile, checkpointgpus)
    if len(gpus) == 0:
        gpus = filegpus
    else:
        if str(filegpus) != str(gpus):
            print("ERROR: profiled GPUs on the different files does not metch")
            exit()

print('GPUs: ', gpus)

for gpu in gpus:

    subDatas = []
    subCheckpointDatas = []
    for data in datas:
        subData = data.loc[data['#deviceId'] == gpu]
        print("subData", subData)
        subDatas.append(subData)
    for checkpointdata in checkpointdatas:
        subCheckpointData = checkpointdata.loc[checkpointdata['#deviceId'] == gpu]
        print("subCheckpointData", subCheckpointData)
        subCheckpointDatas.append(subCheckpointData)

    for group in MetricsGroups.items():
        output_file = os.path.splitext(files[0])[0] + '_' + group[0] + '_' + str(gpu) + ".png"
        print('output_file: ', output_file)

        if group[0] in {'ALUGrp', 'DataTransferGrp'}:
            mysharedy = True
        else:
            mysharedy = 'row'

        print('len(group[1]): ', len(group[1]))
        fig, axes = plt.subplots(len(group[1]), len(datas), figsize=(40, 64), sharex=True, sharey=mysharedy)
        fig.suptitle("Line Plots for %s" %  group[0], y=0.93)
        print('axes: ', axes)

        # Loop over each metric and create a line plot in a separate subplot
        for i, metric in enumerate(group[1]):
            print("    i: ", i, ", metric: ", metric)
            for j, subData in enumerate(subDatas):
                subCheckpointData = subCheckpointDatas[j]
                print("    j: ", j, ", data: ", files[j])
                if i == 0:
                    axes[0,j].set_title( findImplFromFilename( files[j] ) )
                sns.lineplot(data=subData, x='Occurrence', y=metric, hue='#deviceId', ax=axes[i,j], linewidth=2, palette=mypalette)
                for k in subCheckpointData['sample']:
                    axes[i,j].axvline(x=k, color='red', linestyle='--', linewidth=0.8)
                axes[i,j].legend(title="#deviceId", loc="upper right")
            axes[i,0].set_ylabel(metric)

        axes[-1,0].set_xlabel("Occurrence")

        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved as {output_file}")
