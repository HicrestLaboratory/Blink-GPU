import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

mypalette={0: 'b', 1: 'y', 2: 'g', 3: 'r'}

#Resources_grp = ['InstantPower(mW)', 'InstantTemperature(C)', 'TotalEnergy(mJ)']
Resources_grp = ['InstantPower(W)', 'InstantTemperature(C)', 'TotalEnergy(J)']

MetricsGroups = {'ResourcesGrp': Resources_grp}

Implementations = {'Baseline', 'CudaAware', 'Nccl', 'Nvlink'}

CheckpointColors = {'start': 'g', 'allocd': 'k', 'alloc': 'k', 'wait': 'k', 'cycle': 'r', 'stop': 'g'}

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
    data['InstantPower(W)'] = data['InstantPower(mW)'].apply(lambda x: x/1000)
    data['TotalEnergy(J)'] = data['TotalEnergy(mJ)'].apply(lambda x: x/1000)
    
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
        #fig, axes = plt.subplots(len(group[1]), len(datas), figsize=(40, 64), sharex=True, sharey=mysharedy)
        fig, axes = plt.subplots(len(group[1]), len(datas), figsize=(40, 64), sharex='col', sharey=mysharedy)
        fig.suptitle("Line Plots for %s" %  group[0], y=0.93)
        print('axes: ', axes)

        # Loop over each metric and create a line plot in a separate subplot
        for i, metric in enumerate(group[1]):
            print("    i: ", i, ", metric: ", metric)
            for j, subData in enumerate(subDatas):
                bbox = axes[i,j].get_position()
                new_bbox = (bbox.x0 + bbox.width*0.1, bbox.y0 + bbox.height*0.75, bbox.width*0.5, bbox.height*0.2)
                axestmp=fig.add_axes(new_bbox)

                subCheckpointData = subCheckpointDatas[j]
                print("    j: ", j, ", data: ", files[j])
                if i == 0:
                    axes[0,j].set_title( findImplFromFilename( files[j] ) )
                #sns.lineplot(data=subData, x='Occurrence', y=metric, hue='#deviceId', ax=axes[i,j], linewidth=2, palette=mypalette)
                sns.lineplot(data=subData, x='Occurrence', y=metric, hue='#deviceId', ax=axestmp, linewidth=2, palette=mypalette, legend=False)
               
                filteredCheckpointData =  subCheckpointData [ subCheckpointData['class'] == 'cycle' ]
                print('subCheckpointData:', subCheckpointData)
                cyclemin = filteredCheckpointData['sample'].min()
                cyclemax = filteredCheckpointData['sample'].max()
                print('min, max: ' , cyclemin, cyclemax)
                filteredData =  subData [ cyclemax >= subData['Occurrence'] ]
                filteredData =  filteredData [ filteredData['Occurrence'] >= cyclemin ]
                min_value = filteredData[metric].min()
                max_value = filteredData[metric].max()

                cycleCount=0
                cycleLast=len(filteredCheckpointData)
                for index, row in subCheckpointData.iterrows():
                    k = row['sample']
                    h = row['class']
                    #axes[i,j].axvline(x=k, color=CheckpointColors[h], linestyle='--', linewidth=2.0)
                    if h == 'cycle':
                        myaxe=axes[i,j]
                        #myaxe.vlines(k, min_value, max_value, color='lavender', linestyle='--')
                    if h != 'cycle' or cycleCount == 0 or cycleCount == cycleLast-1:
                        myaxe=axestmp
                        myaxe.axvline(x=k, color=CheckpointColors[h], linestyle='--', linewidth=2.0)
                        if h == 'cycle':
                            cycleCount += 1
                #axes[i,j].legend(title="#deviceId", loc="upper right")
                
                if metric == 'TotalEnergy(J)':
                    filteredData['DeltaCycleTotalEnergy(J)'] = filteredData['TotalEnergy(J)'].apply(lambda x: x - min_value)
                    tmpmetric = 'DeltaCycleTotalEnergy(J)'
                    print('min_value: ' , min_value)
                else:
                    tmpmetric = metric

                #sns.lineplot(data=filteredData, x='Occurrence', y=metric, hue='#deviceId', ax=axestmp, linewidth=2, palette=mypalette)
                sns.lineplot(data=filteredData, x='Occurrence', y=tmpmetric, hue='#deviceId', ax=axes[i,j], linewidth=2, palette=mypalette, legend=False)
                

            axes[i,0].set_ylabel(metric)

        axes[-1,0].set_xlabel("Occurrence")

        
        fig.savefig(output_file)
        plt.close()
        print(f"Plot saved as {output_file}")
