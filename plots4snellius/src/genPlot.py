import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import matplotlib.patches as mpatches

my_palette = {'Baseline': 'b', 'CudaAware': 'y', 'Nccl': 'g', 'Nvlink': 'r'}

def line_plot_from_file(input_file, outname, title, xlable='Message Size', ylable='Time', peaks=None, myhue=None):
    """
    Generates a line plot from an input CSV file.

    Parameters:
    - input_file: Path to the input CSV file.
    - outname: Path to the output image file.
    - title: Title of the plot.
    - xlable: Label for the X-axis (default: 'Message Size').
    - ylable: Label for the Y-axis (default: 'Time').
    - peaks: Optional peaks value to plot as a horizontal reference line, multiple values must be divided by ':' (default: None).
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, skipinitialspace=True)
    print(df)

    # Set global font size, line width, and marker size
    my_fontsize = 25
    my_linewidth = 5
    my_markersize = 15

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Extract the relevant columns
    collectorname = "TransferSize(B)"
    valuename = "Bandwidth(GB/s)"
    df[valuename] = df['Bandwidth(GiB/s)'] * 1.07374

    df['Transfer size (B)'] = pd.to_numeric(df['TransferSize(B)'])

    # Line plot with seaborn
    if (myhue != None): 
        cust_palette = my_palette
    else:
        cust_palette = None
    sns.lineplot(data=df, x=collectorname, y=valuename, markers=True, marker='o', dashes=False,
                 linewidth=my_linewidth, markersize=my_markersize, hue=myhue, palette=cust_palette)

    # Plot the peak line if provided
    if peaks is not None:
        peaks_vec = peaks.split(':')
        peaks_vec = list(map(float, list(peaks_vec)))
        max_peak = max(peaks_vec)
        print('Type of peaks_vec[0]: ', type(peaks_vec[0]))
        print('peaks_vec:', peaks_vec)
        print('max_peak:', max_peak)

        for peak_str in peaks_vec:
            peak = float(peak_str)
            plt.axhline(y=peak, color='r', linestyle='--')
            peak_label = f"Theoretical peak: {peak:.2f} GB/s"
            x_min, x_max = plt.gca().get_xlim()
            x_center = (x_min + x_max) / 4
            plt.text(x=20,  y=peak, s=peak_label, color='r', va='center', ha='center', backgroundcolor='w', fontsize=2*my_fontsize/3)

    # Max achieved value line and annotation
    if myhue == None:
        max_achieved = df[valuename].max()
        max_label = f"Max achieved value: {max_achieved:.2f}"
        plt.axhline(y=max_achieved, color='g', linestyle='--')

        x_min, x_max = plt.gca().get_xlim()
        x_center = (x_min + x_max) / 2
        if peaks is not None:
            y_center = max_peak / 2
        else:
            y_center = max_achieved / 2
        plt.text(x=x_max + 0.5, y=y_center, s=max_label, color='g', va='center', ha='left', backgroundcolor='w', rotation=90, fontsize=my_fontsize)
    else:
        maxvalues={}

        huevec = df[myhue].unique().tolist()
        for e in huevec:
            subdataset = df[df[myhue] == e]
            max_achieved = subdataset[valuename].max()
            maxvalues[e] = max_achieved
    
        #custom_legend = [f"{key} ({value:.2f})" for key, value in maxvalues.items()]
        max_achieved = max(maxvalues.values())

    # Customize the labels and ticks
    plt.xlabel(xlable, fontsize=my_fontsize)
    plt.ylabel(ylable, fontsize=my_fontsize)
    plt.xticks(rotation=30, fontsize=my_fontsize)
    plt.yticks(fontsize=my_fontsize)
    plt.xscale('log')

    # Set title
    plt.title(title, fontsize=my_fontsize)

    if myhue != None:
    #    plt.legend(title='Implementation (maxacheved)', labels=custom_legend, fontsize=12)
        my_handles = []
        for key in maxvalues.keys():
            tmp = mpatches.Patch(color=my_palette[key], label='%s (%.2f)' % (key, maxvalues[key]) )
            my_handles.append(tmp)
        plt.legend(title='Implementation (max achieved)', handles=my_handles, fontsize=12)

    # Adjust layout
    plt.tight_layout(pad=1.5)

    # Save the plot
    plt.savefig(outname)
    plt.close()

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Generate a line plot from a CSV file.')

    # Add command-line arguments
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('outname', type=str, help='Path to the output image file')
    parser.add_argument('--title', type=str, default='Bandwidth vs Transfer Size', help='Title of the plot (default: "Bandwidth vs Transfer Size")')
    parser.add_argument('--xlable', type=str, default='Message Size', help='Label for the X-axis (default: "Message Size")')
    parser.add_argument('--ylable', type=str, default='Bandwidth (GB/s)', help='Label for the Y-axis (default: "Time")')
    parser.add_argument('--peaks', type=str, default=None, help='Optional peaks value for reference (default: None)')
    parser.add_argument('--hue', type=str, default=None, help='Optional hue value for merged data (default: None)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    line_plot_from_file(args.input_file, args.outname, args.title, args.xlable, args.ylable, args.peaks, args.hue)

