peakfile="sysTopologies/snellius_theoretical_peaks.csv"
csvgenfile="src/csvgen.sh"
plotgenfile="src/genPlot.py"

resfolder="out/"
midfolder="midfiles/"
csvfolder="csvfiles/"
plotfolder="pngimages/"

for f in ${resfolder}*.out
do 
	filename=$( basename -- ${f} | awk -F. '{ print $1 }' )
	system=$( echo ${filename} | awk -F_ '{ print $1 }' )
	benchmark=$( echo ${filename} | awk -F_ '{ print $2 }' )
	implementation=$( echo ${filename} | awk -F_ '{ print $3 }' )
	topology=$( echo ${filename} | awk -F_ '{ print $4 }' )
	partition=$( echo ${filename} | awk -F_ '{ print $5 }' )
	
	echo "------------------------------------------------------"
	echo "filename: ${filename}"
	echo "system: ${system}"
	echo "benchmark: ${benchmark}"
	echo "implementation: ${implementation}"
	echo "topology: ${topology}"
	echo "partition: ${partition}"

	mkdir -p ${midfolder}	
	grep "(B)" ${resfolder}${filename}.out | grep -v "Average" > ${midfolder}${filename}.txt

	mkdir -p ${csvfolder}
	if [[ ! -f "${csvfolder}${filename}.csv" ]]
	then
		./${csvgenfile} ${midfolder}${filename}.txt ${csvfolder}${filename}.csv "system:${system}" "benchmark:${benchmark}" "implementation:${implementation}" "topology:${topology}" "partition:${partition}"
	fi
	
	string="${system}, ${benchmark}, ${implementation}, ${topology}, ${partition}"
        echo "string: ${string}"
	if ! grep "${string}" ${peakfile} -q
	then
		echo "${string} not found in ${peakfile}"
		exit 1
	fi
	peak=$( grep "${string}" ${peakfile} | awk -F, '{ print $6 }' | tr -d ' ' )
	echo "peak: ${peak}"
        
	optinputs=""
	if [[ "${peak}" != "None" ]]
        then
                optinputs="--peaks ${peak}"
        fi
        echo "Optional Inputs: ${optinputs}"

	mkdir -p ${plotfolder}
	python3 ${plotgenfile} ${csvfolder}${filename}.csv ${plotfolder}${filename}.png ${optinputs}
done
