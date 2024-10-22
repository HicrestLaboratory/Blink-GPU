#!/bin/bash

plotscript="src/genPlot.py"
basefolder="mergedcsv/"
peakfile="sysTopologies/snellius_theoretical_peaks.csv"

. src/utils/utilsLib.sh

mkdir -p ${basefolder}
for f in "${basefolder}"/*
do
	name=$( basename -- ${f} | cut -d. -f1 )

	systems=$( echo ${name} | awk -F_ '{ print $1 }' )
        benchmarks=$( echo ${name} | awk -F_ '{ print $2 }' )
        implementations=$( echo ${name} | awk -F_ '{ print $3 }' )
        topologies=$( echo ${name} | awk -F_ '{ print $4 }' )
        partitions=$( echo ${name} | awk -F_ '{ print $5 }' )

	echo "----------------------------------"
	echo "systems: ${systems}"
	echo "benchmarks: ${benchmarks}"
	echo "implementations: ${implementations}"
	echo "topologies: ${topologies}"
	echo "partitions: ${partitions}"

	echo -e "\n\n"

	expand_HR_strings ${systems} ${benchmarks} ${implementations} ${topologies} ${partitions}

	peaks=()
	for j in "${!utils_lib_result_sys[@]}"
	do
        	s="${utils_lib_result_sys[$j]}"
	        b="${utils_lib_result_ben[$j]}"
        	i="${utils_lib_result_imp[$j]}"
	        t="${utils_lib_result_top[$j]}"
        	p="${utils_lib_result_par[$j]}"

		string="${s}, ${b}, ${i}, ${t}, ${p}"
		if ! grep "${string}" ${peakfile} -q
	        then
        	        echo "${string} not found in ${peakfile}"
                	exit 1
	        fi
        	peak=$( grep "${string}" ${peakfile} | awk -F, '{ print $6 }' | tr -d ' ' )
        	echo "peak: ${peak}"
		
		if [[ "${peak}" != "None" ]]
		then
			peaks+=( "${peak}" )
		fi
	done

	peaksparam=""
	echo "peaks: ${peaks[*]}"
	if [[ "${#peaks[@]}" != "0" ]]
	then
		IFS=" " read -r -a peaks <<< "$(echo "${peaks[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')"
		echo "peaks: ${peaks[*]}"
		tmp=$( echo "${peaks[-1]}" | tr ' ' ':' )
		peaksparam="--peaks ${tmp}"
	fi
	echo "peaksparam: ${peaksparam}"


	python3 ${plotscript} ${basefolder}${name}.csv ${name}.png --title "${benchmarks} ${topologies} Bandwidth (${partitions})" --hue implementation ${peaksparam}
done
