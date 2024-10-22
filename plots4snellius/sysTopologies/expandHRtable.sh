#!/bin/bash

file2expand="${1}"
expandedfile=${file2expand#"HR_"}
rm ${expandedfile}

. ../src/utils/utilsLib.sh

while IFS= read -r line; do
	systems=$( echo ${line} | awk -F, '{ print $1 }' )
	benchmarks=$( echo ${line} | awk -F, '{ print $2 }')
	implementations=$( echo ${line} | awk -F, '{ print $3 }')
	topologies=$( echo ${line} | awk -F, '{ print $4 }')
	partitions=$( echo ${line} | awk -F, '{ print $5 }')
	peak=$( echo ${line} | awk -F, '{ print $6 }')

	if [[ "${line}" == "" ]]
	then
		continue
	fi

	echo "----------------------------------"
	echo "Text read from file: $line"
	echo "systems: ${systems}"
	echo "benchmarks: ${benchmarks}"
	echo "implementations: ${implementations}"
	echo "topologies: ${topologies}"
	echo "partitions: ${partitions}"
	echo "~~~~~~~~~~"

	expand_HR_strings ${systems} ${benchmarks} ${implementations} ${topologies} ${partitions}

	for j in "${!utils_lib_result_sys[@]}"
	do
		s="${utils_lib_result_sys[$j]}"
		b="${utils_lib_result_ben[$j]}"
		i="${utils_lib_result_imp[$j]}"
		t="${utils_lib_result_top[$j]}"
		p="${utils_lib_result_par[$j]}"
		expandedline=$( echo "${s}, ${b}, ${i}, ${t}, ${p}, ${peak}" )
		echo "expandedline: ${expandedline}"
		echo "${expandedline}" >> ${expandedfile}
	done

done < ${file2expand}
