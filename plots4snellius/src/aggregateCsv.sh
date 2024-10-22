#!/bin/bash

basefolder="csvfiles/"
mergedfolder="mergedcsv/"

. src/utils/utilsLib.sh

help_funxtion() {
	echo "Print the help function"
	exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --system)
            system="$2"
            shift 2
            ;;
        --benchmark)
            benchmark="$2"
            shift 2
            ;;
        --implementation)
            implementation="$2"
            shift 2
            ;;
        --topology)
            topology="$2"
            shift 2
            ;;
        --partition)
            partition="$2"
            shift 2
            ;;
        -h|--help)
            help_funxtion
            ;;
        *)
            echo "Unknown parameter passed: $1"
            help_funxtion
            ;;
    esac
done

echo "----------------------------------"
echo "system: ${system}"
echo "benchmark: ${benchmark}"
echo "implementation: ${implementation}"
echo "topology: ${topology}"
echo "partition: ${partition}"

echo -e "\n\n"

expand_HR_strings ${system} ${benchmark} ${implementation} ${topology} ${partition}

files2merge=()
for j in "${!utils_lib_result_sys[@]}"
do
        s="${utils_lib_result_sys[$j]}"
        b="${utils_lib_result_ben[$j]}"
        i="${utils_lib_result_imp[$j]}"
        t="${utils_lib_result_top[$j]}"
        p="${utils_lib_result_par[$j]}"

	file="$( ls "${basefolder}${s}_${b}_${i}_${t}_${p}"* | tail -n 1 )"
        files2merge+=( "${file}" )
done


lables=""
echo -e "\nfiles2merge:"
for e in "${files2merge[@]}"
do
	echo -e "\t${e}"
	if [[ "${lables}" == "" ]]
	then
		lables=$( head -1 ${e} )
	else
		if [[ "$( head -1 ${e} )" != "${lables}" ]]
		then
			echo "Error: haders does not coincide"
			echo -e "\t${lables}"
			echo -e "\t$( head -1 ${e} )"
			exit 1
		fi
	fi
done

echo -e "\nAll the lables coincides:\n${lables}"

mergename="${system}_${benchmark}_${implementation}_${topology}_${partition}.csv"

mkdir -p ${mergedfolder}
echo "${lables}" > ${mergedfolder}${mergename}
for e in "${files2merge[@]}"
do
	grep -v "${lables}" ${e} >> ${mergedfolder}${mergename}
done

echo "Files merged into ${mergename}"
