#!/bin/bash

if [[ "$#" -lt "2" ]]
then
	echo "$0 require a keyword to search"
	exit 1
fi

chunklen=10
if [[ "$#" -gt "2" ]]
then
	chunklen="$3"
fi

info="$1"
key="$2"

#files=( "pp_Baseline.cu" "pp_CudaAware.cu" "pp_Nccl.cu" "pp_Nvlink.cu" )

# Creating the files array
benchmarks=( "pp" "a2a" "ar" "mpp" )
implementations=( "Baseline" "CudaAware" "Nccl" "Nvlink" )

files=()
if [[ "${info}" == "all" ]]
then
	for b in "${benchmarks[@]}"
	do
		for i in "${implementations[@]}"
		do
			files+=( "${b}_${i}.cu" )
		done
	done
else
	for b in "${benchmarks[@]}"
	do
		if [[ "${info}" == "${b}" ]]
		then
			for i in "${implementations[@]}"
			do
				files+=( "${b}_${i}.cu" )
			done
			break
		fi
	done
	
	for i in "${implementations[@]}"
	do
		if [[ "${info}" == "${i}" ]]
		then
			for b in "${benchmarks[@]}"
			do
				files+=( "${b}_${i}.cu" )
			done
			break
		fi
	done
fi

echo "files: ${files[*]}"

IFS=$'\n' sfiles=($(sort <<<"${files[*]}")) ; unset IFS

grepcommand="grep -n "${key}" ${sfiles[*]}"
${grepcommand}

# Check for one out per file
greplines=$( ${grepcommand} | wc -l )
if [[ "${greplines}" != "${#sfiles[@]}" ]]
then 
	echo "Error: keyword does not produce one per file out (greplines: ${greplines}, nfiles: ${#sfiles[@]})"
	exit 1
else
	tmp=$( ${grepcommand} | awk -F: '{ print $1 }' | sort | tr '\n' ' ' )
	IFS=$' ' grep_array=( ${tmp}  ) ; unset IFS
	if [[ "${grep_array[*]}" != "${sfiles[*]}" ]]
	then
		echo -e "Error: grepArray and fileArray are different:"
		echo -e "\tgrep_array: ${grep_array[*]}"
		echo -e "\tsfiles: ${sfiles[*]}"
		exit 1
	fi
fi

tmp=$( ${grepcommand} | awk -F: '{ print $1 }' )
IFS=$'\n' file_array=( ${tmp} ) ; unset IFS
tmp=$( ${grepcommand} | awk -F: '{ print $2 }' )
IFS=$'\n' start_array=( ${tmp} ) ; unset IFS
echo "file_array: ${file_array[*]}"
echo "start_array: ${start_array[*]}"

flag="0"
tmp_file_array=()
for i in "${!file_array[@]}"
do
	let "end = ${start_array[$i]} + ${chunklen}"
	sedcommand="sed -n ${start_array[$i]},${end}p ${file_array[$i]}"
	#echo "sedcommand: ${sedcommand}"

	echo -e "\n--------------------------------------\n${file_array[$i]}\n--------------------------------------\n"
	#${sedcommand}
	filename="tmpfile_$i.txt"
	${sedcommand} > ${filename}
	tmp_file_array+=( "${filename}" )

	if [[ "${i}" -gt "0" ]]
	then
		let "j = $i - 1"
		echo "diff between ${tmp_file_array[$j]} and ${tmp_file_array[$i]}"
		diffcommand="diff ${tmp_file_array[$j]} ${tmp_file_array[$i]}"
		${diffcommand}
		lineindiffout=$( ${diffcommand} | wc -l )
		if [[ "${lineindiffout}" != "0" ]]
		then
			flag="1"
		fi
	fi
done

echo -e "\n\n=========================================================================="
if [[ "${flag}" == "0" ]]
then
	echo "All the given file have tha same ${chunklen}-line chuncks"
	echo "Files:"
	for i in "${!file_array[@]}"
	do
		echo -e "\t${file_array[$i]}\t(start at line ${start_array[$i]})"
	done
	echo -e "\nChunk:"
	${sedcommand}
else
	echo "The files' chunks differs"
fi

rm tmpfile_*.txt
