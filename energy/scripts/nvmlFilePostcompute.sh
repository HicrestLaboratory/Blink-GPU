#!/bin/bash
datafolder="./results/"
keyname="nvmlMesures"
resultfolder="./parsed/"
parsedkeyname="parsedNvmlMesures"

parsefile() {
	file=$1
	grep -v "#" "${file}" | tr -s ' ' | tr ' ' ','
}

metavec_ben=()
metavec_imp=()
metavec_buf=()
metavec_nlo=()
metavec_devid=()
metavec_header=()

mkdir -p ${resultfolder}

# Expected structure: <datafolder>/<keyname>_<benchmark>_<implementation>_<rankid>.csv
for f in ${datafolder}${keyname}_*
do 
	filename=$( basename -- $f | cut -d. -f1 )
	benchmark=$( echo "$filename" | awk -F_ '{ print $2 }' )
	implementation=$( echo "$filename" | awk -F_ '{ print $3 }')
	buffsize=$( echo "$filename" | awk -F_ '{ print $4 }')
	nloops=$( echo "$filename" | awk -F_ '{ print $5 }')
	devid=$( echo "$filename" | awk -F_ '{ print $6 }')
	header=$( grep "#" ${f} | head -1 )
	
	echo "$f --> benchmark: ${benchmark}, implementation: ${implementation}, buffsize: ${buffsize}, nloops: ${nloops}, devid: ${devid}"
	metavec_ben+=( "${benchmark}" )
	metavec_imp+=( "${implementation}" )
	metavec_buf+=( "${buffsize}" )
	metavec_nlo+=( "${nloops}" )
	metavec_devid+=( "${devid}" )
	metavec_header+=( "${header}" )

	parsedfilename="${parsedkeyname}_${benchmark}_${implementation}_${buffsize}_${nloops}.csv"
	
	if [[ "${devid}" == "0" ]]
	then
		echo "${header}" | tr -s ' ' | tr ' ' ',' > "${resultfolder}${parsedfilename}"
	fi
	parsefile "${f}" >> "${resultfolder}${parsedfilename}"
done

echo "metavec_ben: ${metavec_ben[*]}"
echo "metavec_imp: ${metavec_imp[*]}"
echo "metavec_buf: ${metavec_buf[*]}"
echo "metavec_nlo: ${metavec_nlo[*]}"
echo "metavec_devid: ${metavec_devid[*]}"
echo "metavec_header: ${metavec_header[*]}"
