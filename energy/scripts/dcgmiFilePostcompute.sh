#!/bin/bash
datafolder="./results/"
keyname="dcgmiMesures"
parsedkeyname="parsedDcgmiMesures"

parsefile() {
	file=$1
	grep -v "#" "${file}" | grep -v "ID" | sed "s/GPU //g" | tr -s ' ' | tr ' ' ',' | sed 's/.$//'
}

metavec_ben=()
metavec_imp=()
metavec_buf=()
metavec_nlo=()
metavec_header=()

# Expected structure: <datafolder>/<keyname>_<benchmark>_<implementation>_<rankid>.csv
for f in ${datafolder}${keyname}_*
do 
	filename=$( basename -- $f | cut -d. -f1 )
	benchmark=$( echo "$filename" | awk -F_ '{ print $2 }' )
	implementation=$( echo "$filename" | awk -F_ '{ print $3 }')
	buffsize=$( echo "$filename" | awk -F_ '{ print $4 }')
	nloops=$( echo "$filename" | awk -F_ '{ print $5 }')
	header=$( grep "#" ${f} | head -1 )
	
	echo "$f --> benchmark: ${benchmark}, implementation: ${implementation}, buffsize: ${buffsize}, nloops: ${nloops}"
	metavec_ben+=( "${benchmark}" )
	metavec_imp+=( "${implementation}" )
	metavec_buf+=( "${buffsize}" )
	metavec_nlo+=( "${nloops}" )
	metavec_header+=( "${header}" )

	parsedfilename="${parsedkeyname}_${benchmark}_${implementation}_${buffsize}_${nloops}.csv"
	echo "${header}" | tr -s ' ' | tr ' ' ',' | sed 's/.$//' > "${parsedfilename}"
	parsefile "${f}" >> "${parsedfilename}"
done

echo "metavec_ben: ${metavec_ben[*]}"
echo "metavec_imp: ${metavec_imp[*]}"
echo "metavec_buf: ${metavec_buf[*]}"
echo "metavec_nlo: ${metavec_nlo[*]}"
echo "metavec_header: ${metavec_header[*]}"
