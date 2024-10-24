if [[ "$#" -lt "2" ]]
then
	echo "wrong input"
	exit 1
elif [[ "$#" -gt "2" ]]
then
	staticparams_lables=()
	staticparams_values=()
	for e in $@
	do
		if [[ "$e" != "$1" ]] && [[ "$e" != "$2" ]]
		then
			if [[ "$( echo "${e}" | tr -dc ':' )" != ":" ]]
			then
				echo "Error on element ${e}: wrong number of occurrences of ':' char"
				exit 1
			else
				staticlable="$( echo ${e} | awk -F: '{ print $1 }' )"
				staticvalue="$( echo ${e} | awk -F: '{ print $2 }' )"

				staticparams_lables+=( "${staticlable}" )
				staticparams_values+=( "${staticvalue}" )
			fi
		fi
	done

	# --- DEBUG ---
	echo "${#staticparams_lables[@]} static lables:"
	for i in ${!staticparams_lables[@]}
	do
		echo -e "\t${staticparams_lables[$i]}: ${staticparams_values[$i]}"
	done
fi

inputtxt=${1}
outcsv=${2}

string=""
for i in ${!staticparams_lables[@]}
do
	string+="${staticparams_lables[$i]}, "
done
string+="TransferSize(B), TransferTime(s), Bandwidth(GiB/s)"
echo "string: ${string}"
echo "${string}" > ${outcsv}

while IFS= read -r line
do
	#echo "line: ${line}"

	el1=$( echo ${line} | awk -F, '{ print $1 }' | awk -F: '{ print $2 }' )
	el2=$( echo ${line} | awk -F, '{ print $2 }' | awk -F: '{ print $2 }' )
	el3=$( echo ${line} | awk -F, '{ print $3 }' | awk -F: '{ print $2 }' )

	string=""
	for i in ${!staticparams_values[@]}
	do
		string+="${staticparams_values[$i]}, "
	done
	string+="${el1}, ${el2}, ${el3}"
	#echo "string: ${string}"
	echo "${string}" >> ${outcsv}
done < "$inputtxt"

