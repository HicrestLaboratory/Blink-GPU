for f in sout/*
do 
	ext=$( echo ${f} | awk -F. '{ print $2 }' )
	nam=$( basename -- ${f} | awk -F. '{ print $1 }' )
	jid=$( echo ${nam} | awk -F_ '{ print $5 }' )
	tmp=${nam%"_$jid"}

	echo -e "\n-----------------------------------------------------\n"
	echo "filename: ${nam}"
	echo "tmp: ${tmp}"
	echo "ext: ${ext}"
	echo "jobid: ${jid}"


	partition=$( sacct -j ${jid} | head -3 | tail -1 | awk '{ print $3 }' | tr "_" "-" )
	echo "partition: ${partition}"
	newname="${tmp}_${partition}_${jid}.${ext}"
	
	mkdir -p out/

	slurmInfoFile="sout/slurmInfo/slurmInfo_snellius_${jid}.${ext}"
	if [[ -f "${slurmInfoFile}" ]]
	then
		echo "Found ${slurmInfoFile}"
		cat "${slurmInfoFile}" > "out/${newname}"
		echo "${slurmInfoFile} --> ${newname}"
		cat ${f} >> "out/${newname}"
		echo "${f} --> ${newname}"
	else
		echo "${f} --> ${newname}"
		cp ${f} "out/${newname}"
	fi
done
