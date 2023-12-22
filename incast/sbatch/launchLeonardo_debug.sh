#!/bin/bash

#SBATCH --job-name=Incast
#SBATCH --output=sout/Incast_%j.out
#SBATCH --error=sout/Incast_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --time=00:05:00
#SBATCH --qos=boost_qos_dbg

#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

mainfolder=/leonardo/home/userexternal/lpichett/MPI_GPU_banch/incast
binfolder=${mainfolder}/bin
outfolder=${mainfolder}/out

echo " ------ Incast ------ "
echo "     myfolder: $mainfolder"
echo "     binfolder: $binfolder"

for binaryPath in "${binfolder}"/*
do
	binaryFile=$(echo "$binaryPath" | sed 's|.*/||')
	echo "binaryFile --> $binaryFile"
	start_time="$(date -u +%s)"
	exitcode=$(srun "${binaryPath}" -iterations 5 > "${outfolder}/${binaryFile}.out" 2> "${outfolder}/${binaryFile}.err")
	end_time="$(date -u +%s)"

	echo " exitcode:    $exitcode"
	echo " start_time:  $start_time"
	echo " end_time:    $end_time"
done

echo "------------------------"
