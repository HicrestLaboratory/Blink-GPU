#!/bin/bash

#SBATCH --job-name=Halo3DInterNodes
#SBATCH --output=sout/Halo3DInterNodes_%j.out
#SBATCH --error=sout/Halo3DInterNodes_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --time=00:05:00
#SBATCH --qos=boost_qos_dbg

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

mainfolder=/leonardo/home/userexternal/lpichett/MPI_GPU_banch/halo3d
binfolder=${mainfolder}/bin
outfolder=${mainfolder}/out
explayout="intraNode"

echo " ------ Halo3D ------ "
echo "     myfolder: $mainfolder"
echo "     binfolder: $binfolder"
echo "     outfolder: $outfolder"
echo "     explayout: $explayout"

for binaryPath in "${binfolder}"/*
do
	binaryFile=$(echo "$binaryPath" | sed 's|.*/||')
	echo "binaryFile --> $binaryFile"
	start_time="$(date -u +%s)"
	exitcode=$(srun "${binaryPath}" -pex 2 -pey 2 -pez 1 -nx 16 -ny 16 -nz 16 -vars 4 > "${outfolder}/${binaryFile}_${explayout}.out" 2> "${outfolder}/${binaryFile}_${explayout}.err")
	end_time="$(date -u +%s)"

	echo " exitcode:    $exitcode"
	echo " start_time:  $start_time"
	echo " end_time:    $end_time"
done

echo "------------------------"
