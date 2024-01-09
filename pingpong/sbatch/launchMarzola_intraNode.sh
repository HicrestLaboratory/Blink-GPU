#!/bin/bash

#SBATCH --job-name=PingPongIntraNode
#SBATCH --output=sout/PingPongIntraNode_%j.out
#SBATCH --error=sout/PingPongIntraNode_%j.err

#SBATCH --partition=short
#SBATCH --time=00:05:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1

mainfolder="/home/lorenzo.pichetti/MPI_GPU_banch/pingpong"
binfolder="${mainfolder}/bin"
outfolder="${mainfolder}/out"
explayout="intraNode"

echo " ------ PingPong ------ "
echo "     myfolder: $mainfolder"
echo "     binfolder: $binfolder"
echo "     outfolder: $outfolder"
echo "     explayout: $explayout"

for binaryPath in "${binfolder}"/*
do
	binaryFile=$(echo "$binaryPath" | sed 's|.*/||')
	echo "binaryFile --> $binaryFile"
	start_time="$(date -u +%s)"
	exitcode=$(srun "${binaryPath}" > "${outfolder}/${binaryFile}_${explayout}.out" 2> "${outfolder}/${binaryFile}_${explayout}.err")
	end_time="$(date -u +%s)"

	echo " exitcode:    $exitcode"
	echo " start_time:  $start_time"
	echo " end_time:    $end_time"
done

echo "------------------------"
