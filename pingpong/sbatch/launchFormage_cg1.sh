#!/bin/bash

#SBATCH --job-name=PingPong
#SBATCH --output=sout/PingPong_%j.out
#SBATCH --error=sout/PingPong_%j.err

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --partition=cg1-cpu480gb-gpu96gb
#SBATCH --exclusive

# module load cuda/
# module load nccl/
# module load openmpi/

mainfolder=/home/lpichetti/MPI_GPU_banch/pingpong
binfolder=${mainfolder}/bin
outfolder=${mainfolder}/out

echo " ------ PingPong ------ "
echo "     myfolder: $mainfolder"
echo "     binfolder: $binfolder"

for binaryPath in ${binfolder}/*
do
	binaryFile=$(echo "$binaryPath" | sed 's|.*/||')
	echo "binaryFile= --> $binaryFile"
	start_time="$(date -u +%s)"
	exitcode=$(mpirun -np 2 ${binaryPath} > ${outfolder}/${binaryFile}.out 2> ${outfolder}/${binaryFile}.err)
	end_time="$(date -u +%s)"

	echo " exitcode:    $exitcode"
	echo " start_time:  $start_time"
	echo " end_time:    $end_time"
done

echo "------------------------"
