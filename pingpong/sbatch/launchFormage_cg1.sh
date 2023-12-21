#!/bin/bash

#SBATCH --job-name=PingPong
#SBATCH --output=sout/PingPong_%j.out
#SBATCH --error=sout/PingPong_%j.err

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --partition=cg1-cpu480gb-gpu96gb
#SBATCH --exclusive

# Currently Loaded Modulefiles:
#  1) slurm/slurm-01/23.02.6   2) hpcx   3) nvhpc-hpcx-cuda12/23.11   4) openmpi/4.1.5   5) binutils/12.2.0   6) gcc/12.2.0
module unload gcc/13.1.0
module load gcc/12.2.0
module load openmpi/4.1.5
module load nvhpc-hpcx-cuda12/23.11

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
	exitcode=$(mpirun -np 2 ${binaryPath} > ${outfolder}/${binaryFile}_cg1.out 2> ${outfolder}/${binaryFile}_cg1.err)
	end_time="$(date -u +%s)"

	echo " exitcode:    $exitcode"
	echo " start_time:  $start_time"
	echo " end_time:    $end_time"
done

echo "------------------------"
