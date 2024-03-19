#!/bin/bash

#SBATCH --job-name=a2a_Nvlink_1-8-gpu-5-6
#SBATCH --output=sout/lumi_a2a_Nvlink_1-8-gpu-5-6_%j.out
#SBATCH --error=sout/lumi_a2a_Nvlink_1-8-gpu-5-6_%j.err

#SBATCH --partition=ju-standard-g
#SBATCH --time=00:10:00

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=2
#SBATCH --account=project_465000997

CPU_BIND="map_cpu:9,25,41,57,1,17,33,49"

mkdir -p sout


module purge
module load LUMI/23.09
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
export USER_HIP_GPU_MAP=5,6
srun --cpu-bind=${CPU_BIND} bin/a2a_Nvlink

