#!/bin/bash

#SBATCH --job-name=concurrent_hipmemcpy_en_dma_0_6_2
#SBATCH --output=sout/lumi_concurrent_hipmemcpy_en_dma_0_6_2_%j.out
#SBATCH --error=sout/lumi_concurrent_hipmemcpy_en_dma_0_6_2_%j.err

#SBATCH --partition=ju-standard-g
#SBATCH --time=00:10:00

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_465000997

# CPU_BIND="map_cpu:9,25,41,57,1,17,33,49"

mkdir -p sout

export HSA_ENABLE_SDMA=1

module purge
module load LUMI/23.09
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

srun bin/concurrent_hipmemcpy -g0 0 -g1 6 -g2 2
