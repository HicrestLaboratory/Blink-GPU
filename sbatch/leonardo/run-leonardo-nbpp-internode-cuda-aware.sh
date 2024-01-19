#!/bin/bash

#SBATCH --job-name=nbppInternodeCudaAware
#SBATCH --output=sout/leonardo_nbpp_InternodeCudaAware_%j.out
#SBATCH --error=sout/leonardo_nbpp_InternodeCudaAware_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:05:00

#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

MODULE_PATH="moduleload/load_cuda_aware_modules.sh"

mkdir -p sout
source ${MODULE_PATH} && srun bin/nbpp-cuda-aware
