#!/bin/bash

#SBATCH --job-name=testNcclMarzola
#SBATCH --output=sout/testNcclMarzola_%j.out
#SBATCH --error=sout/testNcclMarzola_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=short

MODULE_PATH="moduleload/load_nccl_modules.sh"

mkdir -p sout
source ${MODULE_PATH} && srun bin/test-nccl
