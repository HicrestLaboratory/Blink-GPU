#!/bin/bash

#SBATCH --job-name=a2aNccl
#SBATCH --output=sout/leonardo_a2a_Nccl_%j.out
#SBATCH --error=sout/leonardo_a2a_Nccl_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:05:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

MODULE_PATH="moduleload/load_nccl_modules.sh"

mkdir -p sout
# export NCCL_SHM_DISABLE=1
source ${MODULE_PATH} && export NCCL_P2P_LEVEL=NVL && srun bin/a2a-nccl