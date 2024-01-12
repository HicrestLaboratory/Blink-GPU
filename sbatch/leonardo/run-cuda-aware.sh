#!/bin/bash

#SBATCH --job-name=testCudaAware
#SBATCH --output=sout/testCudaAware_%j.out
#SBATCH --error=sout/testCudaAware_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:05:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=494000MB

mkdir -p sout
srun bin/test-cuda-aware
