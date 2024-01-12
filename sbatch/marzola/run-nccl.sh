#!/bin/bash

#SBATCH --job-name=testNcclMarzola
#SBATCH --output=sout/testNcclMarzola_%j.out
#SBATCH --error=sout/testNcclMarzola_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=short

mkdir -p sout
srun bin/test-nccl
