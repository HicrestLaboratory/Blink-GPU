#!/bin/bash

#SBATCH --job-name=testMarzola
#SBATCH --output=sout/testMarzola_%j.out
#SBATCH --error=sout/testMarzola_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=short

mkdir -p sout
srun bin/test
