#!/bin/bash

#SBATCH --job-name=nbpp
#SBATCH --output=sout/leonardo_nbpp_Baseline_%j.out
#SBATCH --error=sout/leonardo_nbpp_Baseline_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:05:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=494000MB

MODULE_PATH="moduleload/load_baseline_modules.sh"

mkdir -p sout
source ${MODULE_PATH} && srun bin/nbpp
