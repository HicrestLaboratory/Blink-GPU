#!/bin/bash

sbatch sbatch/leonardo/run.sh
sbatch sbatch/leonardo/run-nccl.sh
sbatch sbatch/leonardo/run-nvlink.sh
sbatch sbatch/leonardo/run-cuda-aware.sh
sbatch sbatch/leonardo/run-internode.sh
sbatch sbatch/leonardo/run-internode-nccl.sh
sbatch sbatch/leonardo/run-internode-cuda-aware.sh
