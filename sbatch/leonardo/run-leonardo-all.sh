#!/bin/bash

sbatch sbatch/leonardo/run-leonardo.sh
sbatch sbatch/leonardo/run-leonardo-nccl.sh
sbatch sbatch/leonardo/run-leonardo-nvlink.sh
sbatch sbatch/leonardo/run-leonardo-cuda-aware.sh
sbatch sbatch/leonardo/run-leonardo-internode.sh
sbatch sbatch/leonardo/run-leonardo-internode-nccl.sh
sbatch sbatch/leonardo/run-leonardo-internode-cuda-aware.sh
