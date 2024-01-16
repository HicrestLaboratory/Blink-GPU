#!/bin/bash

sbatch sbatch/leonardo/run-leonardo-a2a.sh
sbatch sbatch/leonardo/run-leonardo-a2a-nccl.sh
sbatch sbatch/leonardo/run-leonardo-a2a-nvlink.sh
sbatch sbatch/leonardo/run-leonardo-a2a-cuda-aware.sh
sbatch sbatch/leonardo/run-leonardo-a2a-internode.sh
sbatch sbatch/leonardo/run-leonardo-a2a-internode-nccl.sh
sbatch sbatch/leonardo/run-leonardo-a2a-internode-cuda-aware.sh
