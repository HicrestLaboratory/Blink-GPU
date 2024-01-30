#!/bin/bash

sbatch sbatch/leonardo/run-leonardo-pp.sh
sbatch sbatch/leonardo/run-leonardo-pp-nccl.sh
sbatch sbatch/leonardo/run-leonardo-pp-nvlink.sh
sbatch sbatch/leonardo/run-leonardo-pp-cuda-aware.sh
sbatch sbatch/leonardo/run-leonardo-pp-internode.sh
sbatch sbatch/leonardo/run-leonardo-pp-internode-nccl.sh
sbatch sbatch/leonardo/run-leonardo-pp-internode-cuda-aware.sh
