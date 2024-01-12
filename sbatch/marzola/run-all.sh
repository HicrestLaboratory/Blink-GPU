#!/bin/bash

sbatch sbatch/marzola/run.sh
sbatch sbatch/marzola/run-nccl.sh
sbatch sbatch/marzola/run-nvlink.sh
sbatch sbatch/marzola/run-cuda-aware.sh
