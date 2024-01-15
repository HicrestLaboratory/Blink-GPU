#!/bin/bash

sbatch sbatch/marzola/run-marzola.sh
sbatch sbatch/marzola/run-marzola-nccl.sh
sbatch sbatch/marzola/run-marzola-nvlink.sh
sbatch sbatch/marzola/run-marzola-cuda-aware.sh
