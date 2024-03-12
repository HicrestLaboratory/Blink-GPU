#! /bin/bash
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

names=("mpp" "a2a" "ar")
types=("Baseline" "CudaAware" "Nccl") # "Nvlink")

for name in "${names[@]}"
do
    for type in "${types[@]}"
    do
        cur_name=${name}_${type}
        cp src/${cur_name}.cu src/${cur_name}.cpp
        sed -i 's/#include <cuda.h>//g' src/${cur_name}.cpp
        sed -i 's/#include "cuda.h"//g' src/${cur_name}.cpp
        sed -i 's/#include <cuda_runtime.h>/#include <hip\/hip_runtime.h>/g' src/${cur_name}.cpp
        sed -i 's/#include "cuda_runtime.h"/#include <hip\/hip_runtime.h>/g' src/${cur_name}.cpp
        sed -i 's/#include "mpi-ext.h"//g' src/${cur_name}.cpp
        sed -i 's/cuda/hip/g' src/${cur_name}.cpp
        sed -i 's/<nccl.h>/<rccl.h>/g' src/${cur_name}.cpp
        sed -i 's/ncclGetUniqueId(&Id)/ncclSuccess/g' src/${cur_name}.cpp

        CC -xhip -DHIP -DOPEN_MPI -lrccl src/${cur_name}.cpp -o bin/${cur_name}  #-DPINNED 

        rm src/${cur_name}.cpp
    done
done