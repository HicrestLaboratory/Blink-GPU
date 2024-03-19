#! /bin/bash
names=("mpp" "a2a" "ar")
types=("Baseline" "CudaAware" "Nccl") # "Nvlink")

rm src/*.cpp

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
        sed -i 's/hipHostAlloc(/hipHostMalloc((void **)/g' src/${cur_name}.cpp
        sed -i 's/hipHostAllocDefault/hipHostMallocDefault/g' src/${cur_name}.cpp
    done
done
