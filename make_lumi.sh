#! /bin/bash
module pruge
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

mkdir sout
mkdir bin

names=("mpp" "pp" "a2a" "ar" "otom")
types=("Baseline" "CudaAware" "Nccl" "Nvlink")

#names=("mpp" "pp" "a2a")
#types=("Nvlink")

for name in "${names[@]}"
do
    for type in "${types[@]}"
    do
        # Combinations to skip
        if [[ $name == "otom" && $type != "Nccl" ]]; then
            continue
        fi
        if [[ $name == "ar" && $type == "Nvlink" ]]; then
            continue
        fi        
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
        sed -i 's/hipDeviceProp/hipDeviceProp_t/g' src/${cur_name}.cpp
        sed -i 's/hipFreeHost/hipHostFree/g' src/${cur_name}.cpp # hipFreeHost deprecated
        sed -i 's/!prop.unifiedAddressing/0/g' src/${cur_name}.cpp # We need this because on old ROCM versions the unifiedAddressing property is not available
        #if [[ $type == "Nvlink" ]]; then
        #    sed -i 's/hipMalloc/hipMallocManaged/g' src/${cur_name}.cpp
        #fi
        
        CC -xhip -DHIP -DOPEN_MPI -DPINNED -lrccl -O3 src/${cur_name}.cpp -o bin/${cur_name}  #

        # rm src/${cur_name}.cpp
    done
done
