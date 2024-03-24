#! /bin/bash
names=("mpp" "pp" "a2a" "a2am" "ar" "otom" "inc")
types=("Baseline" "CudaAware" "Nccl" "Nvlink")

#names=("mpp" "pp" "a2a")
#types=("Nvlink")

for name in "${names[@]}"
do
    for type in "${types[@]}"
    do
        # Combinations to skip
        # otom only available for Nccl and Nvlink
        if [[ $name == "otom" && $type == "Baseline" ]]; then
            continue
        fi
        if [[ $name == "otom" && $type == "CudaAware" ]]; then
            continue
        fi
        # inc only available for Nccl
        if [[ $name == "inc" && $type != "Nccl" ]]; then
            continue
        fi
        # a2am only available for Nccl
        if [[ $name == "a2am" && $type != "Nccl" ]]; then
            continue
        fi
        # ar not available for Nvlink
        if [[ $name == "ar" && $type == "Nvlink" ]]; then
            continue
        fi        
        # a2a on Nccl does not need to be generated
        if [[ $name == "a2a" && $type == "Nccl" ]]; then
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
    done
done