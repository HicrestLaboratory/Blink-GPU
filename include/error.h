#pragma once

#include <nccl.h>
#include "cuda.h"
#include <cuda_runtime.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
        exit(0);                                                                            \
    }                                                                                     \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
