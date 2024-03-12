#pragma once


#include <unistd.h>

#ifdef HIP
#include <hip/hip_runtime.h>
#include <rccl.h>
#else
#include "cuda.h"
#include <cuda_runtime.h>
#include <nccl.h>
#endif

// Macro for checking errors in INTERFACE API calls
#define hipErrorCheck(call)                                                              \
do{                                                                                       \
    hipError_t cuErr = call;                                                             \
    if(hipSuccess != cuErr){                                                             \
        printf("hip Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(cuErr));\
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
