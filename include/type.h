#pragma once

#include "mpi.h"
#include <unistd.h>

#ifdef HIP
#include <rccl.h>
#else
#include <nccl.h>
#endif

#define dtype u_int8_t
#define dtype_big u_int64_t

#define MPI_dtype MPI_UINT8_T 
#define MPI_dtype_big MPI_UINT64_T 

#define ncclDtype ncclChar // ncclUint8
#define ncclDtype_big ncclUint64

#define cktype int32_t
#define MPI_cktype MPI_INT32_T

#define SZTYPE uint64_t

#ifndef MICROBENCH_MAX_GPUS
#define MICROBENCH_MAX_GPUS 16
#endif
