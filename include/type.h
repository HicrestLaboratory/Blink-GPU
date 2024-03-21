#pragma once

#include "mpi.h"
#include <unistd.h>

#ifdef HIP
#include <rccl.h>
#else
#include <nccl.h>
#endif

#define dtype u_int8_t
#define MPI_dtype MPI_UINT8_T 

#define cktype int32_t
#define MPI_cktype MPI_INT32_T

#define ncclDtype ncclChar // ncclUint8

#define SZTYPE uint64_t

#ifndef MICROBENCH_MAX_GPUS
#define MICROBENCH_MAX_GPUS 16
#endif
