#pragma once

#include "mpi.h"
#include <nccl.h>
#include <unistd.h>

#define dtype u_int8_t
#define MPI_dtype MPI_UINT8_T  

#define cktype int32_t
#define MPI_cktype MPI_INT32_T

#define ncclDtype ncclChar // ncclUint8

#define SZTYPE uint64_t

#define MAX_GPUS 4
