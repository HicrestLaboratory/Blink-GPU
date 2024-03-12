#pragma once

#include "mpi.h"
#include <nccl.h>

#define dtype u_int8_t
#define MPI_dtype MPI_CHAR // MPI_UINT8_T?

#define cktype int32_t
#define MPI_cktype MPI_INT // MPI_INT32_T?

#define ncclDtype ncclChar // ncclUint8