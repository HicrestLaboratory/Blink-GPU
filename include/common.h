#pragma once

#include "type.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include "gpu_ops.h"
#include <inttypes.h>

void alloc_host_buffers(int rank,
                        dtype **sendBuffer, SZTYPE sendBufferLen,
                        dtype **recvBuffer, SZTYPE recvBufferLen) {
#ifdef PINNED
        cudaHostAlloc(sendBuffer, sendBufferLen*sizeof(dtype), cudaHostAllocDefault);
        cudaHostAlloc(recvBuffer, recvBufferLen*sizeof(dtype), cudaHostAllocDefault);
#else
        *sendBuffer = (dtype*)malloc(sendBufferLen*sizeof(dtype));
        *recvBuffer = (dtype*)malloc(recvBufferLen*sizeof(dtype));
#endif
        int errorflag = 0;
        if (*sendBuffer == NULL) {
            fprintf(stderr, "[%d] Error while allocating buffers at line %d (%lu Bytes requested)\n", rank, __LINE__, sendBufferLen*sizeof(dtype));
            fflush(stderr);
            errorflag = __LINE__;

        }
        if (*recvBuffer == NULL) {
            fprintf(stderr, "[%d] Error while allocating buffers at line %d (%lu Bytes requested)\n", rank, __LINE__, recvBufferLen*sizeof(dtype));
            fflush(stderr);
            errorflag = __LINE__;

        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (errorflag != 0) MPI_Abort(MPI_COMM_WORLD, errorflag);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("Buffers of size %" PRIu64 " B and %" PRIu64 " B succesfuly allocated by all ranks\n", sendBufferLen*sizeof(dtype), recvBufferLen*sizeof(dtype));
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
}

/* Example:
 * Let buff be a buffer of size n to be initialized with value -1
 * INIT_HOST_BUFFER(buff, n, -1)
 */
#define INIT_HOST_BUFFER(B, L, V) { \
    for (SZTYPE i=0; i<L; i++) {    \
        (B)[i] = V;                 \
    }                               \
}

void alloc_device_buffers(dtype *sendBuffer, dtype **dev_sendBuffer, SZTYPE sendBufferLen,
                          dtype *recvBuffer, dtype **dev_recvBuffer, SZTYPE recvBufferLen) {

    cudaErrorCheck( cudaMalloc(dev_sendBuffer, sendBufferLen*sizeof(dtype)) );
    cudaErrorCheck( cudaMemcpy(*dev_sendBuffer, sendBuffer, sendBufferLen*sizeof(dtype), cudaMemcpyHostToDevice) );

    cudaErrorCheck( cudaMalloc(dev_recvBuffer, recvBufferLen*sizeof(dtype)) );
    cudaErrorCheck( cudaMemcpy(*dev_recvBuffer, recvBuffer, recvBufferLen*sizeof(dtype), cudaMemcpyHostToDevice) );
}

cktype* share_local_checks(int mpi_size, dtype *local_buffer, SZTYPE buffer_len) {

    cktype *local_check = (cktype*)malloc(sizeof(cktype));
    cktype *all_checks  = (cktype*)malloc(sizeof(cktype)*mpi_size);

    *local_check = 0U;
    gpu_device_reduce_max(local_buffer, buffer_len, local_check);
    MPI_Allgather(local_check, 1, MPI_cktype, all_checks, 1, MPI_cktype, MPI_COMM_WORLD);

    return(all_checks);
}

void compute_global_checks(int mpi_size, cktype *all_checks,
                           dtype *dev_recvBuffer, SZTYPE recvBufferLen,
                           cktype *check_on_send, cktype *check_on_recv) {

    cktype gpu_check = 0;
    gpu_device_reduce_max(dev_recvBuffer, recvBufferLen, &gpu_check);
    *check_on_recv = gpu_check;

    cktype tmp = 0;
    for (int i=0; i<mpi_size; i++)
        if (tmp < all_checks[i]) tmp = all_checks[i];
    *check_on_send = tmp;
}
