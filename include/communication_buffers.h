#pragma once

#include "common.h"
#include "type.h"

struct BufferHolder {
    void   *host;
    SZTYPE  bytes;
    void   *device;
    bool allocated = false;


    bool alloc_host(void) {

#ifdef PINNED
        cudaHostAlloc(host, bytes, cudaHostAllocDefault);
#else
        host = (dtype*)malloc(bytes);
#endif
        if (host == NULL) return(false);
        return(true);
    }

    void alloc_device_buffers(void) {
        cudaErrorCheck( cudaMalloc(&device, bytes) );
        cudaErrorCheck( cudaMemcpy( device, host, bytes, cudaMemcpyHostToDevice) );
    }

    bool alloc(SZTYPE bufferByteLen) {
        if (allocated) {
            fprintf(stderr, "Error: set_size on already allocated buff\n");
            return(false);
        }

        bytes = bufferByteLen;

        bool errflag = alloc_host();
        if (!errflag) return(errflag);
        alloc_device_buffers();
        allocated = true;
        return(true);
    }

    void explicitfree() {
    if (host) {
    #ifdef PINNED
            cudaFreeHost(host);
    #else
            free(host);
    #endif
            host = nullptr;
        }

        if (device) {
            cudaErrorCheck(cudaFree(device));
            device = nullptr;
        }

        bytes     = 0;
        allocated = false;
    }

};



typedef enum {
    ALL2ALL,
    ALLREDUCE,
    ALLGATHER,
    SCATTER,
    INCAST,
    SENDRECV
} CommunicatioType;

bool tmp_function(size_t bytes, int *mpicount, MPI_Datatype *mpitype) {
    (*mpicount) = 0;
    if(bytes >= 8 && bytes % 8 == 0){ // Check if I can use 64-bit data types

        (*mpicount) = bytes / 8;
        (*mpitype)  = MPI_dtype_big;
        if ((*mpicount) >= ((u_int64_t) (1UL << 32)) - 1) // If large_count can't be represented on 32 bits
            return(false);
    }else{
        (*mpicount) = bytes;
        (*mpitype)  = MPI_dtype;
        if (bytes >= ((u_int64_t) (1UL << 32)) - 1) // If N can't be represented on 32 bits
            return(false);
    }
    return(true);
}

template <typename T>
struct CommunicationBuffers {
    BufferHolder sBuff;
    BufferHolder rBuff;

    size_t sBuffBytes;
    size_t rBuffBytes;

    int sMpicount, rMpicount;
    MPI_Datatype sMpiDtype, rMpiDtype;

    void init(CommunicatioType type, int msgcount, MPI_Comm comm) {

        int rank, commsize;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &commsize);

        switch (type) {
            case ALL2ALL:
                sBuffBytes = commsize * msgcount * sizeof(T);
                rBuffBytes = commsize * msgcount * sizeof(T);
                break;

            case ALLREDUCE:
                sBuffBytes = msgcount * sizeof(T);
                rBuffBytes = msgcount * sizeof(T);
                break;

            case ALLGATHER:
                sBuffBytes = msgcount * sizeof(T);
                rBuffBytes = commsize * msgcount * sizeof(T);
                break;

            case SCATTER:
                sBuffBytes = commsize * msgcount * sizeof(T); // BUG this should be just the root
                rBuffBytes = msgcount * sizeof(T);
                break;

            case INCAST:
                sBuffBytes = msgcount * sizeof(T);
                rBuffBytes = commsize * msgcount * sizeof(T); // BUG this should be just the root
                break;

            case SENDRECV:
                sBuffBytes = msgcount * sizeof(T);
                rBuffBytes = msgcount * sizeof(T);
                break;

            default:
                sBuffBytes = 0;
                rBuffBytes = 0;
                break;
        }

        tmp_function(msgcount, &sMpicount, &sMpiDtype);
        tmp_function(msgcount, &rMpicount, &rMpiDtype);

        bool errorflagsend = sBuff.alloc(sBuffBytes);
        if (!errorflagsend) {
            fprintf(stderr, "[%d] Error while allocating buffers at line %d (%lu Bytes requested)\n", rank, __LINE__, sBuff.bytes);
            fflush(stderr);
        }

        bool errorflagrecv = rBuff.alloc(rBuffBytes);
        if (!errorflagrecv) {
            fprintf(stderr, "[%d] Error while allocating buffers at line %d (%lu Bytes requested)\n", rank, __LINE__, rBuff.bytes);
            fflush(stderr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if ((!errorflagsend)||(!errorflagrecv)) MPI_Abort(MPI_COMM_WORLD, __LINE__);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) printf("Buffers of size %" PRIu64 " B and %" PRIu64 " B succesfuly allocated by all ranks\n", sBuff.bytes, rBuff.bytes);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void clear(int rank) {
//         fprintf(stdout, "[%d] Freeing buffers: \
// A -> alloc=%d, mpicount=%d, bytes=%lu, host=%p, device=%p, \
// B -> alloc=%d, mpicount=%d, bytes=%lu, host=%p, device=%p\n",
//             rank,
//             sendBuff.allocated, sendMpicount, sendBuff.bytes, sendBuff.host, sendBuff.device,
//             recvBuff.allocated, recvMpicount, recvBuff.bytes, recvBuff.host, recvBuff.device);
        sBuff.explicitfree();
        rBuff.explicitfree();
        sMpicount    = 0;
        rMpicount    = 0;
        sBuffBytes = 0;
        rBuffBytes = 0;
        sMpiDtype = MPI_BYTE;
        rMpiDtype = MPI_BYTE;
        // fprintf(stdout, "[%d] Freed buffers: A -> alloc=%d, host=%p, device=%p, B -> alloc=%d, host=%p, device=%p\n",
        //     rank, sendBuff.allocated, sendBuff.host, sendBuff.device, recvBuff.allocated, recvBuff.host, recvBuff.device);
    }
};
