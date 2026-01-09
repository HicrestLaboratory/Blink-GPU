#pragma once

#include "common.h"
#include "type.h"
#include "myrand.h"

typedef enum {
    RANDOM_UINT8,
    RANDOM_INT8,
    SIGNEDRANK,
    CONST,
    RANK,
    NONE
} InitStrategy;

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

    void init_host_buff (InitStrategy str, int rank) {
        rng_state += rank;

        // double start = MPI_Wtime();
        switch (str) {
            case RANDOM_UINT8:
                for(SZTYPE i=0; i<(bytes/sizeof(u_int8_t)); i++)
                    ((u_int8_t*)host)[i] = rand() % (UINT8_MAX/2);
                break;

            case RANDOM_INT8:
                // for(SZTYPE i=0; i<(bytes/sizeof(int8_t)); i++)
                //     ((int8_t*)host)[i] = (rand() % 17) - 8;
                rand_int8_array((int8_t*)host, bytes/sizeof(int8_t));
                break;

            case SIGNEDRANK:
                for(SZTYPE i=0; i<(bytes/sizeof(int8_t)); i++)
                    ((int8_t*)host)[i] = ((rank%2)!=0) ? (-1 * (rank+1)) : (rank+1) ;
                break;

            case RANK:
                for(SZTYPE i=0; i<(bytes/sizeof(dtype)); i++)
                    ((dtype*)host)[i] = 1U * (rank+1);
                break;

            case CONST:
                for(SZTYPE i=0; i<(bytes/sizeof(dtype)); i++) ((dtype*)host)[i] = 0;
                break;

            case NONE:
                break;

            default:
                break;
        }
        // double stop = MPI_Wtime();
        // fprintf(stdout, "Array initiated in %lf s\n", stop - start);
    }

    bool alloc(SZTYPE bufferByteLen, int rank, InitStrategy str = RANK) {
        if (allocated) {
            fprintf(stderr, "[%d] Error: set_size on already allocated buff\n", rank);
            return(false);
        }

        srand((unsigned int)time(NULL) + rank);
        bytes = bufferByteLen;

        bool errflag = alloc_host();
        if (!errflag) return(errflag);
        init_host_buff (str, rank);
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



bool tmp_function(size_t bytes, int *mpicount, MPI_Datatype *mpitype, ncclDataType_t *nccltype) {
    (*mpicount) = 0;
    if(bytes >= 8 && bytes % 8 == 0){ // Check if I can use 64-bit data types

        (*mpicount) = bytes / 8;
        (*mpitype)  = MPI_dtype_big;
        (*nccltype) = ncclDtype_big;
        if ((*mpicount) >= ((u_int64_t) (1UL << 32)) - 1) // If large_count can't be represented on 32 bits
            return(false);
    }else{
        (*mpicount) = bytes;
        (*mpitype)  = MPI_dtype;
        (*nccltype) = ncclDtype;
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

    ncclDataType_t sNcclType, rNcclType;

    void init(CommunicatioType type, size_t msgcount, MPI_Comm comm, InitStrategy str = RANK, int root = 0) {

        int rank, commsize_int;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &commsize_int);
        size_t commsize = static_cast<size_t>(commsize_int);

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
                sBuffBytes = (rank == root) ? (commsize * msgcount * sizeof(T)) : 0 ;
                rBuffBytes = msgcount * sizeof(T);
                break;

            case GATHER:
                sBuffBytes = msgcount * sizeof(T);
                rBuffBytes = (rank == root) ? (commsize * msgcount * sizeof(T)) : 0 ;
                break;

            case SENDRECV:
                sBuffBytes = msgcount * sizeof(T);
                rBuffBytes = msgcount * sizeof(T);
                break;

            case BCAST:
                sBuffBytes = (rank == root) ? (msgcount * sizeof(T)) : 0 ;
                rBuffBytes = msgcount * sizeof(T);
                break;

            default:
                sBuffBytes = 0;
                rBuffBytes = 0;
                break;
        }

        tmp_function(msgcount, &sMpicount, &sMpiDtype, &sNcclType);
        tmp_function(msgcount, &rMpicount, &rMpiDtype, &rNcclType);

        bool errorflagsend = sBuff.alloc(sBuffBytes, rank, str);
        if (!errorflagsend) {
            fprintf(stderr, "[%d] Error while allocating buffers at line %d (%lu Bytes requested)\n", rank, __LINE__, sBuff.bytes);
            fflush(stderr);
        }

        bool errorflagrecv = rBuff.alloc(rBuffBytes, rank, NONE);
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

    void print(char bchar, int rank, FILE *fp = stdout) {
        if ((bchar != 's') && (bchar != 'r')) {
            fprintf(stderr, "[%d] Error: unsupported bchar %c\n", rank, bchar);
            return;
        }

        size_t  len = (bchar=='s') ? sBuffBytes : rBuffBytes ; len /= sizeof(T);
        int8_t *buf = (bchar=='s') ? (int8_t*)sBuff.host : (int8_t*)rBuff.host ;

        char *s = (char*)malloc( (len * 6 + 64)*sizeof(char) );
        sprintf(s, "[%d] %cBuff (%zu): ", rank, bchar, len);
        for (size_t i=0; i<len; i++)
            sprintf(s+strlen(s), "%d ", buf[i]);
        sprintf(s+strlen(s), "\n");

        fprintf(fp, "%s", s);
        free(s);
    }

    void sendBuff_reduction(cktype *sReduction) {
        if (sBuffBytes != 0)
            gpu_device_reduce(((dtype*)sBuff.device), sBuffBytes / sizeof(T), sReduction);
        else
            sReduction = 0;
    }

    void recvBuff_reduction(cktype *rReduction) {
        if (rBuffBytes != 0)
            gpu_device_reduce(((dtype*)rBuff.device), rBuffBytes / sizeof(T), rReduction);
        else
            rReduction = 0;
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
