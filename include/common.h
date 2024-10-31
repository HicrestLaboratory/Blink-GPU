#pragma once

#include "type.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include "gpu_ops.h"
#include <inttypes.h>

#if defined(USE_NVTX)
#include <nvToolsExt.h>

#if !defined(INITIALIZED_NVTX)
#define INITIALIZED_NVTX
const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
#endif

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

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

void compile_time_check(void) {
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
}

void print_hostnames (int rank, int size) {
    
    int namelen;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(host_name, &namelen);
    printf("Size = %d, myrank = %d, host_name = %s\n", size, rank, host_name);
    fflush(stdout);
}

/* --------------------------------------------------------------------------------------------------------------
 * How to use my_mpi_init() function:
 * --------------------------------------------------------------------------------------------------------------
 *
	MPI_Init(&argc, &argv);
	
	MPI_Status stat;
	MPI_Comm nodeComm;
	int num_devices, my_dev;
	int size, nnodes, rank, mynode, mynodeid, mynodesize;

	my_mpi_init(&size, &nnodes, &rank, &mynode, &num_devices, &my_dev, &nodeComm, &mynodeid, &mynodesize);
 *
*/

void my_mpi_init(int *ptr_size, int *ptr_nnodes, int *ptr_rank, int *ptr_mynode, int *ptr_num_devices, int *ptr_my_dev, MPI_Comm *ptr_nodeComm, int *ptr_mynodeid, int *ptr_mynodesize) {

    // Define WORLD ranks and size
    MPI_Comm_size(MPI_COMM_WORLD, ptr_size);
    MPI_Comm_rank(MPI_COMM_WORLD, ptr_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // Print hostnames
    print_hostnames (*ptr_rank, *ptr_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Map MPI ranks to GPUs and define nodeComm
    cudaErrorCheck( cudaGetDeviceCount(ptr_num_devices) );
    *ptr_my_dev = assignDeviceToProcess(ptr_nodeComm, ptr_nnodes, ptr_mynode);
    MPI_Comm_rank(*ptr_nodeComm, ptr_mynodeid);
    MPI_Comm_size(*ptr_nodeComm, ptr_mynodesize);

    // Print device affiniy
#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(dev);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

}

/* --------------------------------------------------------------------------------------------------------------
 * How to use define_pp_comm() function:
 * --------------------------------------------------------------------------------------------------------------
 *
    int rank2 = size-1;
    MPI_Comm ppComm, firstsenderComm;
    define_pp_comm (rank, 0, rank2, &ppComm, &firstsenderComm);

 *
*/

void define_pp_comm (int myrank, int rank1, int rank2, MPI_Comm *ppComm, MPI_Comm *firstsenderComm) {
    //int rank2 = size-1;

    // Get the group or processes of the default communicator
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Define a communicatior with only the two processes who will perform the ping-png
    int pp_ranks[2];
    pp_ranks[0] = rank1;
    pp_ranks[1] = rank2;
    
    MPI_Group pp_group;
    MPI_Group_incl(world_group, 2, pp_ranks, &pp_group);
    MPI_Comm_create(MPI_COMM_WORLD, pp_group, ppComm);

    // Print if you took part or not in the ppComm
    if(*ppComm == MPI_COMM_NULL) {
        // I am not part of the ppComm.
        printf("Process %d did not take part to the ppComm.\n", myrank);
    } else {
        // I am part of the new ppComm.
        printf("Process %d took part to the ppComm.\n", myrank);
    }

    // Define a communicator with only the first sender process.
    int ranks_0[1];
    ranks_0[0] = 0;

    MPI_Group firstsender_group;
    if (myrank == rank1)
        MPI_Group_incl(world_group, 1, ranks_0, &firstsender_group);
    else
        MPI_Group_incl(world_group, 0, ranks_0, &firstsender_group);

    MPI_Comm_create(MPI_COMM_WORLD, firstsender_group, firstsenderComm);

    // Print if you took part or not in the firstsenderComm
    if(*firstsenderComm == MPI_COMM_NULL) {
        printf("Process %d did not take part to the firstsenderComm.\n", myrank);
    } else {
        printf("Process %d took part to the firstsenderComm.\n", myrank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/* --------------------------------------------------------------------------------------------------------------
 * How to use define_buffer_len() function:
 * --------------------------------------------------------------------------------------------------------------
 *
    SZTYPE N = define_buffer_len(fix_buff_size);
 *
*/

SZTYPE define_buffer_len(int fix_buff_size) {

    SZTYPE N;
    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    return(N);
}

/* --------------------------------------------------------------------------------------------------------------
 * How to use timers_and_checks_alloc() function:
 * --------------------------------------------------------------------------------------------------------------
 *
    int *error, *my_error;
    TTYPE start_time, stop_time;
    cktype *cpu_checks, *gpu_checks;
    TTYPE *elapsed_time, *inner_elapsed_time;
    timers_and_checks_alloc<TTYPE>(buff_cycle, loop_count, error, my_error, cpu_checks, gpu_checks, elapsed_time, inner_elapsed_time);

 *
*/

template <typename T>
void timers_and_checks_alloc(int buff_cycle, int loop_count,
                                int **error, int **my_error,
                                cktype **cpu_checks, cktype **gpu_checks,
                                T **elapsed_time, T **inner_elapsed_time) {

    *error = (int*)malloc(sizeof(int)*buff_cycle);
    *my_error = (int*)malloc(sizeof(int)*buff_cycle);
    *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    *gpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    *elapsed_time = (T*)malloc(sizeof(T)*buff_cycle*loop_count);
    *inner_elapsed_time = (T*)malloc(sizeof(T)*buff_cycle*loop_count);
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
