#include <stdio.h>
#include "mpi.h"

#include <nccl.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/cmd_util.h"

#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"

#define BUFF_CYCLE 28
#define LOOP_COUNT 50

#define WARM_UP 5

#define MPI

static int stringCmp( const void *a, const void *b) {
     return strcmp((const char*)a,(const char*)b);

}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
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




    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int size, nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank, mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int namelen;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(host_name, &namelen);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Size = %d, myrank = %d, host_name = %s\n", size, rank, host_name);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);


//     if(size != 2){
//         if(rank == 0){
//             printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
//         }
//         MPI_Finalize();
//         exit(0);
//     }

    // Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );

    MPI_Comm nodeComm;
    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
    cudaSetDevice(dev);

    int mynodeid = -1, mynodesize = -1;
    MPI_Comm_rank(nodeComm, &mynodeid);
    MPI_Comm_size(nodeComm, &mynodesize);


    /* -------------------------------------------------------------------------------------------
        NCCL Initialization
    --------------------------------------------------------------------------------------------*/
    ncclUniqueId Id;
    ncclComm_t NCCL_COMM_WORLD, NCCL_COMM_NODE;

    ncclGroupStart();
    if (mynodeid == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
    MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, nodeComm);
    NCCLCHECK( ncclCommInitRank(&NCCL_COMM_NODE, mynodesize, Id, mynodeid) );
    ncclGroupEnd();
    MPI_Barrier(MPI_COMM_WORLD);

    ncclGroupStart();
    if (rank == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
    MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCLCHECK( ncclCommInitRank(&NCCL_COMM_WORLD, size, Id, rank) );
    ncclGroupEnd();

    int nccl_w_rk;
    int nccl_w_sz;
    ncclGroupStart();
    NCCLCHECK( ncclCommCount(NCCL_COMM_WORLD, &nccl_w_sz)   );
    NCCLCHECK( ncclCommUserRank(NCCL_COMM_WORLD, &nccl_w_rk) );
    ncclGroupEnd();

    int nccl_n_rk;
    int nccl_n_sz;
    ncclGroupStart();
    NCCLCHECK( ncclCommCount(NCCL_COMM_NODE, &nccl_n_sz)   );
    NCCLCHECK( ncclCommUserRank(NCCL_COMM_NODE, &nccl_n_rk) );
    ncclGroupEnd();

    printf("[%d] NCCL_COMM_WORLD: nccl size = %d, nccl rank = %d\n", rank, nccl_w_sz, nccl_w_rk);
    printf("[%d] NCCL_COMM_NODE:  nccl size = %d, nccl rank = %d\n", rank, nccl_n_sz, nccl_n_rk);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);


    /* -------------------------------------------------------------------------------------------
        Reading command line inputs
    --------------------------------------------------------------------------------------------*/

    int opt;
    int max_j;
    int flag_b = 0;
    int flag_l = 0;
    int flag_x = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;

    // Parse command-line options
    read_line_parameters(argc, argv, rank,
                         &flag_b, &flag_l, &flag_x,
                         &loop_count, &buff_cycle, &fix_buff_size);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;}    
    // Print message based on the flags
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    float start_time, stop_time;
    int *error = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    float *elapsed_time = (float*)malloc(sizeof(float)*buff_cycle*loop_count);
    float *inner_elapsed_time = (float*)malloc(sizeof(float)*buff_cycle*loop_count);
    for(int j=fix_buff_size; j<max_j; j++){

        long int N = 1 << j;
        if (rank == 0) {printf("%i#", j); fflush(stdout);}

        // Allocate memory for A on CPU
        dtype *A, *B;
#ifdef PINNED
        cudaHostAlloc(&A, N*sizeof(dtype), cudaHostAllocDefault);
        cudaHostAlloc(&B, N*sizeof(dtype), cudaHostAllocDefault);
#else
        A = (dtype*)malloc(N*sizeof(dtype));
        B = (dtype*)malloc(N*sizeof(dtype));
#endif
        cktype *my_cpu_check = (cktype*)malloc(sizeof(cktype));
        cktype *recv_cpu_check = (cktype*)malloc(sizeof(cktype)*size), gpu_check = 0;
        *my_cpu_check = 0U;

        // Initialize all elements of A to 0.0
        for(int i=0; i<N; i++) {
            A[i] = 1U * (rank+1);
        }
        *B = 0U;

        dtype *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );

        dtype *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(dtype), cudaMemcpyHostToDevice) );

        gpu_device_reduce_max(d_A, N, my_cpu_check);


        /*

        Implemetantion goes here

        */
        cudaEvent_t start, stop;
        cudaErrorCheck(cudaEventCreate(&start));
        cudaErrorCheck(cudaEventCreate(&stop));

        for(int i=1-(WARM_UP); i<=loop_count; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            cudaErrorCheck(cudaEventRecord(start, NULL));

            ncclAllReduce(d_A, d_B, N, ncclDtype, ncclMax, NCCL_COMM_WORLD, NULL);

            cudaErrorCheck(cudaEventRecord(stop, NULL));
            cudaErrorCheck(cudaEventSynchronize(stop));
            if (i>0) {cudaErrorCheck(cudaEventElapsedTime(&(inner_elapsed_time[j*buff_cycle+i-1]), start, stop));}

            if (rank == 0) {printf("%%"); fflush(stdout);}
        }
        if (rank == 0) {printf("#\n"); fflush(stdout);}



        gpu_device_reduce_max(d_B, N, &gpu_check);
        MPI_Allgather(my_cpu_check, 1, MPI_cktype, recv_cpu_check, 1, MPI_cktype, MPI_COMM_WORLD);

        cpu_checks[j] = 0;
        gpu_checks[j] = gpu_check;
        for (int i=0; i<size; i++)
            if (cpu_checks[j] < recv_cpu_check[i]) cpu_checks[j] = recv_cpu_check[i];
        my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

        cudaErrorCheck( cudaFree(d_A) );
        cudaErrorCheck( cudaFree(d_B) );
        free(recv_cpu_check);
        free(my_cpu_check);
#ifdef PINNED
        cudaFreeHost(A);
        cudaFreeHost(B);
#else
        free(A);
        free(B);
#endif
    }

    MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    for(int j=fix_buff_size; j<max_j; j++){
        long int N = 1 << j;
        long int B_in_GB = 1 << 30;
        long int num_B = sizeof(dtype)*N*((size-1)/(float)size)*2;
        double num_GB = (double)num_B / (double)B_in_GB;

        double avg_time_per_transfer = 0.0;
        for (int i=0; i<loop_count; i++) {
            elapsed_time[j*buff_cycle+i] *= 0.001;
            avg_time_per_transfer += elapsed_time[j*buff_cycle+i];
            if(rank == 0) printf("\tTransfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[j*buff_cycle+i], num_GB/elapsed_time[j*buff_cycle+i], i);
        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
    }
    fflush(stdout);

    char *s = (char*)malloc(sizeof(char)*(20*buff_cycle + 100));
    sprintf(s, "[%d] recv_cpu_check = %u", rank, cpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", cpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    sprintf(s, "[%d] gpu_checks = %u", rank, gpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", gpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    free(error);
    free(my_error);
    free(cpu_checks);
    free(gpu_checks);
    free(elapsed_time);
    free(inner_elapsed_time);
    MPI_Finalize();
    return(0);
}
