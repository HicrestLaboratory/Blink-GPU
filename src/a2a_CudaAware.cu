#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"

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

void read_line_parameters (int argc, char *argv[], int myrank,
                           int *flag_b, int *flag_l, int *flag_x,
                           int *loop_count, int *buff_cycle, int *fix_buff_size ) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-l") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                    fprintf(stderr, "Error: specified -l without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_l = 1;
            *loop_count = atoi(argv[i + 1]);
            if (*loop_count <= 0) {
                fprintf(stderr, "Error: loop_count must be a positive integer.\n");
                exit(__LINE__);
            }
            i++;
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                    fprintf(stderr, "Error: specified -b without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_b = 1;
            *buff_cycle = atoi(argv[i + 1]);
            if (*buff_cycle <= 0) {
                fprintf(stderr, "Error: buff_cycle must be a positive integer.\n");
                exit(__LINE__);
            }
            i++;
        } else if (strcmp(argv[i], "-x") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -x without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_x = 1;
            *fix_buff_size = atoi(argv[i + 1]);
            if (*fix_buff_size < 0) {
                fprintf(stderr, "Error: fixed buff_size must be >= 0.\n");
                exit(__LINE__);
            }

            i++;
        } else {
            if (0 == myrank) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
            }

            exit(__LINE__);
        }
    }
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

    double start_time, stop_time;
    int *error = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    double *elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    for(int j=fix_buff_size; j<max_j; j++){

        long int N = 1 << j;
        if (rank == 0) {printf("%i#", j); fflush(stdout);}

        // Allocate memory for A on CPU
        dtype *A, *B;
#ifdef PINNED
        cudaHostAlloc(&A, size*N*sizeof(dtype), cudaHostAllocDefault);
        cudaHostAlloc(&B, size*N*sizeof(dtype), cudaHostAllocDefault);
#else
        A = (dtype*)malloc(size*N*sizeof(dtype));
        B = (dtype*)malloc(size*N*sizeof(dtype));
#endif
        cktype *my_cpu_check = (cktype*)malloc(sizeof(cktype)*size);
        cktype *recv_cpu_check = (cktype*)malloc(sizeof(cktype)*size), gpu_check = 0;
        for (int i=0; i<size; i++)
            my_cpu_check[i] = 0U;

        // Initialize all elements of A to 0.0
        for(int i=0; i<N*size; i++) {
            A[i] = 1U * (rank+1);
            B[i] = 0U;
        }

        dtype *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, size*N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_B, B, size*N*sizeof(dtype), cudaMemcpyHostToDevice) );

        dtype *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, size*N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, size*N*sizeof(dtype), cudaMemcpyHostToDevice) );

        for (int i=0; i<size; i++)
            gpu_device_reduce(d_A + (i*N)*sizeof(dtype), N, &my_cpu_check[i]);

        /*

        Implemetantion goes here

        */

        for(int i=1-(WARM_UP); i<=loop_count; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            start_time = MPI_Wtime();

            MPI_Alltoall(d_A, N, MPI_dtype, d_B, N, MPI_dtype, MPI_COMM_WORLD);

            stop_time = MPI_Wtime();
            if (i>0) inner_elapsed_time[j*buff_cycle+i-1] = stop_time - start_time;

            if (rank == 0) {printf("%%"); fflush(stdout);}
        }
        if (rank == 0) {printf("#\n"); fflush(stdout);}




        gpu_device_reduce(d_B, size*N, &gpu_check);
        MPI_Alltoall(my_cpu_check, 1, MPI_cktype, recv_cpu_check, 1, MPI_cktype, MPI_COMM_WORLD);

        cpu_checks[j] = 0;
        gpu_checks[j] = gpu_check;
        for (int i=0; i<size; i++)
            cpu_checks[j] += recv_cpu_check[i];
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
    MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int j=fix_buff_size; j<max_j; j++) {
        long int N = 1 << j;
        long int B_in_GB = 1 << 30;
        long int num_B = sizeof(dtype)*N*(size-1);
        double num_GB = (double)num_B / (double)B_in_GB;

        double avg_time_per_transfer = 0.0;
        for (int i=0; i<loop_count; i++) {
            avg_time_per_transfer += elapsed_time[j*buff_cycle+i];
            if(rank == 0) printf("\tTransfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[j*buff_cycle+i], num_GB/elapsed_time[j*buff_cycle+i], i);
        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
        fflush(stdout);
    }

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
