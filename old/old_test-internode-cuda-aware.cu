#include <stdio.h>
#include "mpi.h"

#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"


// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
        exit(0);                                                                            \
    }                                                                                     \
}while(0)




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

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status stat;

//     if(size != 2){
//         if(rank == 0){
//             printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
//         }
//         MPI_Finalize();
//         exit(0);
//     }

    int rank2 = size-1;

    // Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
    cudaErrorCheck( cudaSetDevice(rank % num_devices) );

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    if (rank == 0 || rank == rank2) {
        for(int i=0; i<=27; i++){

            long int N = 1 << i;

            // Allocate memory for A on CPU
            double *A = (double*)malloc(N*sizeof(double));
            double *B = (double*)malloc(N*sizeof(double));
            double my_cpu_check = 1.0, recv_cpu_check;

            // Initialize all elements of A to 0.0
            for(int i=0; i<N; i++){
                A[i] = 1.0 * (rank+1) + i * 0.0001;
                my_cpu_check += A[i];
                B[i] = 0.0;
            }

            double *d_B;
            cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
            cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice) );

            double *d_A;
            cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

            int tag1 = 10;
            int tag2 = 20;

            int loop_count = 50;
        double start_time, stop_time, elapsed_time;
            start_time = MPI_Wtime();
    /*

    Implemetantion goes here

    */
            for(int i=1; i<=loop_count; i++){
                if(rank == 0){
                    MPI_Send(d_A, N, MPI_DOUBLE, rank2, tag1, MPI_COMM_WORLD);
                    MPI_Recv(d_B, N, MPI_DOUBLE, rank2, tag2, MPI_COMM_WORLD, &stat);
                }
                else if(rank == rank2){
                    MPI_Recv(d_B, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                    MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
                }
            }




        stop_time = MPI_Wtime();
            elapsed_time = stop_time - start_time;

            cudaErrorCheck( cudaMemcpy(B, d_B, sizeof(double)*N, cudaMemcpyDeviceToHost) );
            double gpu_check = 1.0;
            for(int i=0; i<N; i++)
                gpu_check += B[i];
            if(rank == 0){
                MPI_Send(&my_cpu_check,   1, MPI_DOUBLE, rank2, tag1, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check, 1, MPI_DOUBLE, rank2, tag2, MPI_COMM_WORLD, &stat);
            } else if(rank == rank2){
                MPI_Recv(&recv_cpu_check, 1, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(&my_cpu_check,   1, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }

            long int num_B = 8*N;
            long int B_in_GB = 1 << 30;
            double num_GB = (double)num_B / (double)B_in_GB;
            double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

            if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %lf\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, fabs(gpu_check - recv_cpu_check) );
            fflush(stdout);
            cudaErrorCheck( cudaFree(d_A) );
            free(A);
        }
    }
    MPI_Finalize();
    return 0;
}
