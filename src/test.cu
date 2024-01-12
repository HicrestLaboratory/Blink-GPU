#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>

#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"

#define dtype double
#define MPI_dtype MPI_DOUBLE

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
        exit(0);                                                                            \
    }                                                                                     \
}while(0)

#define MPI

static int stringCmp( const void *a, const void *b) {
     return strcmp((const char*)a,(const char*)b);

}

int  assignDeviceToProcess(MPI_Comm *nodeComm, int *nnodes, int *mynodeid)
{
#ifdef MPI
      char     host_name[MPI_MAX_PROCESSOR_NAME];
      char (*host_names)[MPI_MAX_PROCESSOR_NAME];

#else
      char     host_name[20];
#endif
      int myrank;
      int gpu_per_node;
      int n, namelen, color, rank, nprocs;
      size_t bytes;
/*
      if (chkseGPU()<1 && 0) {
        fprintf(stderr, "Invalid GPU Serial number\n");
	exit(EXIT_FAILURE);
      }
*/

#ifdef MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
      MPI_Get_processor_name(host_name,&namelen);

      bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
      host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

      strcpy(host_names[rank], host_name);

      for (n=0; n<nprocs; n++)
      {
       MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR,n, MPI_COMM_WORLD);
      }


      qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

      color = 0;

      for (n=0; n<nprocs; n++)
      {
        if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
        if(strcmp(host_name, host_names[n]) == 0) break;
      }

      MPI_Comm_split(MPI_COMM_WORLD, color, 0, nodeComm);

      MPI_Comm_rank(*nodeComm, &myrank);
      MPI_Comm_size(*nodeComm, &gpu_per_node);

      MPI_Allreduce(&color, nnodes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      (*mynodeid) = color;
      (*nnodes) ++;

#else
     //*myrank = 0;
     return 0;
#endif

//      printf ("Assigning device %d  to process on node %s rank %d\n",*myrank,  host_name, rank );
      /* Assign device to MPI process, initialize BLAS and probe device properties */
      //cudaSetDevice(*myrank);
      return myrank;
}


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

    MPI_Status stat;

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
//     cudaErrorCheck( cudaSetDevice(rank % num_devices) );

    MPI_Comm nodeComm;
    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
    cudaSetDevice(dev);

    int mynodeid = -1, mynodesize = -1;
    MPI_Comm_rank(nodeComm, &mynodeid);
    MPI_Comm_size(nodeComm, &mynodesize);

    int rank2 = size-1;

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    dtype cpu_checks[28];
    if (rank == 0 || rank == rank2) {
        for(int i=0; i<=27; i++){

            long int N = 1 << i;

            // Allocate memory for A on CPU
            dtype *A = (dtype*)malloc(N*sizeof(dtype));
            dtype *B = (dtype*)malloc(N*sizeof(dtype));
            dtype my_cpu_check = 1.0, recv_cpu_check;

            // Initialize all elements of A to 0.0
            for(int i=0; i<N; i++){
                A[i] = 1.0 * (rank+1) + i * 0.0001;
                my_cpu_check += A[i];
                B[i] = 0.0;
            }

            dtype *d_B;
            cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            dtype *d_A;
            cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(dtype), cudaMemcpyHostToDevice) );

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
                    cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(dtype), cudaMemcpyDeviceToHost) );
                    MPI_Send(A, N, MPI_dtype, rank2, tag1, MPI_COMM_WORLD);
                    MPI_Recv(B, N, MPI_dtype, rank2, tag2, MPI_COMM_WORLD, &stat);
                    cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );
                }
                else if(rank == rank2){
                    MPI_Recv(B, N, MPI_dtype, 0, tag1, MPI_COMM_WORLD, &stat);
                    cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );
                    cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(dtype), cudaMemcpyDeviceToHost) );
                    MPI_Send(A, N, MPI_dtype, 0, tag2, MPI_COMM_WORLD);
                }
            }




        stop_time = MPI_Wtime();
            elapsed_time = stop_time - start_time;

            cudaErrorCheck( cudaMemcpy(B, d_B, sizeof(dtype)*N, cudaMemcpyDeviceToHost) );
            dtype gpu_check = 1.0;
            for(int i=0; i<N; i++)
                gpu_check += B[i];
            if(rank == 0){
                MPI_Send(&my_cpu_check,   1, MPI_dtype, rank2, tag1, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check, 1, MPI_dtype, rank2, tag2, MPI_COMM_WORLD, &stat);
            } else if(rank == rank2){
                MPI_Recv(&recv_cpu_check, 1, MPI_dtype, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(&my_cpu_check,   1, MPI_dtype, 0, tag2, MPI_COMM_WORLD);
            }

            cpu_checks[i] = recv_cpu_check;
            long int num_B = sizeof(dtype)*N;
            long int B_in_GB = 1 << 30;
            double num_GB = (double)num_B / (double)B_in_GB;
            double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

            if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %lf\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, fabs(gpu_check - recv_cpu_check) );
            fflush(stdout);
            cudaErrorCheck( cudaFree(d_A) );
            free(A);
        }

        char s[10000000];
        sprintf(s, "[%d] recv_cpu_check = %lf", rank, cpu_checks[0]);
        for (int i=0; i<28; i++) {
            sprintf(s+strlen(s), " %10.5lf", cpu_checks[i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
