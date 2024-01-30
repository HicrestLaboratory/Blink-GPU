#include <stdio.h>
#include "mpi.h"

#include <cuda.h>
#include <cuda_runtime.h>

#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"

#define dtype u_int8_t
#define MPI_dtype MPI_CHAR

#define BUFF_CYCLE 31

#define cktype int32_t
#define MPI_cktype MPI_INT

#define NSTREAM 2
#define WARM_UP 5

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

// ---------------------------- For GPU reduction -----------------------------
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

struct char2int
{
  __host__ __device__ cktype operator()(const dtype &x) const
  {
    return static_cast<cktype>(x);
  }
};

int gpu_host_reduce(dtype* input_vec, int len, cktype* out_scalar) {
  int result = thrust::transform_reduce(thrust::host,
                                        input_vec, input_vec + len,
                                        char2int(),
                                        0,
                                        thrust::plus<cktype>());

  *out_scalar = result;

  return 0;
}

int gpu_device_reduce(dtype* d_input_vec, int len, cktype* out_scalar) {
  cktype result = thrust::transform_reduce(thrust::device,
                                        d_input_vec, d_input_vec + len,
                                        char2int(),
                                        0,
                                        thrust::plus<cktype>());

  *out_scalar = result;

  return 0;
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
    fflush(stdout);



    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int size, nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank, mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    if (rank == 0) printf("MPI_COMM_WORLD: rank %d of size %d\n", rank, size);
    if (rank == 0) printf("nodeComm: rank %d of size %d\n", mynode, nnodes);
    fflush(stdout);

    MPI_Request req[NSTREAM];
    MPI_Status status[NSTREAM];

    int rank2 = size-1;

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    cktype cpu_checks[2][BUFF_CYCLE], gpu_checks[2][BUFF_CYCLE];
    if (rank == 0 || rank == rank2) {
        for(int j=0; j<BUFF_CYCLE; j++){

            long int N = 1 << j;

            // Allocate memory for A on CPU
            dtype *A0 = (dtype*)malloc(N*sizeof(dtype));
            dtype *A1 = (dtype*)malloc(N*sizeof(dtype));
            dtype *B0 = (dtype*)malloc(N*sizeof(dtype));
            dtype *B1 = (dtype*)malloc(N*sizeof(dtype));
            cktype my_cpu_check0 = 0, recv_cpu_check0, gpu_check0 = 0;
            cktype my_cpu_check1 = 0, recv_cpu_check1, gpu_check1 = 0;

            // Initialize all elements of A to 0.0
            for(int i=0; i<N; i++){
                A0[i] = 1U * (rank+1);
                A1[i] = 1U * (size - rank);
                B0[i] = 0U;
                B1[i] = 0U;
            }

            dtype *d_B0;
            cudaErrorCheck( cudaMalloc(&d_B0, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B0, B0, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            dtype *d_B1;
            cudaErrorCheck( cudaMalloc(&d_B1, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B1, B1, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            dtype *d_A0;
            cudaErrorCheck( cudaMalloc(&d_A0, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A0, A0, N*sizeof(dtype), cudaMemcpyHostToDevice) );
            gpu_device_reduce(d_A0, N, &my_cpu_check0);

            dtype *d_A1;
            cudaErrorCheck( cudaMalloc(&d_A1, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A1, A1, N*sizeof(dtype), cudaMemcpyHostToDevice) );
            gpu_device_reduce(d_A1, N, &my_cpu_check1);

            int tag1 = 10, tag2 = 20, tag3 = 30, tag4 = 40;

            int loop_count = 50;
            double start_time, stop_time, inner_elapsed_time, elapsed_time = 0.0;

            printf("[%d] check at line %d\n", rank, __LINE__);
            fflush(stdout);

            /*

            Implemetantion goes here

            */
            for(int i=1-(WARM_UP); i<=loop_count; i++){
                start_time = MPI_Wtime();

                if(rank == 0){
                    MPI_Isend(d_A0, N, MPI_dtype, rank2, tag1, MPI_COMM_WORLD, &req[0]);
                    MPI_Isend(d_A1, N, MPI_dtype, rank2, tag2, MPI_COMM_WORLD, &req[1]);

                    /* Implicit syncronization, rank2 wait for receving before sending */

                    MPI_Irecv(d_B0, N, MPI_dtype, rank2, tag3, MPI_COMM_WORLD, &req[0]);
                    MPI_Irecv(d_B1, N, MPI_dtype, rank2, tag4, MPI_COMM_WORLD, &req[1]);
                }
                else if(rank == rank2){
                    MPI_Irecv(d_B0, N, MPI_dtype, 0, tag1, MPI_COMM_WORLD, &req[0]);
                    MPI_Irecv(d_B1, N, MPI_dtype, 0, tag2, MPI_COMM_WORLD, &req[1]);

                    MPI_Waitall(2, req, status);
                    MPI_Isend(d_A0, N, MPI_dtype, 0, tag3, MPI_COMM_WORLD, &req[0]);
                    MPI_Isend(d_A1, N, MPI_dtype, 0, tag4, MPI_COMM_WORLD, &req[1]);
                }
                MPI_Waitall(2, req, status);

                stop_time = MPI_Wtime();
                inner_elapsed_time = stop_time - start_time;
                if(rank == 0) printf("\t\tCycle: %d, Elapsed Time (s): %15.9f\n", j, inner_elapsed_time);
                elapsed_time += inner_elapsed_time;
            }





            gpu_device_reduce(d_B0, N, &gpu_check0);
            gpu_device_reduce(d_B1, N, &gpu_check1);
            if(rank == 0){
                MPI_Send(&my_cpu_check0,   1, MPI_cktype, rank2, tag1, MPI_COMM_WORLD);
                MPI_Send(&my_cpu_check1,   1, MPI_cktype, rank2, tag2, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check0, 1, MPI_cktype, rank2, tag3, MPI_COMM_WORLD, &status[1]);
                MPI_Recv(&recv_cpu_check1, 1, MPI_cktype, rank2, tag4, MPI_COMM_WORLD, &status[2]);
            } else if(rank == rank2) {
                MPI_Recv(&recv_cpu_check0, 1, MPI_cktype, 0, tag1, MPI_COMM_WORLD, &status[1]);
                MPI_Recv(&recv_cpu_check1, 1, MPI_cktype, 0, tag2, MPI_COMM_WORLD, &status[2]);
                MPI_Send(&my_cpu_check0,   1, MPI_cktype, 0, tag3, MPI_COMM_WORLD);
                MPI_Send(&my_cpu_check1,   1, MPI_cktype, 0, tag4, MPI_COMM_WORLD);
            }

            gpu_checks[0][j] = gpu_check0;
            gpu_checks[1][j] = gpu_check1;
            cpu_checks[0][j] = recv_cpu_check0;
            cpu_checks[1][j] = recv_cpu_check1;
            long int num_B = sizeof(dtype)*N;
            long int B_in_GB = 1 << 30;
            double num_GB = (double)num_B / (double)B_in_GB;
            double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

            if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error0: %d, Error1: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, abs(gpu_check0 - recv_cpu_check0), abs(gpu_check1 - recv_cpu_check1) );
            fflush(stdout);

            cudaErrorCheck( cudaFree(d_A0) );
            cudaErrorCheck( cudaFree(d_A1) );
            cudaErrorCheck( cudaFree(d_B0) );
            cudaErrorCheck( cudaFree(d_B1) );
            free(A0);
            free(A1);
            free(B0);
            free(B1);
        }

        char s[10000000];
        sprintf(s, "[%d] recv_cpu_check[0] = %u", rank, cpu_checks[0][0]);
        for (int i=0; i<BUFF_CYCLE; i++) {
            sprintf(s+strlen(s), " %10d", cpu_checks[0][i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);

        sprintf(s, "[%d] recv_cpu_check[1] = %u", rank, cpu_checks[1][0]);
        for (int i=0; i<BUFF_CYCLE; i++) {
            sprintf(s+strlen(s), " %10d", cpu_checks[1][i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);

        sprintf(s, "[%d] gpu_checks[0] = %u", rank, gpu_checks[0][0]);
        for (int i=0; i<BUFF_CYCLE; i++) {
            sprintf(s+strlen(s), " %10d", gpu_checks[0][i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);

        sprintf(s, "[%d] gpu_checks[1] = %u", rank, gpu_checks[1][0]);
        for (int i=0; i<BUFF_CYCLE; i++) {
            sprintf(s+strlen(s), " %10d", gpu_checks[1][i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);
    }
    MPI_Finalize();
    return(0);
}
