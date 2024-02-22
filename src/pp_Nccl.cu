#include <stdio.h>
#include "mpi.h"

#include <nccl.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <unistd.h>

#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"

#define dtype u_int8_t
#define MPI_dtype MPI_CHAR
#define ncclDtype ncclChar

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define cktype int32_t
#define MPI_cktype MPI_INT

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

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

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

    // Get the group or processes of the default communicator
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Keep only the processes 0 and 1 in the new group.
    int ranks[2];
    ranks[0] = 0;
    ranks[1] = rank2;
    MPI_Group pp_group;
    MPI_Group_incl(world_group, 2, ranks, &pp_group);

    // Create the new communicator from that group of processes.
    MPI_Comm ppComm;
    MPI_Comm_create(MPI_COMM_WORLD, pp_group, &ppComm);

    // Do a broadcast only between the processes of the new communicator.

    if(ppComm == MPI_COMM_NULL) {
        // I am not part of the ppComm.
        printf("Process %d did not take part to the ppComm.\n", rank);
    } else {
        // I am part of the new ppComm.
        printf("Process %d took part to the ppComm.\n", rank);
    }
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

    // Print message based on the flags
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

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
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    float start_time, stop_time;
    int my_error[buff_cycle], error[buff_cycle];
    cktype cpu_checks[buff_cycle], gpu_checks[buff_cycle];
    float inner_elapsed_time[buff_cycle][loop_count], elapsed_time[buff_cycle][loop_count];
    if (rank == 0 || rank == rank2) {
        for(int j=fix_buff_size; j<max_j; j++){

            long int N = 1 << j;
            if (rank == 0) {printf("%i#", j); fflush(stdout);}

            // Allocate memory for A on CPU
            dtype *A = (dtype*)malloc(N*sizeof(dtype));
            dtype *B = (dtype*)malloc(N*sizeof(dtype));
            cktype my_cpu_check = 0, recv_cpu_check, gpu_check = 0;

            // Initialize all elements of A to 0.0
            for(int i=0; i<N; i++){
                A[i] = 1U * (rank+1);
                B[i] = 0U;
            }

            dtype *d_B;
            cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            dtype *d_A;
            cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(dtype), cudaMemcpyHostToDevice) );
            gpu_device_reduce(d_A, N, &my_cpu_check);

            int tag1 = 10;
            int tag2 = 20;

            /*

            Implemetantion goes here

            */
            cudaEvent_t start, stop;
            cudaErrorCheck(cudaEventCreate(&start));
            cudaErrorCheck(cudaEventCreate(&stop));

            for(int i=1-(WARM_UP); i<=loop_count; i++){
                MPI_Barrier(ppComm);
                cudaErrorCheck(cudaEventRecord(start, NULL));

                ncclGroupStart();
                if(rank == 0){
                    ncclSend(d_A, N, ncclDtype, rank2, NCCL_COMM_WORLD, NULL);
                }
                else if(rank == rank2){
                    ncclRecv(d_B, N, ncclDtype, 0, NCCL_COMM_WORLD, NULL);
                }
                ncclGroupEnd();

                ncclGroupStart();
                if(rank == 0){
                    ncclRecv(d_B, N, ncclDtype, rank2, NCCL_COMM_WORLD, NULL);
                }
                else if(rank == rank2){
                    ncclSend(d_A, N, ncclDtype, 0, NCCL_COMM_WORLD, NULL);
                }
                ncclGroupEnd();

                cudaErrorCheck(cudaEventRecord(stop, NULL));
                cudaErrorCheck(cudaEventSynchronize(stop));
                if (i>0) {cudaErrorCheck(cudaEventElapsedTime(&(inner_elapsed_time[j][i-1]), start, stop));}

                if (rank == 0) {printf("%%"); fflush(stdout);}
            }
            if (rank == 0) {printf("#\n"); fflush(stdout);}



            gpu_device_reduce(d_B, N, &gpu_check);
            if(rank == 0){
                MPI_Send(&my_cpu_check,   1, MPI_cktype, rank2, tag1, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, rank2, tag2, MPI_COMM_WORLD, &stat);
            } else if(rank == rank2){
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(&my_cpu_check,   1, MPI_cktype, 0, tag2, MPI_COMM_WORLD);
            }

            gpu_checks[j] = gpu_check;
            cpu_checks[j] = recv_cpu_check;
            my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

            cudaErrorCheck( cudaFree(d_A) );
            cudaErrorCheck( cudaFree(d_B) );
            free(A);
            free(B);
        }

        MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, ppComm);
        MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_FLOAT, MPI_MAX, ppComm);
        for(int j=fix_buff_size; j<max_j; j++) {
            long int N = 1 << j;
            long int B_in_GB = 1 << 30;
            long int num_B = sizeof(dtype)*N;
            double num_GB = (double)num_B / (double)B_in_GB;

            double avg_time_per_transfer = 0.0;
            for (int i=0; i<loop_count; i++) {
                elapsed_time[j][i] *= 0.001;
                avg_time_per_transfer += elapsed_time[j][i];
                if(rank == 0) printf("\tTransfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[j][i], num_GB/elapsed_time[j][i], i);
            }
            avg_time_per_transfer /= (2.0*(double)loop_count);

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
    }
    MPI_Finalize();
    return(0);
}
