#include <stdio.h>
#include "mpi.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"

#define dtype u_int8_t
#define MPI_dtype MPI_CHAR

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

      printf ("Assigning device %d  to process on node %s rank %d\n", myrank, host_name, rank);
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
                           int *flag_b, int *flag_l, int *flag_x, int *flag_p,
                           int *loop_count, int *buff_cycle, int *fix_buff_size, int *ncouples ) {

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
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -p without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_p = 1;
            *ncouples = atoi(argv[i + 1]);
            if (*ncouples < 0) {
                fprintf(stderr, "Error: number of ping-pong couples must be >= 1.\n");
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

    MPI_Status stat;

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

    // Check that all the nodes has the same size
    int nodesize;
    MPI_Allreduce(&mynodesize, &nodesize, sizeof(int), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (nodesize != mynodesize) {
        fprintf(stderr, "Error at node %d: mynodesize (%d) does not metch with nodesize (%d)\n", rank, mynodesize, nodesize);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    } else {
        if (rank == 0) printf("All the nodes (%d) have the same size (%d)\n", nnodes, nodesize);
        fflush(stdout);
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
    int flag_p = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;
    int ncouples = 4;

    // Parse command-line options
    read_line_parameters(argc, argv, rank,
                         &flag_b, &flag_l, &flag_x, &flag_p,
                         &loop_count, &buff_cycle, &fix_buff_size, &ncouples);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;} 
    if(!flag_p){ncouples = size / 2;}
       
    // Print message based on the flags
    if (flag_p && rank == 0) printf("Flag p was set with argument: %d\n", ncouples);
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);


    if (flag_p) {
        if (nnodes > 1) {
            if (nodesize < ncouples) {
                fprintf(stderr, "Error: mynode (%s) has less gpus (%d) then the required by -p flag (%d)\n", host_name, nodesize, ncouples);
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
            }
        } else {
            if (ncouples > 1) {
                fprintf(stderr, "Error: Multi-Ping-Pong does not support the single node set-up\n");
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
            }
        }
    }

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

    /* -------------------------------------------------------------------------------------------
        MPI Initialize Peer-to-peer communicators
    --------------------------------------------------------------------------------------------*/


    int peers[2];
    MPI_Comm ppComm;
    int my_peer = -1;

    MPI_Comm allppComm;
    int allppComm_rank = -1;
    int tmp_all_peers[2*nodesize];
    int *allpeers = (int*)malloc(sizeof(int)*ncouples*2);

    MPI_Comm allfirstsenderComm;
    int allfirstsender[ncouples];

    for (int i=0; i<2*nodesize; i++) tmp_all_peers[i] = -1;

    if ( mynode == 0 || mynode == nnodes-1 ) {
        MPI_Group pp_group;
        MPI_Group all_pp_group;
        MPI_Group all_firstsender_group;

        if ( (rank % nodesize) < ncouples ) {
            my_peer = (mynode == 0) ? (rank + (nnodes-1)*nodesize) : (rank - (nnodes-1)*nodesize);

            // Keep only the processes 0 and 1 in the new group.
            peers[0] = rank;
            peers[1] = my_peer;
            tmp_all_peers[rank] = 1;
        }

        // Get the group or processes of the default communicator
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        if ( (rank % nodesize) < ncouples )
            MPI_Group_incl(world_group, 2, peers, &pp_group);
        else
            MPI_Group_incl(world_group, 0, peers, &pp_group);

        // Create the new communicator from that group of processes.
        MPI_Comm_create(MPI_COMM_WORLD, pp_group, &ppComm);

        if(ppComm == MPI_COMM_NULL) {
            // I am not part of the ppComm.
            printf("Process %d did not take part to any ppComm.\n", rank);
        } else {
            // I am part of the new ppComm.
            printf("Process %d took part to a ppComm (%d, %d).\n", rank, peers[0], peers[1]);
        }
        fflush(stdout);

        MPI_Allgather(&my_peer, 1, MPI_INT, tmp_all_peers, 1, MPI_INT, MPI_COMM_WORLD);
        for (int i=0, j=0, k=0; i<size && j<2*ncouples; i++) {
            if (tmp_all_peers[i] != -1) {
                allpeers[j] = i;
                j++;
                if (i < tmp_all_peers[i]) {
                    allfirstsender[k] = i;
                    k++;
                }
            }
        }
        if ( (rank % nodesize) < ncouples )
            MPI_Group_incl(world_group, 2*ncouples, allpeers, &all_pp_group);
        else
            MPI_Group_incl(world_group, 0, allpeers, &all_pp_group);
        // Create the new communicator from that group of processes.
        MPI_Comm_create(MPI_COMM_WORLD, all_pp_group, &allppComm);
        if(allppComm != MPI_COMM_NULL)
            MPI_Comm_rank(allppComm, &allppComm_rank);

        if(allppComm == MPI_COMM_NULL) {
            // I am not part of the ppComm.
            printf("Process %d did not take part to the allppComm.\n", rank);
        } else {
            // I am part of the new ppComm.
            printf("Process %d took part to the allppComm (with rank %d).\n", rank, allppComm_rank);
        }
        fflush(stdout);


        if ( (rank % nodesize) < ncouples && rank < my_peer)
            MPI_Group_incl(world_group, ncouples, allfirstsender, &all_firstsender_group);
        else
            MPI_Group_incl(world_group, ncouples, allfirstsender, &all_firstsender_group);
        // Create the new communicator from that group of processes.
        MPI_Comm_create(MPI_COMM_WORLD, all_firstsender_group, &allfirstsenderComm);

        if(allfirstsenderComm == MPI_COMM_NULL) {
            // I am not part of the ppComm.
            printf("Process %d did not take part to the allfirstsenderComm.\n", rank);
        } else {
            // I am part of the new ppComm.
            printf("Process %d took part to the allfirstsenderComm.\n", rank);
        }
        fflush(stdout);

    }
    MPI_Barrier(MPI_COMM_WORLD);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    double start_time, stop_time;
    int my_error[buff_cycle], error[buff_cycle];
    cktype cpu_checks[buff_cycle], gpu_checks[buff_cycle];
    double inner_elapsed_time[buff_cycle][loop_count], elapsed_time[buff_cycle][loop_count];
    if (ppComm != MPI_COMM_NULL) {
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
            MPI_Request request[2*ncouples];

            /*

            Implemetantion goes here

            */

            for(int i=1-(WARM_UP); i<=loop_count; i++){
                MPI_Barrier(allppComm);
                start_time = MPI_Wtime();

                if(rank < my_peer){
                    MPI_Isend(d_A, N, MPI_dtype, my_peer, tag1, MPI_COMM_WORLD, &(request[allppComm_rank]));
                    MPI_Recv(d_B, N, MPI_dtype, my_peer, tag2, MPI_COMM_WORLD, &stat);
                }
                else {
                    MPI_Recv(d_B, N, MPI_dtype, my_peer, tag1, MPI_COMM_WORLD, &stat);
                    MPI_Isend(d_A, N, MPI_dtype, my_peer, tag2, MPI_COMM_WORLD, &(request[allppComm_rank]));
                }
                MPI_Wait(&(request[allppComm_rank]), MPI_STATUS_IGNORE);

                stop_time = MPI_Wtime();
                if (i>0) inner_elapsed_time[j][i-1] = stop_time - start_time;

                if (rank == 0) {printf("%%"); fflush(stdout);}
            }
            if (rank == 0) {printf("#\n"); fflush(stdout);}




            gpu_device_reduce(d_B, N, &gpu_check);
            if(rank < my_peer){
                MPI_Send(&my_cpu_check,   1, MPI_cktype, my_peer, tag1, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, my_peer, tag2, MPI_COMM_WORLD, &stat);
            } else {
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, my_peer, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(&my_cpu_check,   1, MPI_cktype, my_peer, tag2, MPI_COMM_WORLD);
            }

            gpu_checks[j] = gpu_check;
            cpu_checks[j] = recv_cpu_check;
            my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

            fflush(stdout);
            cudaErrorCheck( cudaFree(d_A) );
            cudaErrorCheck( cudaFree(d_B) );
            free(A);
            free(B);
        }

        MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, allppComm);
        MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, allfirstsenderComm);
        for(int j=fix_buff_size; j<max_j; j++) {
            long int N = 1 << j;
            long int B_in_GB = 1 << 30;
            long int num_B = sizeof(dtype)*N*ncouples;
            double num_GB = (double)num_B / (double)B_in_GB;

            double avg_time_per_transfer = 0.0;
            for (int i=0; i<loop_count; i++) {
                elapsed_time[j][i] /= 2.0;
                avg_time_per_transfer += elapsed_time[j][i];
                if(rank == 0) printf("\tTransfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[j][i], num_GB/elapsed_time[j][i], i);
            }
            avg_time_per_transfer /= (double)loop_count;

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
