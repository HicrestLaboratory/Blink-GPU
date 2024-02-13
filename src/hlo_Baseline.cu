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

#include <time.h>
#include <sys/time.h>
#include "../include/experiment_utils.h"

#define dtype u_int8_t
#define MPI_dtype MPI_CHAR

#define BUFF_CYCLE 24

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

//      printf ("Assigning device %d  to process on node %s rank %d\n",*myrank,  host_name, rank );
      /* Assign device to MPI process, initialize BLAS and probe device properties */
      //cudaSetDevice(*myrank);
      return myrank;
}


// ---------------------------- For GPU reduction -----------------------------
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "../include/debug_utils.h"

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

// ------------------------------- For Halo 3D --------------------------------

void get_position(const int rank, const int pex, const int pey, const int pez,
                  int* myX, int* myY, int* myZ) {
  const int plane = rank % (pex * pey);
  *myY = plane / pex;
  *myX = (plane % pex) != 0 ? (plane % pex) : 0;
  *myZ = rank / (pex * pey);
}

int convert_position_to_rank(const int pX, const int pY, const int pZ,
                             const int myX, const int myY, const int myZ) {
  // Check if we are out of bounds on the grid
  if ((myX < 0) || (myY < 0) || (myZ < 0) || (myX >= pX) || (myY >= pY) ||
      (myZ >= pZ)) {
    return -1;
  } else {
    return (myZ * (pX * pY)) + (myY * pX) + myX;
  }
}

void read_line_parameters (int argc, char *argv[], int myrank,
                           int *nx,  int *ny,  int *nz,
                           int *pex, int *pey, int *pez,
                           int *repeats, int *vars, long *sleep ) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-nx") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -nx without a value.\n");
            }

            exit(-1);
        }

        *nx = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-ny") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -ny without a value.\n");
            }

            exit(-1);
        }

        *ny = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-nz") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -nz without a value.\n");
            }

            exit(-1);
        }

        *nz = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-pex") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -pex without a value.\n");
            }

            exit(-1);
        }

        *pex = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-pey") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -pey without a value.\n");
            }

            exit(-1);
        }

        *pey = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-pez") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -pez without a value.\n");
            }

            exit(-1);
        }

        *pez = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-iterations") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -iterations without a value.\n");
            }

            exit(-1);
        }

        *repeats = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-vars") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -vars without a value.\n");
            }

            exit(-1);
        }

        *vars = atoi(argv[i + 1]);
        ++i;
        } else if (strcmp(argv[i], "-sleep") == 0) {
        if (i == argc) {
            if (myrank == 0) {
            fprintf(stderr, "Error: specified -sleep without a value.\n");
            }

            exit(-1);
        }

        *sleep = atol(argv[i + 1]);
        ++i;
        } else {
        if (0 == myrank) {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
        }

        exit(-1);
        }
    }
}

// HB --> Host_buffer, DB --> Device_buffer, DT --> data type, SZ --> size
#define INIT_HALO3D_BUFFER(HB, DB, DT, SZ) {                \
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);         \
    HB = (DT*)malloc(sizeof(DT) * SZ);                      \
    cudaErrorCheck( cudaMalloc(&DB, sizeof(DT) * SZ) );     \
    cudaErrorCheck( cudaMemset(DB, 0, sizeof(DT) * SZ) );   \
    cudaErrorCheck( cudaDeviceSynchronize() );              \
}

#define FREE_HALO3D_BUFFER(HB, DB) {                \
    cudaErrorCheck( cudaFree(DB) );                 \
    free(HB);                                       \
}

#define BLK_SIZE 256
#define GRD_SIZE 4
#define TID_DIGITS 10000

__global__
void init_kernel(int n, dtype *input, int rank) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < n)
      input[tid] = (dtype)(rank+1);

}

void halo3d_run_axes(int BufferSize, int UpFlag, int DownFlag,
                     dtype *host_UpSendBuffer,   dtype *dev_UpSendBuffer,   dtype *host_UpRecvBuffer,   dtype *dev_UpRecvBuffer,
                     dtype *host_DownSendBuffer, dtype *dev_DownSendBuffer, dtype *host_DownRecvBuffer, dtype *dev_DownRecvBuffer,
                     double *timeTakenCUDA, double *timeTakenMPI, int tag) {
    TIMER_DEF(0);

    int requestcount = 0;
    MPI_Status* status;
    status = (MPI_Status*)malloc(sizeof(MPI_Status) * 4);

    MPI_Request* requests;
    requests = (MPI_Request*)malloc(sizeof(MPI_Request) * 4);
    // =================================================================================================================

    // ---------------------------------------
    TIMER_START(0);
    if (UpFlag > -1) {
      cudaErrorCheck( cudaMemcpy(host_UpSendBuffer, dev_UpSendBuffer, BufferSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    if (DownFlag > -1) {
      cudaErrorCheck( cudaMemcpy(host_DownSendBuffer, dev_DownSendBuffer, BufferSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    TIMER_STOP(0);
    *timeTakenCUDA += TIMER_ELAPSED(0);
    MPI_Barrier(MPI_COMM_WORLD);
    // ---------------------------------------

    TIMER_START(0);
    if (UpFlag > -1) {
      MPI_Irecv(host_UpRecvBuffer, BufferSize, MPI_dtype, UpFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(host_UpSendBuffer, BufferSize, MPI_dtype, UpFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (DownFlag > -1) {
      MPI_Irecv(host_DownRecvBuffer, BufferSize, MPI_dtype, DownFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(host_DownSendBuffer, BufferSize, MPI_dtype, DownFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    *timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (UpFlag > -1) {
      cudaErrorCheck( cudaMemcpy(dev_UpRecvBuffer, host_UpRecvBuffer, BufferSize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    if (DownFlag > -1) {
      cudaErrorCheck( cudaMemcpy(dev_DownRecvBuffer, host_DownRecvBuffer, BufferSize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    cudaErrorCheck( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    *timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // =================================================================================================================
}

unsigned int check_recv_buffer (int my_rank, char axe,
                                int UpFlag, dtype *dev_UpBuffer,
                                int DownFlag, dtype *dev_DownBuffer,
                                int BufferSize) {

    cktype UpCheck = 0, DownCheck = 0;
    if (UpFlag>-1) gpu_device_reduce(dev_UpBuffer, BufferSize, &UpCheck);
    if (DownFlag>-1) gpu_device_reduce(dev_DownBuffer, BufferSize, &DownCheck);

    unsigned int result = 0U;
    if ( UpFlag>-1 && UpCheck != BufferSize*(UpFlag+1) ) result |= 1U;
    if ( DownFlag>-1 && DownCheck != BufferSize*(DownFlag+1) ) result |= 2U;
//     printf("[BufferSize=%d, myRank=%d, axe=%c] UpFlag = %d, UpCheck = %d, DownFlag = %d, DownCheck = %d --> %u\n", BufferSize, my_rank, axe, UpFlag, UpCheck, DownFlag, DownCheck, result);
    return(result);
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

    int rank2 = size-1;
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------------------------
        Halo 3D Initialization
    --------------------------------------------------------------------------------------------*/

    // Define halo3D parameters
    int pex = size, pey = 1, pez = 1;
    int nx = 10, ny = 10, nz = 10;
    long sleep = 1000;
    int repeats = 100;
    int vars = 1;

    // Read input parameters
    read_line_parameters(argc, argv, rank, &nx, &ny, &nz, &pex, &pey, &pez, &repeats, &vars, &sleep);
    MPI_Barrier(MPI_COMM_WORLD);

    // Check for correct phisical initizlization
    if ((pex * pey * pez) != size) {
        if (0 == rank) {
        fprintf(stderr, "Error: rank grid does not equal number of ranks.\n");
        fprintf(stderr, "%7d x %7d x %7d != %7d\n", pex, pey, pez, size);
        }

        exit(-1);
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Print data
    if (rank == 0) {
        printf("# MPI Nearest Neighbor Communication\n");
        printf("# Info:\n");
        printf("# Processor Grid:         %7d x %7d x %7d\n", pex, pey, pez);
        printf("# Data Grid (per rank):   %7d x %7d x %7d\n", nx, ny, nz);
        printf("# Iterations:             %7d\n", repeats);
        printf("# Variables:              %7d\n", vars);
        printf("# Sleep:                  %7ld\n", sleep);
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute position and flags
    int posX, posY, posZ;
    get_position(rank, pex, pey, pez, &posX, &posY, &posZ);

    int xUp = convert_position_to_rank(pex, pey, pez, posX + 1, posY, posZ);
    int yUp = convert_position_to_rank(pex, pey, pez, posX, posY + 1, posZ);
    int zUp = convert_position_to_rank(pex, pey, pez, posX, posY, posZ + 1);
    int xDown = convert_position_to_rank(pex, pey, pez, posX - 1, posY, posZ);
    int yDown = convert_position_to_rank(pex, pey, pez, posX, posY - 1, posZ);
    int zDown = convert_position_to_rank(pex, pey, pez, posX, posY, posZ - 1);

    // Declare buffers and sizes variables
    size_t xSize, ySize, zSize;

    dtype     *xUpSendBuffer,     *xUpRecvBuffer,     *xDownSendBuffer,     *xDownRecvBuffer;
    dtype *dev_xUpSendBuffer, *dev_xUpRecvBuffer, *dev_xDownSendBuffer, *dev_xDownRecvBuffer;

    dtype     *yUpSendBuffer,     *yUpRecvBuffer,     *yDownSendBuffer,     *yDownRecvBuffer;
    dtype *dev_yUpSendBuffer, *dev_yUpRecvBuffer, *dev_yDownSendBuffer, *dev_yDownRecvBuffer;

    dtype     *zUpSendBuffer,     *zUpRecvBuffer,     *zDownSendBuffer,     *zDownRecvBuffer;
    dtype *dev_zUpSendBuffer, *dev_zUpRecvBuffer, *dev_zDownSendBuffer, *dev_zDownRecvBuffer;


     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    int loop_count = 50;
    double start_time, stop_time;
    double cuda_timer[3], mpi_timer[3];
    unsigned int halo_checks[BUFF_CYCLE];
    double inner_elapsed_time[BUFF_CYCLE][loop_count], elapsed_time[BUFF_CYCLE][loop_count];
    for(int j=0; j<BUFF_CYCLE; j++){

        // Define cycle sizes
        long int N = 1 << j;
        xSize = ny * nz * N;
        ySize = nx * nz * N;
        zSize = nx * ny * N;
        halo_checks[j] = 0U;

//         STR_COLL_DEF
//         STR_COLL_INIT

        // Alloc x axe
        INIT_HALO3D_BUFFER(xUpSendBuffer, dev_xUpSendBuffer, dtype, xSize)
        INIT_HALO3D_BUFFER(xUpRecvBuffer, dev_xUpRecvBuffer, dtype, xSize)
        INIT_HALO3D_BUFFER(xDownSendBuffer, dev_xDownSendBuffer, dtype, xSize)
        INIT_HALO3D_BUFFER(xDownRecvBuffer, dev_xDownRecvBuffer, dtype, xSize)

        // Alloc y axe
        INIT_HALO3D_BUFFER(yUpSendBuffer, dev_yUpSendBuffer, dtype, ySize)
        INIT_HALO3D_BUFFER(yUpRecvBuffer, dev_yUpRecvBuffer, dtype, ySize)
        INIT_HALO3D_BUFFER(yDownSendBuffer, dev_yDownSendBuffer, dtype, ySize)
        INIT_HALO3D_BUFFER(yDownRecvBuffer, dev_yDownRecvBuffer, dtype, ySize)

        // Alloc z axe
        INIT_HALO3D_BUFFER(zUpSendBuffer, dev_zUpSendBuffer, dtype, zSize)
        INIT_HALO3D_BUFFER(zUpRecvBuffer, dev_zUpRecvBuffer, dtype, zSize)
        INIT_HALO3D_BUFFER(zDownSendBuffer, dev_zDownSendBuffer, dtype, zSize)
        INIT_HALO3D_BUFFER(zDownRecvBuffer, dev_zDownRecvBuffer, dtype, zSize)

//         MPI_ALL_PRINT(fprintf(fp, "%s", STR_COLL_GIVE);)
//         STR_COLL_FREE

        cudaErrorCheck( cudaDeviceSynchronize() );
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);

        // Init send buffers (Recv buffers stay initialized as 0)
        {
            dim3 block_size(BLK_SIZE, 1, 1);
            dim3 grid_size(GRD_SIZE, 1, 1);
            init_kernel<<<grid_size, block_size>>>(xSize, dev_xUpSendBuffer, rank);
            init_kernel<<<grid_size, block_size>>>(ySize, dev_yUpSendBuffer, rank);
            init_kernel<<<grid_size, block_size>>>(zSize, dev_zUpSendBuffer, rank);
            init_kernel<<<grid_size, block_size>>>(xSize, dev_xDownSendBuffer, rank);
            init_kernel<<<grid_size, block_size>>>(ySize, dev_yDownSendBuffer, rank);
            init_kernel<<<grid_size, block_size>>>(zSize, dev_zDownSendBuffer, rank);
            cudaErrorCheck( cudaDeviceSynchronize() );
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {printf("%d#", j); fflush(stdout);}
        cudaErrorCheck( cudaDeviceSynchronize() );
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);

        /*

        Implemetantion goes here

        */

        for(int i=1-(WARM_UP); i<=loop_count; i++) {
            for (int k=0; k<3; k++) {cuda_timer[k] = 0.0; mpi_timer[k] = 0.0;}
            MPI_Barrier(MPI_COMM_WORLD);
            start_time = MPI_Wtime();

            halo3d_run_axes(xSize, xUp, xDown,
                     xUpSendBuffer, dev_xUpSendBuffer, xUpRecvBuffer, dev_xUpRecvBuffer,
                     xDownSendBuffer, dev_xDownSendBuffer, xDownRecvBuffer, dev_xDownRecvBuffer,
                     &(cuda_timer[0]), &(mpi_timer[0]), 1000);

            halo3d_run_axes(ySize, yUp, yDown,
                     yUpSendBuffer, dev_yUpSendBuffer, yUpRecvBuffer, dev_yUpRecvBuffer,
                     yDownSendBuffer, dev_yDownSendBuffer, yDownRecvBuffer, dev_yDownRecvBuffer,
                     &(cuda_timer[1]), &(mpi_timer[1]), 2000);

            halo3d_run_axes(zSize, zUp, zDown,
                     zUpSendBuffer, dev_zUpSendBuffer, zUpRecvBuffer, dev_zUpRecvBuffer,
                     zDownSendBuffer, dev_zDownSendBuffer, zDownRecvBuffer, dev_zDownRecvBuffer,
                     &(cuda_timer[2]), &(mpi_timer[2]), 3000);

            stop_time = MPI_Wtime();
            if (i>0) inner_elapsed_time[j][i-1] = stop_time - start_time;

            if (rank == 0) {printf("%%"); fflush(stdout);}

            unsigned int xCheck = check_recv_buffer(rank, 'x', xUp, dev_xUpRecvBuffer, xDown, dev_xDownRecvBuffer, xSize);
            unsigned int yCheck = check_recv_buffer(rank, 'y', yUp, dev_yUpRecvBuffer, yDown, dev_yDownRecvBuffer, ySize);
            unsigned int zCheck = check_recv_buffer(rank, 'z', zUp, dev_zUpRecvBuffer, zDown, dev_zDownRecvBuffer, zSize);
            xCheck = xCheck << 4;
            yCheck = yCheck << 2;
            halo_checks[j] |= xCheck;
            halo_checks[j] |= yCheck;
            halo_checks[j] |= zCheck;
        }
        if (rank == 0) {printf("#\n"); fflush(stdout);}



//         int tag1 = 10;
//         int tag2 = 20;
//         gpu_device_reduce(d_B, N, &gpu_check);
//         if(rank == 0){
//             MPI_Send(&my_cpu_check,   1, MPI_cktype, rank2, tag1, MPI_COMM_WORLD);
//             MPI_Recv(&recv_cpu_check, 1, MPI_cktype, rank2, tag2, MPI_COMM_WORLD, &stat);
//         } else if(rank == rank2){
//             MPI_Recv(&recv_cpu_check, 1, MPI_cktype, 0, tag1, MPI_COMM_WORLD, &stat);
//             MPI_Send(&my_cpu_check,   1, MPI_cktype, 0, tag2, MPI_COMM_WORLD);
//         }
//
//         gpu_checks[j] = gpu_check;
//         cpu_checks[j] = recv_cpu_check;

        // Free x axe
        FREE_HALO3D_BUFFER(xUpSendBuffer, dev_xUpSendBuffer)
        FREE_HALO3D_BUFFER(xUpRecvBuffer, dev_xUpRecvBuffer)
        FREE_HALO3D_BUFFER(xDownSendBuffer, dev_xDownSendBuffer)
        FREE_HALO3D_BUFFER(xDownRecvBuffer, dev_xDownRecvBuffer)

        // Free y axe
        FREE_HALO3D_BUFFER(yUpSendBuffer, dev_yUpSendBuffer)
        FREE_HALO3D_BUFFER(yUpRecvBuffer, dev_yUpRecvBuffer)
        FREE_HALO3D_BUFFER(yDownSendBuffer, dev_yDownSendBuffer)
        FREE_HALO3D_BUFFER(yDownRecvBuffer, dev_yDownRecvBuffer)

        // Free z axe
        FREE_HALO3D_BUFFER(zUpSendBuffer, dev_zUpSendBuffer)
        FREE_HALO3D_BUFFER(zUpRecvBuffer, dev_zUpRecvBuffer)
        FREE_HALO3D_BUFFER(zDownSendBuffer, dev_zDownSendBuffer)
        FREE_HALO3D_BUFFER(zDownRecvBuffer, dev_zDownRecvBuffer)
    }

    MPI_Allreduce(inner_elapsed_time, elapsed_time, BUFF_CYCLE*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int j=0; j<BUFF_CYCLE; j++) {
        long int N = 1 << j;
        long int B_in_GB = 1 << 30;
        long int num_B = 0;
        if (xUp > -1) num_B += ny * nz * N;
        if (yUp > -1) num_B += nx * nz * N;
        if (zUp > -1) num_B += nx * ny * N;
        if (xDown > -1) num_B += ny * nz * N;
        if (yDown > -1) num_B += nx * nz * N;
        if (zDown > -1) num_B += nx * ny * N;
        double num_GB = (double)num_B / (double)B_in_GB;

        double avg_time_per_transfer[BUFF_CYCLE];
        for (int i=0; i<loop_count; i++) {
            avg_time_per_transfer[j] += elapsed_time[j][i];
            if(rank == 0) printf("\tTransfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, inner_elapsed_time[j][i], num_GB/inner_elapsed_time[j][i], i);
        }
        avg_time_per_transfer[j] /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer[j], num_GB/avg_time_per_transfer[j], halo_checks[j] );
        fflush(stdout);
    }

//     char s[10000000];
//     sprintf(s, "[%d] recv_cpu_check = %u", rank, cpu_checks[0]);
//     for (int i=0; i<BUFF_CYCLE; i++) {
//         sprintf(s+strlen(s), " %10d", cpu_checks[i]);
//     }
//     sprintf(s+strlen(s), " (for Error)\n");
//     printf("%s", s);
//     fflush(stdout);
//
//     sprintf(s, "[%d] gpu_checks = %u", rank, gpu_checks[0]);
//     for (int i=0; i<BUFF_CYCLE; i++) {
//         sprintf(s+strlen(s), " %10d", gpu_checks[i]);
//     }
//     sprintf(s+strlen(s), " (for Error)\n");
//     printf("%s", s);
//     fflush(stdout);

    MPI_Finalize();
    return(0);
}
