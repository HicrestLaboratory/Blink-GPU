#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#define MPI

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/prints.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#include <time.h>
#include <sys/time.h>
#include "../include/experiment_utils.h"

#define BUFF_CYCLE 24
#define LOOP_COUNT 50

#define WARM_UP 5

#include "../include/debug_utils.h"

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
                           int *repeats, int *vars, long *sleep,
                           int *flag_b, int *flag_l, int *flag_x,
                           int *loop_count, int *buff_cycle, int *fix_buff_size ) {

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
        } else if (strcmp(argv[i], "-l") == 0) {
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

        exit(-1);
        }
    }
}

// HB --> Host_buffer, DB --> Device_buffer, DT --> data type, SZ --> size
#ifdef PINNED
#define INIT_HALO3D_BUFFER(HB, DB, DT, SZ) {                \
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);         \
    cudaHostAlloc(&HB, sizeof(DT) * SZ, cudaHostAllocDefault);                    \
    cudaErrorCheck( cudaMalloc(&DB, sizeof(DT) * SZ) );     \
    cudaErrorCheck( cudaMemset(DB, 0, sizeof(DT) * SZ) );   \
    cudaErrorCheck( cudaDeviceSynchronize() );              \
}

#define FREE_HALO3D_BUFFER(HB, DB) {                \
    cudaErrorCheck( cudaFree(DB) );                 \
    cudaFreeHost(HB);                               \
}
#else
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
#endif

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

    TIMER_START(0);
    if (UpFlag > -1) {
      MPI_Irecv(dev_UpRecvBuffer, BufferSize, MPI_dtype, UpFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(dev_UpSendBuffer, BufferSize, MPI_dtype, UpFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (DownFlag > -1) {
      MPI_Irecv(dev_DownRecvBuffer, BufferSize, MPI_dtype, DownFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(dev_DownSendBuffer, BufferSize, MPI_dtype, DownFlag, tag,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    *timeTakenMPI += TIMER_ELAPSED(0);

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

    MPI_Comm nodeComm;
    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
    // print device affiniy
#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(dev);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int mynodeid = -1, mynodesize = -1;
    MPI_Comm_rank(nodeComm, &mynodeid);
    MPI_Comm_size(nodeComm, &mynodesize);

    int rank2 = size-1;
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

    // Define halo3D parameters
    int pex = size, pey = 1, pez = 1;
    int nx = 10, ny = 10, nz = 10;
    long sleep = 1000;
    int repeats = 100;
    int vars = 1;

    // Set default 3D grid
    {
        int M=0, K=0, H=0;
        int n=0, k=0, h=0;
        while ( ((size)%(1<<(n+1))) == 0 ) n++;

        k = n/3;
        h = (n - k)/2;

        printf("n = %d --> k = %d, h = %d\n", n, k, h);

        K = 1 << k;
        H = 1 << h;
        M = size / (1 << (k+h));

        printf("size = %d --> %d x %d x %d\n", size, M, H, K);

        pex = M;
        pey = H;
        pez = K;
    }

    // Read input parameters
    read_line_parameters(argc, argv, rank, &nx, &ny, &nz, &pex, &pey, &pez, &repeats, &vars, &sleep,
                                           &flag_b, &flag_l, &flag_x, &loop_count, &buff_cycle, &fix_buff_size);
    MPI_Barrier(MPI_COMM_WORLD);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;}    
    // Print message based on the flags
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

    /* -------------------------------------------------------------------------------------------
        Halo 3D Initialization
    --------------------------------------------------------------------------------------------*/

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

    SZTYPE N;
    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    double start_time, stop_time;
    double cuda_timer[3], mpi_timer[3];
    double *elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    unsigned int *halo_checks = (unsigned int*)malloc(sizeof(unsigned int)*buff_cycle);
    for(int j=fix_buff_size; j<max_j; j++){

        // Define cycle sizes
        (j!=0) ? (N <<= 1) : (N = 1);
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
            size_t maxSize = (xSize > ySize) ? xSize : ySize;
            if (zSize > maxSize) maxSize = zSize;
            size_t run_time_grid_size = (maxSize % BLK_SIZE == 0) ? (maxSize/BLK_SIZE) : ((maxSize/BLK_SIZE) +1);
            dim3 block_size(BLK_SIZE, 1, 1);
            dim3 grid_size(run_time_grid_size, 1, 1);
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
                     NULL, dev_xUpSendBuffer,   NULL, dev_xUpRecvBuffer,
                     NULL, dev_xDownSendBuffer, NULL, dev_xDownRecvBuffer,
                     &(cuda_timer[0]), &(mpi_timer[0]), 1000);

            halo3d_run_axes(ySize, yUp, yDown,
                     NULL, dev_yUpSendBuffer,   NULL, dev_yUpRecvBuffer,
                     NULL, dev_yDownSendBuffer, NULL, dev_yDownRecvBuffer,
                     &(cuda_timer[1]), &(mpi_timer[1]), 2000);

            halo3d_run_axes(zSize, zUp, zDown,
                     NULL, dev_zUpSendBuffer,   NULL, dev_zUpRecvBuffer,
                     NULL, dev_zDownSendBuffer, NULL, dev_zDownRecvBuffer,
                     &(cuda_timer[2]), &(mpi_timer[2]), 3000);

            stop_time = MPI_Wtime();
            if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

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

    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int j=fix_buff_size; j<max_j; j++) {
        (j!=0) ? (N <<= 1) : (N = 1);

        SZTYPE num_B, int_num_GB;
        double num_GB;

        if (j < 31) {
            SZTYPE B_in_GB = 1 << 30;
            num_B = sizeof(dtype)*N*((size-1)/(float)size)*2;
            num_GB = (double)num_B / (double)B_in_GB;
        } else {
            SZTYPE M = 1 << (j - 30);
            num_B = N*((size-1)/(float)size)*2*sizeof(dtype);
            num_GB = sizeof(dtype)*M*((size-1)/(float)size)*2;
        }

        if (xUp > -1) num_B += ny * nz * N;
        if (yUp > -1) num_B += nx * nz * N;
        if (zUp > -1) num_B += nx * ny * N;
        if (xDown > -1) num_B += ny * nz * N;
        if (yDown > -1) num_B += nx * nz * N;
        if (zDown > -1) num_B += nx * ny * N;

        double avg_time_per_transfer = 0.0;
        for (int i=0; i<loop_count; i++) {
            avg_time_per_transfer += elapsed_time[(j-fix_buff_size)*loop_count+i];
            if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GiB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i);
        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GiB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, halo_checks[j] );
        fflush(stdout);
    }

    free(halo_checks);
    free(elapsed_time);
    free(inner_elapsed_time);
    MPI_Finalize();
    return(0);
}
