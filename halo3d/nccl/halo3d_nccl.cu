// Copyright 2009-2018 Sandia Corporation. Under the terms
// of Contract DE-NA0003525 with Sandia Corporation, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2018, Sandia Corporation
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#include <errno.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define NCCL

#include "../../include/experiment_utils.h"
#include "../../include/debug_utils.h"
#include "../../include/helper_cuda.h"

#define BLK_SIZE 256
#define GRD_SIZE 4
#define dtype float
#define MPI_dtype MPI_FLOAT

// for nccl
#include <nccl.h>
// --------

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

static int stringCmp( const void *a, const void *b) {
     return strcmp((const char*)a,(const char*)b);

}

int  assignDeviceToProcess(MPI_Comm *nodeComm, int *nnodes, int *mynodeid)
{
      char     host_name[MPI_MAX_PROCESSOR_NAME];
      char (*host_names)[MPI_MAX_PROCESSOR_NAME];

      int myrank;
      int gpu_per_node;
      int n, namelen, color, rank, nprocs;
      size_t bytes;

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

//      printf ("Assigning device %d  to process on node %s rank %d\n",*myrank,  host_name, rank );
      /* Assign device to MPI process, initialize BLAS and probe device properties */
      //cudaSetDevice(*myrank);
      return myrank;
}

__global__
void init_kernel(int n, dtype *input, int scale) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i=0; i<n; i++) {
    int val_coord = tid * scale;
    if (tid < n)
        input[tid] = (dtype)val_coord;
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int me = -1, mynode = -1;
  int world = -1, nnodes = -1;
  double timeTakenMPI = 0.0, timeTakenCUDA = 0.0, TotalTimeTaken = 0.0;

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  int pex = world;
  int pey = 1;
  int pez = 1;

  int nx = 10;
  int ny = 10;
  int nz = 10;

  int repeats = 100;
  int vars = 1;

  long sleep = 1000;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-nx") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -nx without a value.\n");
        }

        exit(-1);
      }

      nx = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-ny") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -ny without a value.\n");
        }

        exit(-1);
      }

      ny = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-nz") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -nz without a value.\n");
        }

        exit(-1);
      }

      nz = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-pex") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -pex without a value.\n");
        }

        exit(-1);
      }

      pex = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-pey") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -pey without a value.\n");
        }

        exit(-1);
      }

      pey = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-pez") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -pez without a value.\n");
        }

        exit(-1);
      }

      pez = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-iterations") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -iterations without a value.\n");
        }

        exit(-1);
      }

      repeats = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-vars") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -vars without a value.\n");
        }

        exit(-1);
      }

      vars = atoi(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "-sleep") == 0) {
      if (i == argc) {
        if (me == 0) {
          fprintf(stderr, "Error: specified -sleep without a value.\n");
        }

        exit(-1);
      }

      sleep = atol(argv[i + 1]);
      ++i;
    } else {
      if (0 == me) {
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
      }

      exit(-1);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if ((pex * pey * pez) != world) {
    if (0 == me) {
      fprintf(stderr, "Error: rank grid does not equal number of ranks.\n");
      fprintf(stderr, "%7d x %7d x %7d != %7d\n", pex, pey, pez, world);
    }

    exit(-1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (me == 0) {
    printf("# MPI Nearest Neighbor Communication\n");
    printf("# Info:\n");
    printf("# Processor Grid:         %7d x %7d x %7d\n", pex, pey, pez);
    printf("# Data Grid (per rank):   %7d x %7d x %7d\n", nx, ny, nz);
    printf("# Iterations:             %7d\n", repeats);
    printf("# Variables:              %7d\n", vars);
    printf("# Sleep:                  %7ld\n", sleep);
  }

  // ----------------------------------------------------------------------------------------------
  // PICO asign device
  int dev, deviceCount = 0;

  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (me == 0) {
    printf("#\n");
    printf("# Number of GPUs: %d\n", deviceCount);
  }

  MPI_Comm nodeComm;
  dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
  cudaSetDevice(dev);

  int mynodeid = -1, mynodesize = -1;
  MPI_Comm_rank(nodeComm, &mynodeid);
  MPI_Comm_size(nodeComm, &mynodesize);

  MPI_ALL_PRINT( fprintf(fp, "mydev is %d, mynode is %d, nnodes are %d, mynodeid is %d and mynodesize is %d\n", dev, mynode, nnodes, mynodeid, mynodesize); )

  DBG_CHECK(1)

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i=0; i<world; i++) {
    if (me == i)
      printf("#\tMPI process %d has device %d\n", me, dev);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ----------------------------------------------------------------------------------------------

  int posX, posY, posZ;
  get_position(me, pex, pey, pez, &posX, &posY, &posZ);

  int xUp = convert_position_to_rank(pex, pey, pez, posX + 1, posY, posZ);
  int xDown = convert_position_to_rank(pex, pey, pez, posX - 1, posY, posZ);
  int yUp = convert_position_to_rank(pex, pey, pez, posX, posY + 1, posZ);
  int yDown = convert_position_to_rank(pex, pey, pez, posX, posY - 1, posZ);
  int zUp = convert_position_to_rank(pex, pey, pez, posX, posY, posZ + 1);
  int zDown = convert_position_to_rank(pex, pey, pez, posX, posY, posZ - 1);

  size_t xSize = ny * nz * vars, ySize = nx * nz * vars, zSize = nx * ny * vars;

  // ---------------------------------------------------------------------------------------
  dtype *dev_xUpSendBuffer, *dev_xUpRecvBuffer, *dev_xDownSendBuffer, *dev_xDownRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_xUpSendBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xUpRecvBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xDownSendBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xDownRecvBuffer, sizeof(dtype) * xSize) );
  // ---------------------------------------------------------------------------------------

  // ---------------------------------------
  checkCudaErrors( cudaMemset(dev_xUpSendBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xUpRecvBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xDownSendBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xDownRecvBuffer, 0, sizeof(dtype) * xSize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xUpSendBuffer, me+1);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xDownSendBuffer, me+1);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  // ---------------------------------------------------------------------------------------
  dtype *dev_yUpSendBuffer, *dev_yUpRecvBuffer, *dev_yDownSendBuffer, *dev_yDownRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_yUpSendBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yUpRecvBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yDownSendBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yDownRecvBuffer, sizeof(dtype) * ySize) );
  // --------------------------------------------------------------------------------------

  // ---------------------------------------
  checkCudaErrors( cudaMemset(dev_yUpSendBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yUpRecvBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yDownSendBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yDownRecvBuffer, 0, sizeof(dtype) * ySize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_yUpSendBuffer, me+1);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_yDownSendBuffer, me+1);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  // ---------------------------------------------------------------------------------------
  dtype *dev_zUpSendBuffer, *dev_zUpRecvBuffer, *dev_zDownSendBuffer, *dev_zDownRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_zUpSendBuffer, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMalloc(&dev_zUpRecvBuffer, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMalloc(&dev_zDownSendBuffer, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMalloc(&dev_zDownRecvBuffer, sizeof(dtype) * zSize) );
  // --------------------------------------------------------------------------------------

  // ---------------------------------------
  checkCudaErrors( cudaMemset(dev_zUpSendBuffer, 0, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMemset(dev_zUpRecvBuffer, 0, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMemset(dev_zDownSendBuffer, 0, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMemset(dev_zDownRecvBuffer, 0, sizeof(dtype) * zSize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(zSize, dev_zUpSendBuffer, me+1);
    init_kernel<<<grid_size, block_size>>>(zSize, dev_zDownSendBuffer, me+1);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  // ---------------------------------------
  ncclUniqueId Id;
  ncclComm_t NCCL_COMM_WORLD, NCCL_COMM_NODE;

  ncclGroupStart();
  if (mynodeid == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
  MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, nodeComm);
  NCCLCHECK( ncclCommInitRank(&NCCL_COMM_NODE, mynodesize, Id, mynodeid) );
  ncclGroupEnd();
  MPI_Barrier(MPI_COMM_WORLD);

  ncclGroupStart();
  if (me == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
  MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  NCCLCHECK( ncclCommInitRank(&NCCL_COMM_WORLD, world, Id, me) );
  ncclGroupEnd();

  MPI_ALL_PRINT(
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

    fprintf(fp, "NCCL_COMM_WORLD: nccl size = %d, nccl rank = %d\n", nccl_w_sz, nccl_w_rk);
    fprintf(fp, "NCCL_COMM_NODE:  nccl size = %d, nccl rank = %d\n", nccl_n_sz, nccl_n_rk);
  )

  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

  struct timeval start;
  struct timeval end;

  struct timespec sleepTS;
  sleepTS.tv_sec = 0;
  sleepTS.tv_nsec = sleep;

  struct timespec remainTS;

  INIT_EXPS
  TIMER_DEF(0);
  SET_EXPERIMENT_NAME(0, "halo3d")
  SET_EXPERIMENT_TYPE(0, "nccl")
  SET_EXPERIMENT(0, "TOTAL")
  gettimeofday(&start, NULL);

  for (int i = 0; i < repeats; ++i) {

    if (nanosleep(&sleepTS, &remainTS) == EINTR) {
      while (nanosleep(&remainTS, &remainTS) == EINTR)
        ;
    }

    // =================================================================================================================

    TIMER_START(0);
    if (xUp > -1) {
      ncclGroupStart();
      ncclSend(dev_xUpSendBuffer, xSize, ncclChar, xUp, NCCL_COMM_WORLD, NULL);
      ncclRecv(dev_xUpRecvBuffer, xSize, ncclChar, xUp, NCCL_COMM_WORLD, NULL);
      ncclGroupEnd();
    }

    if (xDown > -1) {
      ncclGroupStart();
      ncclSend(dev_xDownSendBuffer, xSize, ncclChar, xDown, NCCL_COMM_WORLD, NULL);
      ncclRecv(dev_xDownRecvBuffer, xSize, ncclChar, xDown, NCCL_COMM_WORLD, NULL);
      ncclGroupEnd();
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);

    // =================================================================================================================

    TIMER_START(0);
    if (yUp > -1) {
      ncclGroupStart();
      ncclSend(dev_yUpSendBuffer, ySize, ncclChar, yUp, NCCL_COMM_WORLD, NULL);
      ncclRecv(dev_yUpRecvBuffer, ySize, ncclChar, yUp, NCCL_COMM_WORLD, NULL);
      ncclGroupEnd();
    }

    if (yDown > -1) {
      ncclGroupStart();
      ncclSend(dev_yDownSendBuffer, ySize, ncclChar, yDown, NCCL_COMM_WORLD, NULL);
      ncclRecv(dev_yDownRecvBuffer, ySize, ncclChar, yDown, NCCL_COMM_WORLD, NULL);
      ncclGroupEnd();
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);

    // =================================================================================================================

    TIMER_START(0);
    if (zUp > -1) {
      ncclGroupStart();
      ncclSend(dev_zUpSendBuffer, zSize, ncclChar, zUp, NCCL_COMM_WORLD, NULL);
      ncclRecv(dev_zUpRecvBuffer, zSize, ncclChar, zUp, NCCL_COMM_WORLD, NULL);
      ncclGroupEnd();
    }

    if (zDown > -1) {
      ncclGroupStart();
      ncclSend(dev_zDownSendBuffer, zSize, ncclChar, zDown, NCCL_COMM_WORLD, NULL);
      ncclRecv(dev_zDownRecvBuffer, zSize, ncclChar, zDown, NCCL_COMM_WORLD, NULL);
      ncclGroupEnd();
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);

    // =================================================================================================================
  }

  gettimeofday(&end, NULL);
  ADD_TIME_EXPERIMENT(0, timeTakenCUDA)

  MPI_Barrier(MPI_COMM_WORLD);

  if (convert_position_to_rank(pex, pey, pez, pex / 2, pey / 2, pez / 2) ==
      me) {
    printf("# Results from rank: %d\n", me);

    const double timeTaken =
        (((double)end.tv_sec) + ((double)end.tv_usec) * 1.0e-6) -
        (((double)start.tv_sec) + ((double)start.tv_usec) * 1.0e-6);
    const double bytesXchng =
        ((double)(xUp > -1 ? sizeof(dtype) * xSize * 2 : 0)) +
        ((double)(xDown > -1 ? sizeof(dtype) * xSize * 2 : 0)) +
        ((double)(yUp > -1 ? sizeof(dtype) * ySize * 2 : 0)) +
        ((double)(yDown > -1 ? sizeof(dtype) * ySize * 2 : 0)) +
        ((double)(zUp > -1 ? sizeof(dtype) * zSize * 2 : 0)) +
        ((double)(zDown > -1 ? sizeof(dtype) * zSize * 2 : 0));

    printf("# %20s %20s %20s\n", "Time", "KBytesXchng/Rank-Max", "MB/S/Rank");
    printf("  %20.6f %20.4f %20.4f\n", timeTaken, bytesXchng / 1024.0,
           (bytesXchng / 1024.0) / timeTaken);
  }

  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
}
