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

#include "../../include/experiment_utils.h"
#include "../../include/debug_utils.h"
#include "../../include/helper_cuda.h"

#define BLK_SIZE 256
#define GRD_SIZE 4
#define TID_DIGITS 10000

#define dtype float
#define MPI_dtype MPI_FLOAT

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
void init_kernel(int n, dtype *input, int rank) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  dtype floattid = tid/(dtype)TID_DIGITS;
  dtype val_coord = rank + floattid;
  if (tid < n)
      input[tid] = (dtype)val_coord;

}

__global__
void test_kernel(int n, int ninputs, size_t *sizes, dtype **inputs, dtype *output) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  dtype tmp = 0.0;

  if (tid < n) {
    for (int i=0; i<ninputs; i++)
      if (tid < sizes[i])
        tmp += inputs[i][tid];
  }
  output[tid] = tmp;

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

  int requestcount = 0;
  MPI_Status* status;
  status = (MPI_Status*)malloc(sizeof(MPI_Status) * 4);

  MPI_Request* requests;
  requests = (MPI_Request*)malloc(sizeof(MPI_Request) * 4);

  dtype* xUpSendBuffer = (dtype*)malloc(sizeof(dtype) * xSize);
  dtype* xUpRecvBuffer = (dtype*)malloc(sizeof(dtype) * xSize);

  dtype* xDownSendBuffer = (dtype*)malloc(sizeof(dtype) * xSize);
  dtype* xDownRecvBuffer = (dtype*)malloc(sizeof(dtype) * xSize);

  // ---------------------------------------------------------------------------------------
  dtype *dev_xUpSendBuffer, *dev_xUpRecvBuffer, *dev_xDownSendBuffer, *dev_xDownRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_xUpSendBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xUpRecvBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xDownSendBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xDownRecvBuffer, sizeof(dtype) * xSize) );
  // ---------------------------------------------------------------------------------------

  // ---------------------------------------
//   for (int i = 0; i < xSize; i++) {
//     xUpSendBuffer[i] = i;
//     xUpRecvBuffer[i] = i;
//     xDownSendBuffer[i] = i;
//     xDownRecvBuffer[i] = i;
//   }
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  checkCudaErrors( cudaMemset(dev_xUpSendBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xUpRecvBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xDownSendBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xDownRecvBuffer, 0, sizeof(dtype) * xSize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xUpSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xDownSendBuffer, me);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  dtype* yUpSendBuffer = (dtype*)malloc(sizeof(dtype) * ySize);
  dtype* yUpRecvBuffer = (dtype*)malloc(sizeof(dtype) * ySize);

  dtype* yDownSendBuffer = (dtype*)malloc(sizeof(dtype) * ySize);
  dtype* yDownRecvBuffer = (dtype*)malloc(sizeof(dtype) * ySize);

  // ---------------------------------------------------------------------------------------
  dtype *dev_yUpSendBuffer, *dev_yUpRecvBuffer, *dev_yDownSendBuffer, *dev_yDownRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_yUpSendBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yUpRecvBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yDownSendBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yDownRecvBuffer, sizeof(dtype) * ySize) );
  // --------------------------------------------------------------------------------------

  // ---------------------------------------
//   for (int i = 0; i < ySize; i++) {
//     yUpSendBuffer[i] = i;
//     yUpRecvBuffer[i] = i;
//     yDownSendBuffer[i] = i;
//     yDownRecvBuffer[i] = i;
//   }
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  checkCudaErrors( cudaMemset(dev_yUpSendBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yUpRecvBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yDownSendBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yDownRecvBuffer, 0, sizeof(dtype) * ySize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_yUpSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_yDownSendBuffer, me);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  dtype* zUpSendBuffer = (dtype*)malloc(sizeof(dtype) * zSize);
  dtype* zUpRecvBuffer = (dtype*)malloc(sizeof(dtype) * zSize);

  dtype* zDownSendBuffer = (dtype*)malloc(sizeof(dtype) * zSize);
  dtype* zDownRecvBuffer = (dtype*)malloc(sizeof(dtype) * zSize);

  // ---------------------------------------------------------------------------------------
  dtype *dev_zUpSendBuffer, *dev_zUpRecvBuffer, *dev_zDownSendBuffer, *dev_zDownRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_zUpSendBuffer, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMalloc(&dev_zUpRecvBuffer, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMalloc(&dev_zDownSendBuffer, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMalloc(&dev_zDownRecvBuffer, sizeof(dtype) * zSize) );
  // --------------------------------------------------------------------------------------

  // ---------------------------------------
//   for (int i = 0; i < zSize; i++) {
//     zUpSendBuffer[i] = i;
//     zUpRecvBuffer[i] = i;
//     zDownSendBuffer[i] = i;
//     zDownRecvBuffer[i] = i;
//   }
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  checkCudaErrors( cudaMemset(dev_zUpSendBuffer, 0, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMemset(dev_zUpRecvBuffer, 0, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMemset(dev_zDownSendBuffer, 0, sizeof(dtype) * zSize) );
  checkCudaErrors( cudaMemset(dev_zDownRecvBuffer, 0, sizeof(dtype) * zSize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(zSize, dev_zUpSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(zSize, dev_zDownSendBuffer, me);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  // ---------------------------------------
  {
    float tmp[3][2], *tmp0;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);
    int size = (xSize > ySize) ? xSize : ySize;
    if (zSize > size) size = zSize;
    tmp0 = (dtype*)malloc(sizeof(dtype)*(size));
    for (int i=0; i<6; i++) tmp[i/2][i%2] = 0.0;
    checkCudaErrors( cudaMemcpy(tmp0, dev_xUpSendBuffer,   xSize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[0][0] = tmp0[x];
    checkCudaErrors( cudaMemcpy(tmp0, dev_xDownSendBuffer, xSize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[0][1] = tmp0[x];
    checkCudaErrors( cudaMemcpy(tmp0, dev_yUpSendBuffer,   ySize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[1][0] = tmp0[x];
    checkCudaErrors( cudaMemcpy(tmp0, dev_yDownSendBuffer, ySize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[1][1] = tmp0[x];
    checkCudaErrors( cudaMemcpy(tmp0, dev_zUpSendBuffer,   zSize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[2][0] = tmp0[x];
    checkCudaErrors( cudaMemcpy(tmp0, dev_zDownSendBuffer, zSize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[2][1] = tmp0[x];
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
      fprintf(fp, "extracted tid = %d\n", x);
      fprintf(fp, "xUpSendBuffer = %6.4f, xDownSendBuffer = %6.4f\n", tmp[0][0], tmp[0][1]);
      fprintf(fp, "yUpSendBuffer = %6.4f, yDownSendBuffer = %6.4f\n", tmp[1][0], tmp[1][1]);
      fprintf(fp, "zUpSendBuffer = %6.4f, zDownSendBuffer = %6.4f\n", tmp[2][0], tmp[2][1]);
    )
    free(tmp0);
  }
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
  SET_EXPERIMENT_TYPE(0, "baseline")
  SET_EXPERIMENT(0, "CUDA")

  SET_EXPERIMENT_NAME(1, "halo3d")
  SET_EXPERIMENT_TYPE(1, "baseline")
  SET_EXPERIMENT(1, "MPI")

  SET_EXPERIMENT_NAME(2, "halo3d")
  SET_EXPERIMENT_TYPE(2, "baseline")
  SET_EXPERIMENT(2, "TOTAL")
  gettimeofday(&start, NULL);

  for (int i = 0; i < repeats; ++i) {
    requestcount = 0;

    if (nanosleep(&sleepTS, &remainTS) == EINTR) {
      while (nanosleep(&remainTS, &remainTS) == EINTR)
        ;
    }

    // =================================================================================================================

    // ---------------------------------------
    TIMER_START(0);
    if (xUp > -1) {
      checkCudaErrors( cudaMemcpy(xUpSendBuffer, dev_xUpSendBuffer, xSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }

    if (xDown > -1) {
      checkCudaErrors( cudaMemcpy(xDownSendBuffer, dev_xDownSendBuffer, xSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    TIMER_START(0);
    if (xUp > -1) {

      MPI_Irecv(xUpRecvBuffer, xSize, MPI_dtype, xUp, 1000,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(xUpSendBuffer, xSize, MPI_dtype, xUp, 1000,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (xDown > -1) {
      MPI_Irecv(xDownRecvBuffer, xSize, MPI_dtype, xDown, 1000,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(xDownSendBuffer, xSize, MPI_dtype, xDown, 1000,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (xUp > -1) {
      checkCudaErrors( cudaMemcpy(dev_xUpRecvBuffer, xUpRecvBuffer, xSize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    if (xDown > -1) {
      checkCudaErrors( cudaMemcpy(dev_xDownRecvBuffer, xDownRecvBuffer, xSize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // =================================================================================================================

    // ---------------------------------------
    TIMER_START(0);
    if (yUp > -1) {
      checkCudaErrors( cudaMemcpy(yUpSendBuffer, dev_yUpSendBuffer, ySize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }

    if (yDown > -1) {
      checkCudaErrors( cudaMemcpy(yDownSendBuffer, dev_yDownSendBuffer, ySize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    TIMER_START(0);
    if (yUp > -1) {
      MPI_Irecv(yUpRecvBuffer, ySize, MPI_dtype, yUp, 2000,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(yUpSendBuffer, ySize, MPI_dtype, yUp, 2000,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (yDown > -1) {
      MPI_Irecv(yDownRecvBuffer, ySize, MPI_dtype, yDown, 2000,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(yDownSendBuffer, ySize, MPI_dtype, yDown, 2000,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (yUp > -1) {
      checkCudaErrors( cudaMemcpy(dev_yUpRecvBuffer, yUpRecvBuffer, ySize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    if (yDown > -1) {
      checkCudaErrors( cudaMemcpy(dev_yDownRecvBuffer, yDownRecvBuffer, ySize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // =================================================================================================================

    // ---------------------------------------
    TIMER_START(0);
    if (zUp > -1) {
      checkCudaErrors( cudaMemcpy(zUpSendBuffer, dev_zUpSendBuffer, zSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }

    if (zDown > -1) {
      checkCudaErrors( cudaMemcpy(zDownSendBuffer, dev_zDownSendBuffer, zSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    TIMER_START(0);
    if (zUp > -1) {
      MPI_Irecv(zUpRecvBuffer, zSize, MPI_dtype, zUp, 4000,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(zUpSendBuffer, zSize, MPI_dtype, zUp, 4000,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (zDown > -1) {
      MPI_Irecv(zDownRecvBuffer, zSize, MPI_dtype, zDown, 4000,
                MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(zDownSendBuffer, zSize, MPI_dtype, zDown, 4000,
                MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (zUp > -1) {
      checkCudaErrors( cudaMemcpy(dev_zUpRecvBuffer, zUpRecvBuffer, zSize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    if (zDown > -1) {
      checkCudaErrors( cudaMemcpy(dev_zDownRecvBuffer, zDownRecvBuffer, zSize*sizeof(dtype), cudaMemcpyHostToDevice) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // =================================================================================================================
  }

  gettimeofday(&end, NULL);
  ADD_TIME_EXPERIMENT(0, timeTakenCUDA)
  ADD_TIME_EXPERIMENT(1, timeTakenMPI)
  TotalTimeTaken = timeTakenCUDA + timeTakenMPI;
  ADD_TIME_EXPERIMENT(2, TotalTimeTaken)

  MPI_Barrier(MPI_COMM_WORLD);

  // ---------------------------------------
  {
    size_t test_sizes[6], *dev_test_sizes;
    dtype *test_vector[6], **dev_test_vector;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);
    int maxSize = (xSize > ySize) ? xSize : ySize;
    if (zSize > maxSize) maxSize = zSize;

    dtype *dev_checkVector, *checkVector;
    checkVector = (dtype*) malloc(sizeof(dtype)*maxSize);
    checkCudaErrors( cudaMalloc(&dev_checkVector,   sizeof(dtype) * maxSize) );
    checkCudaErrors( cudaMemset(dev_checkVector, 0, sizeof(dtype) * maxSize) );

    test_sizes[0] = xSize;
    test_sizes[1] = xSize;
    test_sizes[2] = ySize;
    test_sizes[3] = ySize;
    test_sizes[4] = zSize;
    test_sizes[5] = zSize;
    test_vector[0] = dev_xUpRecvBuffer;
    test_vector[1] = dev_xDownRecvBuffer;
    test_vector[2] = dev_yUpRecvBuffer;
    test_vector[3] = dev_yDownRecvBuffer;
    test_vector[4] = dev_zUpRecvBuffer;
    test_vector[5] = dev_zDownRecvBuffer;

    checkCudaErrors( cudaMalloc(&dev_test_sizes,  sizeof(size_t) * 6) );
    checkCudaErrors( cudaMalloc(&dev_test_vector, sizeof(dtype*) * 6) );
    checkCudaErrors( cudaMemcpy(dev_test_sizes,  test_sizes,  sizeof(size_t) * 6, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_test_vector, test_vector, sizeof(dtype*) * 6, cudaMemcpyHostToDevice) );

    {
      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(GRD_SIZE, 1, 1);
      test_kernel<<<grid_size, block_size>>>(maxSize, 6, dev_test_sizes, dev_test_vector, dev_checkVector);
      checkCudaErrors( cudaDeviceSynchronize() );
    }
    checkCudaErrors( cudaMemcpy(checkVector, dev_checkVector, maxSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
      fprintf(fp, "xUp = %d, xDown = %d, yUp = %d, yDown = %d, zUp = %d, zDown = %d\n", xUp, xDown, yUp, yDown, zUp, zDown);
      fprintf(fp, "extracted tid = %d\n", x);
      fprintf(fp, "checkVector = %6.4f\n", checkVector[x]);
    )

    checkCudaErrors( cudaFree(dev_test_vector) );
    checkCudaErrors( cudaFree(dev_checkVector) );
    checkCudaErrors( cudaFree(dev_test_sizes) );
    free(checkVector);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

  free(xUpRecvBuffer);
  free(xDownRecvBuffer);
  free(yUpRecvBuffer);
  free(yDownRecvBuffer);
  free(zUpRecvBuffer);
  free(zDownRecvBuffer);

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

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
}
