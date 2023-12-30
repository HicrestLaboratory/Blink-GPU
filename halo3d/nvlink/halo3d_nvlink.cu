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
  fflush(stdout);
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
  status = (MPI_Status*)malloc(sizeof(MPI_Status) * 4 * 2);

  MPI_Request* requests;
  requests = (MPI_Request*)malloc(sizeof(MPI_Request) * 4 * 2);

  // ---------------------------------------
  // PICO enable peer access
  STR_COLL_DEF
  STR_COLL_INIT

  // Pick all the devices that can access each other's memory for this test
  // Keep in mind that CUDA has minimal support for fork() without a
  // corresponding exec() in the child process, but in this case our
  // spawnProcess will always exec, so no need to worry.
  cudaDeviceProp prop;
  int allPeers = 1, myIPC = 1, allIPC;
  checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

  int* canAccesPeer = (int*) malloc(sizeof(int)*deviceCount*deviceCount);
  for (int i = 0; i < deviceCount*deviceCount; i++) canAccesPeer[i] = 0;

  // CUDA IPC is only supported on devices with unified addressing
  if (!prop.unifiedAddressing) {
    STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d does not support unified addressing, skipping...\n", dev); )
    myIPC = 0;
  } else {
    STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d support unified addressing\n", dev); )
  }
  // This sample requires two processes accessing each device, so we need
  // to ensure exclusive or prohibited mode is not set
  if (prop.computeMode != cudaComputeModeDefault) {
    STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d is in an unsupported compute mode for this sample\n", dev); )
    myIPC = 0;
  } else {
    STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d is in a supported compute mode for this sample\n", dev); )
  }

  MPI_Allreduce(&myIPC, &allIPC, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allIPC) {
    MPI_ALL_PRINT( fprintf(fp, "%s", STR_COLL_GIVE); )
    exit(__LINE__);
  }

  if (me == 0) {
    for (int i = 0; i < deviceCount; i++) {
      for (int j = 0; j < deviceCount; j++) {
        if (j != i) {
          int canAccessPeerIJ, canAccessPeerJI;
          checkCudaErrors( cudaDeviceCanAccessPeer(&canAccessPeerJI, j, i) );
          checkCudaErrors( cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, j) );

          canAccesPeer[i * deviceCount + j] = (canAccessPeerIJ) ? 1 : 0;
          canAccesPeer[j * deviceCount + i] = (canAccessPeerJI) ? 1 : 0;
          if (!canAccessPeerIJ || !canAccessPeerJI) allPeers = 0;
        } else {
          canAccesPeer[i * deviceCount + j] = -1;
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&allPeers, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(canAccesPeer, deviceCount*deviceCount, MPI_INT, 0, MPI_COMM_WORLD);

  if (allPeers) {
    // Enable peers here.  This isn't necessary for IPC, but it will
    // setup the peers for the device.  For systems that only allow 8
    // peers per GPU at a time, this acts to remove devices from CanAccessPeer
    for (int j = 0; j < deviceCount; j++) {
      if (j != dev) {
        checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
        STR_COLL_APPEND( sprintf(str_coll.buff, "Enabled access from device %d to device %d\n", dev, j); )
      }
    }
  } else {
    if (me == 0) printf(str_coll.buff, "CUDA IPC is not supported by all the node's GPUs\n");
  }

  MPI_ALL_PRINT(
    fprintf(fp, "%s", STR_COLL_GIVE);
    FPRINT_MATRIX(fp, canAccesPeer, deviceCount, deviceCount)
  )
  STR_COLL_FREE
  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

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
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xUpSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xDownSendBuffer, me);
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
    init_kernel<<<grid_size, block_size>>>(ySize, dev_yUpSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_yDownSendBuffer, me);
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
  SET_EXPERIMENT_TYPE(0, "nvlink")
  SET_EXPERIMENT(0, "CUDA")

  SET_EXPERIMENT_NAME(1, "halo3d")
  SET_EXPERIMENT_TYPE(1, "nvlink")
  SET_EXPERIMENT(1, "MPI")

  SET_EXPERIMENT_NAME(2, "halo3d")
  SET_EXPERIMENT_TYPE(2, "nvlink")
  SET_EXPERIMENT(2, "TOTAL")

  if (nnodes > 1) {
    SET_EXPERIMENT_LAYOUT(0, "interNodes")
    SET_EXPERIMENT_LAYOUT(1, "interNodes")
    SET_EXPERIMENT_LAYOUT(2, "interNodes")
  } else {
    SET_EXPERIMENT_LAYOUT(0, "intraNode")
    SET_EXPERIMENT_LAYOUT(1, "intraNode")
    SET_EXPERIMENT_LAYOUT(2, "intraNode")
  }


  if (nnodes != 1) {
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) printf("The NVLINK version is only implemented for intraNode communication\n");
    if (me == 0) PRINT_EXPARIMENT_STATS
    MPI_Barrier(MPI_COMM_WORLD);
    exit(__LINE__);
  }

  dtype *xUpPeerBuffer, *xDownPeerBuffer;
  cudaEvent_t xUpSendEvent, xDownSendEvent, xUpRecvEvent, xDownRecvEvent;
  cudaIpcMemHandle_t xUpSendHandle, xUpRecvHandle, xDownSendHandle, xDownRecvHandle;
  cudaIpcEventHandle_t xUpSendEventHandle, xUpRecvEventHandle, xDownSendEventHandle, xDownRecvEventHandle;

  dtype *yUpPeerBuffer, *yDownPeerBuffer;
  cudaEvent_t yUpSendEvent, yDownSendEvent, yUpRecvEvent, yDownRecvEvent;
  cudaIpcMemHandle_t yUpSendHandle, yUpRecvHandle, yDownSendHandle, yDownRecvHandle;
  cudaIpcEventHandle_t yUpSendEventHandle, yUpRecvEventHandle, yDownSendEventHandle, yDownRecvEventHandle;

  dtype *zUpPeerBuffer, *zDownPeerBuffer;
  cudaEvent_t zUpSendEvent, zDownSendEvent, zUpRecvEvent, zDownRecvEvent;
  cudaIpcMemHandle_t zUpSendHandle, zUpRecvHandle, zDownSendHandle, zDownRecvHandle;
  cudaIpcEventHandle_t zUpSendEventHandle, zUpRecvEventHandle, zDownSendEventHandle, zDownRecvEventHandle;

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
      checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&xUpSendHandle, dev_xUpSendBuffer) );
      checkCudaErrors( cudaEventCreate(&xUpSendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
      checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&xUpSendEventHandle, xUpSendEvent) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (xDown > -1) {
      checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&xDownSendHandle, dev_xDownSendBuffer) );
      checkCudaErrors( cudaEventCreate(&xDownSendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
      checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&xDownSendEventHandle, xDownSendEvent) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    TIMER_START(0);
    if (xUp > -1) {
      MPI_Irecv(&xUpRecvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, xUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Irecv(&xUpRecvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, xUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&xUpSendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, xUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&xUpSendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, xUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (xDown > -1) {
      MPI_Irecv(&xDownRecvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, xDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Irecv(&xDownRecvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, xDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&xDownSendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, xDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&xDownSendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, xDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (xUp > -1) {
      checkCudaErrors( cudaIpcOpenMemHandle((void**)&xUpPeerBuffer, *(cudaIpcMemHandle_t*)&xUpRecvHandle, cudaIpcMemLazyEnablePeerAccess) );
      checkCudaErrors( cudaIpcOpenEventHandle(&xUpRecvEvent, *(cudaIpcEventHandle_t *)&xUpRecvEventHandle) );

      checkCudaErrors( cudaMemcpy(dev_xUpRecvBuffer, xUpPeerBuffer, sizeof(dtype)*xSize, cudaMemcpyDeviceToDevice) );
    }
    if (xDown > -1) {
      checkCudaErrors( cudaIpcOpenMemHandle((void**)&xDownPeerBuffer, *(cudaIpcMemHandle_t*)&xDownRecvHandle, cudaIpcMemLazyEnablePeerAccess) );
      checkCudaErrors( cudaIpcOpenEventHandle(&xDownRecvEvent, *(cudaIpcEventHandle_t *)&xDownRecvEventHandle) );

      checkCudaErrors( cudaMemcpy(dev_xDownRecvBuffer, xDownPeerBuffer, sizeof(dtype)*xSize, cudaMemcpyDeviceToDevice) );
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // ---------------------------------------
    TIMER_START(0);
    if (xUp > -1) {
      checkCudaErrors( cudaIpcCloseMemHandle(xUpPeerBuffer) );
      checkCudaErrors( cudaEventDestroy(xUpRecvEvent) );
    }
    if (xDown > -1) {
      checkCudaErrors( cudaIpcCloseMemHandle(xDownPeerBuffer) );
      checkCudaErrors( cudaEventDestroy(xDownRecvEvent) );
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // =================================================================================================================

    // ---------------------------------------
    TIMER_START(0);
    if (yUp > -1) {
      checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&yUpSendHandle, dev_yUpSendBuffer) );
      checkCudaErrors( cudaEventCreate(&yUpSendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
      checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&yUpSendEventHandle, yUpSendEvent) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (yDown > -1) {
      checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&yDownSendHandle, dev_yDownSendBuffer) );
      checkCudaErrors( cudaEventCreate(&yDownSendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
      checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&yDownSendEventHandle, yDownSendEvent) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    TIMER_START(0);
    if (yUp > -1) {
      MPI_Irecv(&yUpRecvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, yUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Irecv(&yUpRecvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, yUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&yUpSendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, yUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&yUpSendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, yUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (yDown > -1) {
      MPI_Irecv(&yDownRecvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, yDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Irecv(&yDownRecvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, yDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&yDownSendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, yDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&yDownSendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, yDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (yUp > -1) {
      checkCudaErrors( cudaIpcOpenMemHandle((void**)&yUpPeerBuffer, *(cudaIpcMemHandle_t*)&yUpRecvHandle, cudaIpcMemLazyEnablePeerAccess) );
      checkCudaErrors( cudaIpcOpenEventHandle(&yUpRecvEvent, *(cudaIpcEventHandle_t *)&yUpRecvEventHandle) );

      checkCudaErrors( cudaMemcpy(dev_yUpRecvBuffer, yUpPeerBuffer, sizeof(dtype)*ySize, cudaMemcpyDeviceToDevice) );
    }
    if (yDown > -1) {
      checkCudaErrors( cudaIpcOpenMemHandle((void**)&yDownPeerBuffer, *(cudaIpcMemHandle_t*)&yDownRecvHandle, cudaIpcMemLazyEnablePeerAccess) );
      checkCudaErrors( cudaIpcOpenEventHandle(&yDownRecvEvent, *(cudaIpcEventHandle_t *)&yDownRecvEventHandle) );

      checkCudaErrors( cudaMemcpy(dev_yDownRecvBuffer, yDownPeerBuffer, sizeof(dtype)*ySize, cudaMemcpyDeviceToDevice) );
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // ---------------------------------------
    TIMER_START(0);
    if (yUp > -1) {
      checkCudaErrors( cudaIpcCloseMemHandle(yUpPeerBuffer) );
      checkCudaErrors( cudaEventDestroy(yUpRecvEvent) );
    }
    if (yDown > -1) {
      checkCudaErrors( cudaIpcCloseMemHandle(yDownPeerBuffer) );
      checkCudaErrors( cudaEventDestroy(yDownRecvEvent) );
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // =================================================================================================================

    // ---------------------------------------
    TIMER_START(0);
    if (zUp > -1) {
      checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&zUpSendHandle, dev_zUpSendBuffer) );
      checkCudaErrors( cudaEventCreate(&zUpSendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
      checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&zUpSendEventHandle, zUpSendEvent) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (zDown > -1) {
      checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&zDownSendHandle, dev_zDownSendBuffer) );
      checkCudaErrors( cudaEventCreate(&zDownSendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
      checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&zDownSendEventHandle, zDownSendEvent) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    TIMER_START(0);
    if (zUp > -1) {
      MPI_Irecv(&zUpRecvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, zUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Irecv(&zUpRecvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, zUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&zUpSendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, zUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&zUpSendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, zUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
    }

    if (zDown > -1) {
      MPI_Irecv(&zDownRecvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, zDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Irecv(&zDownRecvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, zDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&zDownSendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, zDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
      MPI_Isend(&zDownSendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, zDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
    }

    MPI_Waitall(requestcount, requests, status);
    requestcount = 0;
    TIMER_STOP(0);
    timeTakenMPI += TIMER_ELAPSED(0);

    // ---------------------------------------
    TIMER_START(0);
    if (zUp > -1) {
      checkCudaErrors( cudaIpcOpenMemHandle((void**)&zUpPeerBuffer, *(cudaIpcMemHandle_t*)&zUpRecvHandle, cudaIpcMemLazyEnablePeerAccess) );
      checkCudaErrors( cudaIpcOpenEventHandle(&zUpRecvEvent, *(cudaIpcEventHandle_t *)&zUpRecvEventHandle) );

      checkCudaErrors( cudaMemcpy(dev_zUpRecvBuffer, zUpPeerBuffer, sizeof(dtype)*zSize, cudaMemcpyDeviceToDevice) );
    }
    if (zDown > -1) {
      checkCudaErrors( cudaIpcOpenMemHandle((void**)&zDownPeerBuffer, *(cudaIpcMemHandle_t*)&zDownRecvHandle, cudaIpcMemLazyEnablePeerAccess) );
      checkCudaErrors( cudaIpcOpenEventHandle(&zDownRecvEvent, *(cudaIpcEventHandle_t *)&zDownRecvEventHandle) );

      checkCudaErrors( cudaMemcpy(dev_zDownRecvBuffer, zDownPeerBuffer, sizeof(dtype)*zSize, cudaMemcpyDeviceToDevice) );
    }
    TIMER_STOP(0);
    timeTakenCUDA += TIMER_ELAPSED(0);
    // ---------------------------------------

    // ---------------------------------------
    TIMER_START(0);
    if (zUp > -1) {
      checkCudaErrors( cudaIpcCloseMemHandle(zUpPeerBuffer) );
      checkCudaErrors( cudaEventDestroy(zUpRecvEvent) );
    }
    if (zDown > -1) {
      checkCudaErrors( cudaIpcCloseMemHandle(zDownPeerBuffer) );
      checkCudaErrors( cudaEventDestroy(zDownRecvEvent) );
    }
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

  // ---------------------------------------
  // PICO disable peer access
  MPI_Barrier(MPI_COMM_WORLD);
  for (int j = 0; j < deviceCount; j++) {
    if (j != dev) {
      checkCudaErrors(cudaDeviceDisablePeerAccess(j));
      printf("[%d] Disable access from device %d to device %d\n", me, dev, j);
    }
  }
  // ---------------------------------------

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

  if (convert_position_to_rank(pex, pey, pez, pex / 2, pey / 2, pez / 2) == me) {
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
