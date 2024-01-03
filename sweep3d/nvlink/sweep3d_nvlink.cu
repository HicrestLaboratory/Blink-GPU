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

void get_position(const int rank, const int pex, const int pey, int* myX,
                  int* myY) {
  *myX = rank % pex;
  *myY = rank / pex;
}

void compute(long sleep) {
  struct timespec sleepTS;
  sleepTS.tv_sec = 0;
  sleepTS.tv_nsec = sleep;

  struct timespec remainTS;

  if (nanosleep(&sleepTS, &remainTS) == EINTR) {
    while (nanosleep(&remainTS, &remainTS) == EINTR)
      ;
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

  int pex = -1;
  int pey = -1;
  int nx = 256;
  int ny = 256;
  int nz = 100;
  int kba = 4;
  int repeats = 1;

  int vars = 1;
  long sleep = 1000;

  for (int i = 0; i < argc; ++i) {
    if (strcmp("-pex", argv[i]) == 0) {
      pex = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-pey", argv[i]) == 0) {
      pey = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-iterations", argv[i]) == 0) {
      repeats = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-nx", argv[i]) == 0) {
      nx = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-ny", argv[i]) == 0) {
      ny = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-nz", argv[i]) == 0) {
      nz = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-sleep", argv[i]) == 0) {
      sleep = atol(argv[i + 1]);
      i++;
    } else if (strcmp("-vars", argv[i]) == 0) {
      vars = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-kba", argv[i]) == 0) {
      kba = atoi(argv[i + 1]);
      i++;
    }
  }

  if (kba == 0) {
    if (me == 0) {
      fprintf(stderr,
              "K-Blocking Factor must not be zero. Please specify -kba <value "
              "> 0>\n");
    }

    exit(-1);
  }

  if (nz % kba != 0) {
    if (me == 0) {
      fprintf(stderr,
              "KBA must evenly divide NZ, KBA=%d, NZ=%d, remainder=%d (must be "
              "zero)\n",
              kba, nz, (nz % kba));
    }

    exit(-1);
  }

  if ((pex * pey) != world) {
    if (0 == me) {
      fprintf(
          stderr,
          "Error: processor decomposition (%d x %d) != number of ranks (%d)\n",
          pex, pey, world);
    }

    exit(-1);
  }

  if (me == 0) {
    printf("# Sweep3D Communication Pattern\n");
    printf("# Info:\n");
    printf("# Px:              %8d\n", pex);
    printf("# Py:              %8d\n", pey);
    printf("# Nx x Ny x Nz:    %8d x %8d x %8d\n", nx, ny, nz);
    printf("# KBA:             %8d\n", kba);
    printf("# Variables:       %8d\n", vars);
    printf("# Iterations:      %8d\n", repeats);
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

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i=0; i<world; i++) {
    if (me == i)
      printf("#\tMPI process %d has device %d\n", me, dev);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  // ----------------------------------------------------------------------------------------------

  int myX = -1;
  int myY = -1;

  get_position(me, pex, pey, &myX, &myY);

  const int xUp = (myX != (pex - 1)) ? me + 1 : -1;
  const int xDown = (myX != 0) ? me - 1 : -1;

  const int yUp = (myY != (pey - 1)) ? me + pex : -1;
  const int yDown = (myY != 0) ? me - pex : -1;

  MPI_Status status;

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

  const int xSize = nx * kba * vars, ySize = ny * kba * vars;
  dtype* xRecvBuffer = (dtype*)malloc(sizeof(dtype) * xSize);
  dtype* xSendBuffer = (dtype*)malloc(sizeof(dtype) * xSize);

  dtype* yRecvBuffer = (dtype*)malloc(sizeof(dtype) * ySize);
  dtype* ySendBuffer = (dtype*)malloc(sizeof(dtype) * ySize);

  // ---------------------------------------------------------------------------------------
  dtype *dev_xSendBuffer, *dev_xRecvBuffer, *dev_ySendBuffer, *dev_yRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_xSendBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xRecvBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_ySendBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yRecvBuffer, sizeof(dtype) * ySize) );
  // ---------------------------------------------------------------------------------------

  // ---------------------------------------
  checkCudaErrors( cudaMemset(dev_xSendBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xRecvBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_ySendBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yRecvBuffer, 0, sizeof(dtype) * ySize) );

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_ySendBuffer, me);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  // ---------------------------------------
  {
    float tmp[2][1], *tmp0;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);
    int size = (xSize > ySize) ? xSize : ySize;
    tmp0 = (dtype*)malloc(sizeof(dtype)*(size));
    for (int i=0; i<2; i++) tmp[i][0] = 0.0;
    checkCudaErrors( cudaMemcpy(tmp0, dev_xSendBuffer,   xSize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[0][0] = tmp0[x];
    checkCudaErrors( cudaMemcpy(tmp0, dev_ySendBuffer,   ySize*sizeof(float), cudaMemcpyDeviceToHost) );
    tmp[1][0] = tmp0[x];
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
      fprintf(fp, "extracted tid = %d\n", x);
      fprintf(fp, "xSendBuffer = %6.4f, ySendBuffer = %6.4f\n", tmp[0][0], tmp[1][0]);
    )
    free(tmp0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

  struct timeval start;
  struct timeval end;

  INIT_EXPS
  TIMER_DEF(0);
  SET_EXPERIMENT_NAME(0, "sweep3d")
  SET_EXPERIMENT_TYPE(0, "nvlink")
  SET_EXPERIMENT(0, "CUDA")

  SET_EXPERIMENT_NAME(1, "sweep3d")
  SET_EXPERIMENT_TYPE(1, "nvlink")
  SET_EXPERIMENT(1, "MPI")

  SET_EXPERIMENT_NAME(2, "sweep3d")
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

  dtype *xPeerBuffer, *yPeerBuffer;
  cudaEvent_t xSendEvent, ySendEvent, xRecvEvent, yRecvEvent;
  cudaIpcMemHandle_t xSendHandle, xRecvHandle, ySendHandle, yRecvHandle;
  cudaIpcEventHandle_t xSendEventHandle, xRecvEventHandle, ySendEventHandle, yRecvEventHandle;

  gettimeofday(&start, NULL);

  // We repeat this sequence twice because there are really 8 vertices in the 3D
  // data domain and we sweep from each of them, processing the top four first
  // and then the bottom four vertices next.
  for (int i = 0; i < (repeats * 2); i++) {

    // =================================================================================================================
    // Recreate communication pattern of sweep from (0,0) towards (Px,Py)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xDown > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_dtype, xDown, 1000, MPI_COMM_WORLD, &status);
      }

      if (yDown > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_dtype, yDown, 1000, MPI_COMM_WORLD, &status);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);

      // ---------------------------------------
      TIMER_START(0);
      if (xUp > -1) {
        checkCudaErrors( cudaMemcpy(dev_xRecvBuffer, xRecvBuffer, xSize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      if (yUp > -1) {
        checkCudaErrors( cudaMemcpy(dev_yRecvBuffer, yRecvBuffer, ySize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      compute(sleep);  // NOTE has sense to become GPU computation

      // ---------------------------------------
      TIMER_START(0);
      if (xDown > -1) {
        checkCudaErrors( cudaMemcpy(xSendBuffer, dev_xSendBuffer, xSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      if (yDown > -1) {
        checkCudaErrors( cudaMemcpy(ySendBuffer, dev_ySendBuffer, ySize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      TIMER_START(0);
      if (xUp > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_dtype, xUp, 1000, MPI_COMM_WORLD);
      }

      if (yUp > -1) {
        MPI_Send(ySendBuffer, (ny * kba * vars), MPI_dtype, yUp, 1000, MPI_COMM_WORLD);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);
    }
    // =================================================================================================================


    // =================================================================================================================
    // Recreate communication pattern of sweep from (Px,0) towards (0,Py)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xUp > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_dtype, xUp, 2000, MPI_COMM_WORLD, &status);
      }

      if (yDown > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_dtype, yDown, 2000, MPI_COMM_WORLD, &status);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);

      // ---------------------------------------
      TIMER_START(0);
      if (xDown > -1) {
        checkCudaErrors( cudaMemcpy(dev_xRecvBuffer, xRecvBuffer, xSize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      if (yUp > -1) {
        checkCudaErrors( cudaMemcpy(dev_yRecvBuffer, yRecvBuffer, ySize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      compute(sleep);  // NOTE has sense to become GPU computation

      // ---------------------------------------
      TIMER_START(0);
      if (xUp > -1) {
        checkCudaErrors( cudaMemcpy(xSendBuffer, dev_xSendBuffer, xSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      if (yDown > -1) {
        checkCudaErrors( cudaMemcpy(ySendBuffer, dev_ySendBuffer, ySize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      TIMER_START(0);
      if (xDown > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_dtype, xDown, 2000, MPI_COMM_WORLD);
      }

      if (yUp > -1) {
        MPI_Send(ySendBuffer, (ny * kba * vars), MPI_dtype, yUp, 2000, MPI_COMM_WORLD);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);
    }
    // =================================================================================================================


    // =================================================================================================================
    // Recreate communication pattern of sweep from (Px,Py) towards (0,0)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xUp > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_dtype, xUp, 3000, MPI_COMM_WORLD, &status);
      }

      if (yUp > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_dtype, yUp, 3000, MPI_COMM_WORLD, &status);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);

      // ---------------------------------------
      TIMER_START(0);
      if (xDown > -1) {
        checkCudaErrors( cudaMemcpy(dev_xRecvBuffer, xRecvBuffer, xSize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      if (yDown > -1) {
        checkCudaErrors( cudaMemcpy(dev_yRecvBuffer, yRecvBuffer, ySize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      compute(sleep);  // NOTE has sense to become GPU computation

      // ---------------------------------------
      TIMER_START(0);
      if (xUp > -1) {
        checkCudaErrors( cudaMemcpy(xSendBuffer, dev_xSendBuffer, xSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      if (yUp > -1) {
        checkCudaErrors( cudaMemcpy(ySendBuffer, dev_ySendBuffer, ySize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      TIMER_START(0);
      if (xDown > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_dtype, xDown, 3000, MPI_COMM_WORLD);
      }

      if (yDown > -1) {
        MPI_Send(ySendBuffer, (ny * kba * vars), MPI_dtype, yDown, 3000, MPI_COMM_WORLD);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);
    }
    // =================================================================================================================


    // =================================================================================================================
    // Recreate communication pattern of sweep from (0,Py) towards (Px,0)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xDown > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_dtype, xDown, 4000, MPI_COMM_WORLD, &status);
      }

      if (yUp > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_dtype, yUp, 4000, MPI_COMM_WORLD, &status);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);

      // ---------------------------------------
      TIMER_START(0);
      if (xUp > -1) {
        checkCudaErrors( cudaMemcpy(dev_xRecvBuffer, xRecvBuffer, xSize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      if (yDown > -1) {
        checkCudaErrors( cudaMemcpy(dev_yRecvBuffer, yRecvBuffer, ySize*sizeof(dtype), cudaMemcpyHostToDevice) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      compute(sleep);  // NOTE has sense to become GPU computation

      // ---------------------------------------
      TIMER_START(0);
      if (xDown > -1) {
        checkCudaErrors( cudaMemcpy(xSendBuffer, dev_xSendBuffer, xSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      if (yUp > -1) {
        checkCudaErrors( cudaMemcpy(ySendBuffer, dev_ySendBuffer, ySize*sizeof(dtype), cudaMemcpyDeviceToHost) );
      }

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
      // ---------------------------------------

      TIMER_START(0);
      if (xUp > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_dtype, xUp, 4000, MPI_COMM_WORLD);
      }

      if (yDown > -1) {
        MPI_Send(ySendBuffer, (ny * kba * vars), MPI_dtype, yDown, 4000, MPI_COMM_WORLD);
      }
      TIMER_STOP(0);
      timeTakenMPI += TIMER_ELAPSED(0);
    }
    // =================================================================================================================
  }

  MPI_Barrier(MPI_COMM_WORLD);
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
    // BUG to check the sense of the test
    size_t test_sizes[2], *dev_test_sizes;
    dtype *test_vector[2], **dev_test_vector;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);
    int maxSize = (xSize > ySize) ? xSize : ySize;

    dtype *dev_checkVector, *checkVector;
    checkVector = (dtype*) malloc(sizeof(dtype)*maxSize);
    checkCudaErrors( cudaMalloc(&dev_checkVector,   sizeof(dtype) * maxSize) );
    checkCudaErrors( cudaMemset(dev_checkVector, 0, sizeof(dtype) * maxSize) );

    test_sizes[0] = xSize;
    test_sizes[1] = ySize;
    test_vector[0] = dev_xRecvBuffer;
    test_vector[1] = dev_yRecvBuffer;

    checkCudaErrors( cudaMalloc(&dev_test_sizes,  sizeof(size_t) * 2) );
    checkCudaErrors( cudaMalloc(&dev_test_vector, sizeof(dtype*) * 2) );
    checkCudaErrors( cudaMemcpy(dev_test_sizes,  test_sizes,  sizeof(size_t) * 2, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_test_vector, test_vector, sizeof(dtype*) * 2, cudaMemcpyHostToDevice) );

    {
      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(GRD_SIZE, 1, 1);
      test_kernel<<<grid_size, block_size>>>(maxSize, 2, dev_test_sizes, dev_test_vector, dev_checkVector);
      checkCudaErrors( cudaDeviceSynchronize() );
    }
    checkCudaErrors( cudaMemcpy(checkVector, dev_checkVector, maxSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
      fprintf(fp, "xUp = %d, xDown = %d, yUp = %d, yDown = %d\n", xUp, xDown, yUp, yDown);
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

  const double timeTaken =
      (((double)end.tv_sec) + ((double)end.tv_usec) * 1.0e-6) -
      (((double)start.tv_sec) + ((double)start.tv_usec) * 1.0e-6);
  const double bytesXchng =
      ((double)repeats) *
      (((double)(xUp > -1 ? sizeof(dtype) * nx * kba * vars * 2 : 0)) +
       ((double)(xDown > -1 ? sizeof(dtype) * nx * kba * vars * 2 : 0)) +
       ((double)(yUp > -1 ? sizeof(dtype) * ny * kba * vars * 2 : 0)) +
       ((double)(yDown > -1 ? sizeof(dtype) * ny * kba * vars * 2 : 0)));

  if ((myX == (pex / 2)) && (myY == (pey / 2))) {
    printf("# Results from rank: %d\n", me);
    printf("# %20s %20s %20s\n", "Time", "KBytesXchng/Rank-Max", "MB/S/Rank");
    printf("  %20.6f %20.4f %20.4f\n", timeTaken, bytesXchng / 1024.0,
           (bytesXchng / 1024.0) / timeTaken);
  }

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
}
