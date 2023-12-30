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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MPI
#define CUDA
#define NVLINK

#ifdef CUDA
#include "../../include/helper_cuda.h"
#include "../../include/experiment_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif

#ifdef NCCL
#include <nccl.h>
#endif

#ifdef GPUDIRECT

#ifdef __cplusplus
extern "C" {
#endif

#include <nv-p2p.h>

  #ifdef __cplusplus
}
#endif

#include <builtin_types.h>
// for boundary alignment requirement
#define GPU_BOUND_SHIFT   16
#define GPU_BOUND_SIZE    ((uint64_t)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET  (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK    (~GPU_BOUND_OFFSET)

#endif

#ifdef NVLINK

#include "../../include/helper_multiprocess.h"

#define MAX_DEVICES (32)

#endif

#define DEBUG 0
#include "../../include/debug_utils.h"

#define BLK_SIZE 256
#define GRD_SIZE 4
#define TID_DIGITS 10000

#define dtype float
#define MPI_dtype MPI_FLOAT

#define PINGPONG_REPEATS 1000
#define PINGPONG_MSG_SIZE 1024

#ifdef CUDA

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

#ifdef NVLINK

typedef struct shmStruct_st {
  size_t nprocesses;
  int barrier;
  int sense;
  int devices[MAX_DEVICES];
  int *canAccesPeer;
  cudaIpcMemHandle_t memHandle[MAX_DEVICES];
  cudaIpcEventHandle_t eventHandle[MAX_DEVICES];
} shmStruct;

#endif

#endif

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int me = -1, mynode = -1;
  int world = -1, nnodes = -1;
  double timeTakenMPI = 0.0, timeTakenCUDA = 0.0;
  unsigned long long int interror = 0ULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  DBG_CHECK(1)

  int msgSize = PINGPONG_MSG_SIZE;
  int repeats = PINGPONG_REPEATS;

  if (argc > 1) {
    msgSize = atoi(argv[1]);
  }

  if (argc > 2) {
    repeats = atoi(argv[2]);
  }

  if (0 == me) {
    printf("# MPI PingPong Pattern\n");
    printf("# Info:\n");
    printf("# - Total Ranks:     %10d\n", world);
    printf("# - Message Size:    %10d Bytes\n", msgSize);
    printf("# - Repeats:         %10d\n", repeats);
  }

  if (world < 2) {
    printf("No MPI is run because there are not 2 or more processors.\n");
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef CUDA
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
//   MPI_Barrier(MPI_COMM_WORLD);
//   exit(42);

  DBG_CHECK(1)

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i=0; i<world; i++) {
    if (me == i)
      printf("#\tMPI process %d has device %d\n", me, dev);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  MPI_Status status;

  DBG_CHECK(1)

#ifdef CUDA
  dtype *dev_sendBuffer, *dev_recvBuffer;
  checkCudaErrors( cudaMalloc(&dev_sendBuffer, msgSize*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_recvBuffer, msgSize*sizeof(dtype)) );
  checkCudaErrors( cudaMemset(dev_sendBuffer, 0, msgSize*sizeof(dtype)) );
  checkCudaErrors( cudaMemset(dev_recvBuffer, 0, msgSize*sizeof(dtype)) );
#endif

  DBG_CHECK(1)

#ifdef CUDA
  dim3 block_size(BLK_SIZE, 1, 1);
  dim3 grid_size(GRD_SIZE, 1, 1);
//   printf("block_size = %d, grid_size = %d, elements per thread = %f\n", block_size.x, grid_size.x, (float)msgSize/(block_size.x*grid_size.x));
  init_kernel<<<grid_size, block_size>>>(msgSize, dev_sendBuffer, me);
  checkCudaErrors( cudaDeviceSynchronize() );
#endif

  DBG_CHECK(1)

  // ---------------------------------------
  {
    float *tmp0;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);
    tmp0 = (dtype*)malloc(sizeof(dtype)*(msgSize));
    for (int i=0; i<(GRD_SIZE*BLK_SIZE); i++) tmp0[i] = 0.0;
    checkCudaErrors( cudaMemcpy(tmp0, dev_sendBuffer, msgSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
      fprintf(fp, "extracted tid = %d\n", x);
      fprintf(fp, "dev_sendBuffer = %6.4f\n", tmp0[x]);
    )
    free(tmp0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

  INIT_EXPS
  TIMER_DEF(0);
  TIMER_DEF(1);
  SET_EXPERIMENT_NAME(0, "pingpong")
  SET_EXPERIMENT_NAME(1, "pingpong")
  SET_EXPERIMENT_NAME(2, "pingpong")
  SET_EXPERIMENT_TYPE(0, "nvlink")
  SET_EXPERIMENT_TYPE(1, "nvlink")
  SET_EXPERIMENT_TYPE(2, "nvlink")
  if (nnodes > 1) {
    SET_EXPERIMENT_LAYOUT(0, "interNodes")
    SET_EXPERIMENT_LAYOUT(1, "interNodes")
    SET_EXPERIMENT_LAYOUT(2, "interNodes")
  } else {
    SET_EXPERIMENT_LAYOUT(0, "intraNode")
    SET_EXPERIMENT_LAYOUT(1, "intraNode")
    SET_EXPERIMENT_LAYOUT(2, "intraNode")
  }
  SET_EXPERIMENT(0, "CUDA")
  SET_EXPERIMENT(1, "MPI")
  SET_EXPERIMENT(2, "TOTAL")


  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == me) printf("# ----------------- NV LINK ------------------\n");
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  DBG_CHECK(4)

#ifdef NVLINK
  if (nnodes == 1) {

    // --------------------------------------------------------------
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


    dtype *peerBuffer;
    cudaEvent_t event;
    cudaIpcMemHandle_t sendHandle, recvHandle;
    cudaIpcEventHandle_t sendEventHandle, recvEventHandle;

    for (int k = 0; k < repeats; k++) {

      TIMER_START(0);
      if (me == 0 || me == world-1) {
        checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&sendHandle, dev_sendBuffer) );
        checkCudaErrors( cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess) );
        checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&sendEventHandle, event) );
      }
      MPI_Barrier(MPI_COMM_WORLD);
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

      TIMER_START(1);
      if (me == 0) {
        MPI_Send(&sendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, world-1, 0, MPI_COMM_WORLD);
        MPI_Send(&sendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, world-1, 0, MPI_COMM_WORLD);
        MPI_Recv(&recvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, world-1, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&recvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, world-1, 1, MPI_COMM_WORLD, &status);
      }
      if (me == world-1) {
        MPI_Recv(&recvHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&recvEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&sendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&sendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      TIMER_STOP(1);
      timeTakenMPI += TIMER_ELAPSED(1);

      TIMER_START(0);
      if (me == 0 || me == world-1) {
        checkCudaErrors( cudaIpcOpenMemHandle((void**)&peerBuffer, *(cudaIpcMemHandle_t*)&recvHandle, cudaIpcMemLazyEnablePeerAccess) );
        checkCudaErrors( cudaIpcOpenEventHandle(&event, *(cudaIpcEventHandle_t *)&recvEventHandle) );

        checkCudaErrors( cudaMemcpy(dev_recvBuffer, peerBuffer, sizeof(dtype)*msgSize, cudaMemcpyDeviceToDevice) );
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

      TIMER_START(0);
      if (me == 0 || me == world-1) {
        checkCudaErrors( cudaIpcCloseMemHandle(peerBuffer) );
        checkCudaErrors( cudaEventDestroy(event) );
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

    }
    ADD_TIME_EXPERIMENT(0, timeTakenCUDA);
    ADD_TIME_EXPERIMENT(1, timeTakenMPI);
    ADD_TIME_EXPERIMENT(2, timeTakenMPI + timeTakenCUDA);

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
    // --------------------------------------------------------------

  } else {
    if (0 == me) printf("The NVLINK version is only implemented for intraNode communication\n");
  }
#else
  if (0 == me) printf("# the NVLINK macro is disabled\n");
#endif

  // ---------------------------------------
  {
    size_t test_sizes[1], *dev_test_sizes;
    dtype *test_vector[1], **dev_test_vector;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);

    dtype *dev_checkVector, *checkVector;
    checkVector = (dtype*) malloc(sizeof(dtype)*msgSize);
    checkCudaErrors( cudaMalloc(&dev_checkVector,   sizeof(dtype) * msgSize) );
    checkCudaErrors( cudaMemset(dev_checkVector, 0, sizeof(dtype) * msgSize) );

    test_sizes[0] = msgSize;
    test_vector[0] = dev_recvBuffer;

    checkCudaErrors( cudaMalloc(&dev_test_sizes,  sizeof(size_t)) );
    checkCudaErrors( cudaMalloc(&dev_test_vector, sizeof(dtype*)) );
    checkCudaErrors( cudaMemcpy(dev_test_sizes,  test_sizes,  sizeof(size_t), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_test_vector, test_vector, sizeof(dtype*), cudaMemcpyHostToDevice) );

    {
      dim3 block_size(BLK_SIZE, 1, 1);
      dim3 grid_size(GRD_SIZE, 1, 1);
      test_kernel<<<grid_size, block_size>>>(msgSize, 1, dev_test_sizes, dev_test_vector, dev_checkVector);
      checkCudaErrors( cudaDeviceSynchronize() );
    }
    checkCudaErrors( cudaMemcpy(checkVector, dev_checkVector, msgSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
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

  if (0 == me) printf("# --------------------------------------------\n");
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
  return 0;
}
