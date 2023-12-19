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

// #define DEBUG 3
#include "../../include/debug_utils.h"

#define BLK_SIZE 256
#define GRD_SIZE 4

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
void init_kernel(int n, char *input, int scale) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i=0; i<n; i++) {
    int val_coord = tid * scale;
    if (tid < n)
        input[tid] = (char)val_coord;
  }
}

unsigned long long int check_result(int n, int myid, char* dev_recvBuffer, int recvid) {

  unsigned long long int error = 0ULL;
  char* checkBuffer = (char*)malloc(sizeof(char*) * n);

  checkCudaErrors( cudaMemcpy(checkBuffer, dev_recvBuffer, n*sizeof(char), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaDeviceSynchronize() );

  for (int i=0; i<n; i++)
    error += (unsigned long long int)((recvid+1)*i - checkBuffer[i]);

  free(checkBuffer);

  return(error);
}

#endif

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int me = -1, mynode = -1;
  int world = -1, nnodes = -1;
  double timeTaken = 0.0, timeTakenCUDA = 0.0, TotalTimeTaken = 0.0;
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

  char* sendBuffer   = (char*)malloc(sizeof(char*) * msgSize);
  char* recvBuffer   = (char*)malloc(sizeof(char*) * msgSize);

  for (int i = 0; i < msgSize; ++i) {
    sendBuffer[i]  = 0;
    recvBuffer[i]  = 0;
  }

  MPI_Status status;

  DBG_CHECK(1)

#ifdef CUDA
  char *dev_sendBuffer, *dev_recvBuffer;
  checkCudaErrors( cudaMalloc(&dev_sendBuffer, msgSize*sizeof(char)) );
  checkCudaErrors( cudaMalloc(&dev_recvBuffer, msgSize*sizeof(char)) );
  checkCudaErrors( cudaMemset(dev_sendBuffer, 0, msgSize*sizeof(char)) );
  checkCudaErrors( cudaMemset(dev_recvBuffer, 0, msgSize*sizeof(char)) );
#endif

  TIMER_DEF(0);
  TIMER_DEF(1);

  DBG_CHECK(1)

#ifdef CUDA
  dim3 block_size(BLK_SIZE, 1, 1);
  dim3 grid_size(GRD_SIZE, 1, 1);
  printf("block_size = %d, grid_size = %d, elements per thread = %f\n", block_size.x, grid_size.x, (float)msgSize/(block_size.x*grid_size.x));
  init_kernel<<<grid_size, block_size>>>(msgSize, dev_sendBuffer, me+1);
  checkCudaErrors( cudaDeviceSynchronize() );
#else
  for (int i = 0; i < msgSize; ++i)
    sendBuffer[i]  = i;
#endif


  DBG_CHECK(1)

  INIT_EXPS


  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == me) printf("# ----------------- NV LINK ------------------\n");
  SET_EXPERIMENT(3, "NV LINK")
  DBG_CHECK(4)
  interror = 0ULL;
  timeTaken = 0.0;
  timeTakenCUDA = 0.0;

#ifdef NVLINK
//   if (mynodeid == 0 || mynodeid == 1) {
    int canAccess = -1;
    if (mynodeid == 0) {
      cudaDeviceEnablePeerAccess(1,0);
      checkCudaErrors( cudaDeviceCanAccessPeer(&canAccess, mynodeid, 1) );
    } else if (mynodeid == 1) {
      cudaDeviceEnablePeerAccess(0,0);
      checkCudaErrors( cudaDeviceCanAccessPeer(&canAccess, mynodeid, 0) );
    }
    checkCudaErrors( cudaDeviceSynchronize() );


    int *my_devpointer = NULL, *recv_devpointer = NULL, *peer_devpointer = NULL, host_sendflag, host_recvflag;
    cudaIpcMemHandle_t *memHandles[mynodesize], *peerHandle = NULL;
    host_sendflag = (mynodeid+1)*10;
    host_recvflag = -1;

    MPI_ALL_PRINT( fprintf(fp, "canAccess = %d\nBEFORE: host_sendflag = %d, host_recvflag = %d\n", canAccess, host_sendflag, host_recvflag); )

    checkCudaErrors( cudaMalloc(&my_devpointer, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&recv_devpointer, sizeof(int)) );
    checkCudaErrors( cudaMemcpy(my_devpointer, &host_sendflag, sizeof(int), cudaMemcpyHostToDevice) );

    for (int i=0; i<mynodesize; i++)
      memHandles[i] = NULL;

    MPI_ALL_PRINT(
      fprintf(fp, "BEFORE: my_devpointer = %p, recv_devpointer = %p, peerHandle = %p\nmemHandles: ", my_devpointer, recv_devpointer, peerHandle);
      for (int i=0; i<mynodesize; i++)
        fprintf(fp, "%p ", memHandles[i]);
      fprintf(fp, "\n");
    )


    cudaIpcMemHandle_t memHandle;
    checkCudaErrors( cudaIpcGetMemHandle ( &memHandle, my_devpointer ) );
    MPI_Allgather(&memHandle, sizeof(memHandle), MPI_BYTE, memHandles, sizeof(memHandle), MPI_BYTE, nodeComm);

    if (mynodeid == 0) {
      peerHandle = memHandles[1];
    } else if (mynodeid == 1) {
      peerHandle = memHandles[0];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_ALL_PRINT(
      fprintf(fp, "AFTER: my_devpointer = %p, recv_devpointer = %p, peerHandle = %p\nmemHandles: ", my_devpointer, recv_devpointer, peerHandle);
      for (int i=0; i<mynodesize; i++)
        fprintf(fp, "%p ", memHandles[i]);
      fprintf(fp, "\n");
    )

    if (mynodeid == 0 || mynodeid == 1) {
      if (mynodeid == 0) {
//         checkCudaErrors( cudaMemcpyPeer(recv_devpointer, 0, peer_pointer, 1, sizeof(int)) );
        checkCudaErrors( cudaIpcOpenMemHandle ((void**)&peer_devpointer, memHandle, cudaIpcMemLazyEnablePeerAccess) );
        checkCudaErrors( cudaMemcpy(recv_devpointer, peer_devpointer, sizeof(int), cudaMemcpyDeviceToDevice) );
      }else {
//         checkCudaErrors( cudaMemcpyPeer(recv_devpointer, 1, peer_pointer, 0, sizeof(int)) );
        checkCudaErrors( cudaIpcOpenMemHandle ((void**)&peer_devpointer, memHandle, cudaIpcMemLazyEnablePeerAccess) );
        checkCudaErrors( cudaMemcpy(recv_devpointer, peer_devpointer, sizeof(int), cudaMemcpyDeviceToDevice) );
      }
      checkCudaErrors( cudaDeviceSynchronize() );
      checkCudaErrors( cudaMemcpy(&host_recvflag, recv_devpointer, sizeof(int), cudaMemcpyDeviceToHost) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    MPI_ALL_PRINT( fprintf(fp, "canAccess = %d\nAFTER: host_sendflag = %d, host_recvflag = %d\n", canAccess, host_sendflag, host_recvflag); )
//   }
#else
  if (0 == me) printf("# the NVLINK macro is disabled\n");
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == me) printf("# --------------------------------------------\n");
  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
  return 0;
}
