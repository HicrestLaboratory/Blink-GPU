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
#define NCCL

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
  TIMER_DEF(0);
  TIMER_DEF(1);
  SET_EXPERIMENT_NAME(0, "pingpong")
  SET_EXPERIMENT_TYPE(0, "nccl")
  SET_EXPERIMENT(0, "MPI+memcpy")

  if (0 == me) printf("# ---------------- Start NCCL ----------------\n");
  MPI_Barrier(MPI_COMM_WORLD);
  DBG_CHECK(3)

#ifdef NCCL
  ncclUniqueId Id;
  ncclComm_t NCCL_COMM_WORLD, NCCL_COMM_NODE;

  ncclGroupStart();
  if (mynodeid == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
  MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, nodeComm);
  NCCLCHECK( ncclCommInitRank(&NCCL_COMM_NODE, mynodesize, Id, mynodeid) );
  ncclGroupEnd();
  DBG_CHECK(3)
  MPI_Barrier(MPI_COMM_WORLD);

  {
    int check_rk, check_size;
    ncclGroupStart();
    NCCLCHECK( ncclCommCount(NCCL_COMM_NODE, &check_size)   );
    NCCLCHECK( ncclCommUserRank(NCCL_COMM_NODE, &check_rk) );
    ncclGroupEnd();
    printf("[%d] NCCL_COMM_NODE: nccl size = %d, nccl rank = %d\n", me, check_size, check_rk);
  }
  DBG_CHECK(3)
  MPI_Barrier(MPI_COMM_WORLD);

//   NCCLCHECK( ncclCommSplit(NCCL_COMM_WORLD, mynode, (me % mynodesize), &NCCL_COMM_NODE, NULL) );
//   DBG_STOP(1)

  if (mynodeid == 0 || mynodeid == 1) {
    for (int i = 0; i < repeats; i++) {
      TIMER_START(0);
      ncclGroupStart();
      if (mynodeid == 0) {
        ncclSend(dev_sendBuffer, msgSize, ncclChar, 1, NCCL_COMM_NODE, NULL);
        ncclRecv(dev_recvBuffer, msgSize, ncclChar, 1, NCCL_COMM_NODE, NULL);
      } else if (mynodeid == 1) {
        ncclSend(dev_sendBuffer, msgSize, ncclChar, 0, NCCL_COMM_NODE, NULL);
        ncclRecv(dev_recvBuffer, msgSize, ncclChar, 0, NCCL_COMM_NODE, NULL);
      }
      ncclGroupEnd();
      TIMER_STOP(0);

      timeTaken = TIMER_ELAPSED(0);
      interror = check_result( msgSize, mynodeid, dev_recvBuffer, ((mynodeid==0)?1:0) );
      ADD_INTERROR_EXPERIMENT(0, interror);
      ADD_TIME_EXPERIMENT(0, timeTaken);
    }
    DBG_CHECK(3)
  }

  MPI_Barrier(MPI_COMM_WORLD);
  SET_EXPERIMENT_NAME(1, "pingpong")
  SET_EXPERIMENT_TYPE(1, "nccl")
  SET_EXPERIMENT(1, "l2NCCL")

  if (0 == me) printf("2node layout...\n");

  interror = 0ULL;
  timeTaken = 0.0;
  timeTakenCUDA = 0.0;

  ncclGroupStart();
  if (me == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
  MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  NCCLCHECK( ncclCommInitRank(&NCCL_COMM_WORLD, world, Id, me) );
  ncclGroupEnd();
  DBG_CHECK(3)

  {
    int check_rk, check_size;
    ncclGroupStart();
    NCCLCHECK( ncclCommCount(NCCL_COMM_WORLD, &check_size)   );
    NCCLCHECK( ncclCommUserRank(NCCL_COMM_WORLD, &check_rk) );
    ncclGroupEnd();
    printf("[%d] NCCL_COMM_WORLD: nccl size = %d, nccl rank = %d\n", me, check_size, check_rk);
  }
  DBG_CHECK(3)
  MPI_Barrier(MPI_COMM_WORLD);

  if (me == 0 || me == 4) {
    for (int i = 0; i < repeats; i++) {
      TIMER_START(0);
      ncclGroupStart();
      if (me == 0) {
        ncclSend(dev_sendBuffer, msgSize, ncclChar, 4, NCCL_COMM_WORLD, NULL);
        ncclRecv(dev_recvBuffer, msgSize, ncclChar, 4, NCCL_COMM_WORLD, NULL);
      } else if (me == 4) {
        ncclSend(dev_sendBuffer, msgSize, ncclChar, 0, NCCL_COMM_WORLD, NULL);
        ncclRecv(dev_recvBuffer, msgSize, ncclChar, 0, NCCL_COMM_WORLD, NULL);
      }
      ncclGroupEnd();
      TIMER_STOP(0);

      timeTaken = TIMER_ELAPSED(0);
      interror = check_result( msgSize, mynodeid, dev_recvBuffer, ((mynodeid==0)?1:0) );
      ADD_INTERROR_EXPERIMENT(1, interror);
      ADD_TIME_EXPERIMENT(1, timeTaken);
    }
    DBG_CHECK(3)
  }
  MPI_Barrier(MPI_COMM_WORLD);

#endif

  if (0 == me) printf("# --------------------------------------------\n");
  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
  return 0;
}
