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
// #define NCCL
// #define GPUDIRECT
#define NVLINK

#ifdef CUDA
#include "../include/helper_cuda.h"
#include "../include/experiment_utils.h"
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
#include "../include/debug_utils.h"

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

  if (0 == me) {
    printf("# Beginning benchmarking...\n");
    printf("# ------------- Start MPI+memcpy -------------\n");
  }
  SET_EXPERIMENT(0, "MPI+memcpy")
  MPI_Barrier(MPI_COMM_WORLD);

  if (me < 2) {

    for (int i = 0; i < repeats; ++i) {

#ifdef CUDA
      TIMER_START(1);
      checkCudaErrors( cudaMemcpy(sendBuffer, dev_sendBuffer, msgSize*sizeof(char), cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(1);

      timeTakenCUDA += TIMER_ELAPSED(1);
#endif

      TIMER_START(0);
      if (0 == me) {
        MPI_Send(sendBuffer, msgSize, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(recvBuffer, msgSize, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
      } else {
        MPI_Recv(recvBuffer, msgSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(sendBuffer, msgSize, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
      TIMER_STOP(0);

      timeTaken += TIMER_ELAPSED(0);

#ifdef CUDA
      TIMER_START(1);
      checkCudaErrors( cudaMemcpy(dev_recvBuffer, recvBuffer, msgSize*sizeof(char), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(1);

      timeTakenCUDA += TIMER_ELAPSED(1);
      interror = check_result( msgSize, me, dev_recvBuffer, ((me==0)?1:0) );
      ADD_INTERROR_EXPERIMENT(0, interror);

      checkCudaErrors( cudaMemset(dev_recvBuffer, 0, msgSize*sizeof(char)) );
#endif

    }

    DBG_CHECK(1)

    if (0 == me) {
      printf("# Statistics:\n");

      const double bytesXchng = ((double)msgSize) * 2.0 * ((double)repeats);
      const double MbytesXchng = bytesXchng / (1024.0 * 1024.0);
      const double msgsXchng = ((double)repeats) * 2.0;
      const double KMsgsXchng = msgsXchng / 1000.0;

      printf("#%10s %9s %11s %17s %14s %16s %14s\n", "Type", "MsgSize", "Time", "KMsgs",
             "MB", "KMsg/S", "MB/S");
      printf("%10s  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", "MPI", (double)msgSize,
             timeTaken, KMsgsXchng, MbytesXchng, KMsgsXchng / timeTaken,
             MbytesXchng / timeTaken);
#ifdef CUDA
      printf("%10s  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", "memcpy", (double)msgSize,
             timeTakenCUDA, KMsgsXchng, MbytesXchng, KMsgsXchng / timeTakenCUDA,
             MbytesXchng / timeTakenCUDA);
      TotalTimeTaken = timeTaken + timeTakenCUDA;
      printf("%10s  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", "MPI+memcpy", (double)msgSize,
             TotalTimeTaken, KMsgsXchng, MbytesXchng, KMsgsXchng / TotalTimeTaken,
             MbytesXchng / TotalTimeTaken);
#endif
      ADD_TIME_EXPERIMENT(0, TotalTimeTaken)
    }
  }

  DBG_CHECK(1)

  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == me) printf("2node layout...\n");
  SET_EXPERIMENT(5, "l2MPI+memc")
  interror = 0ULL;
  timeTaken = 0.0;
  timeTakenCUDA = 0.0;

  if (nnodes > 1 && mynode < 2 && mynodeid == 0) {
    DBG_CHECK(1)
    printf("[%d] myid = %d\n", __LINE__, me);
    fflush(stdout);

    for (int i = 0; i < repeats; ++i) {

#ifdef CUDA
      TIMER_START(1);
      checkCudaErrors( cudaMemcpy(sendBuffer, dev_sendBuffer, msgSize*sizeof(char), cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(1);

      timeTakenCUDA += TIMER_ELAPSED(1);
#endif

      TIMER_START(0);
      if (0 == me) {
        MPI_Send(sendBuffer, msgSize, MPI_CHAR, mynodesize, 0, MPI_COMM_WORLD);
        MPI_Recv(recvBuffer, msgSize, MPI_CHAR, mynodesize, 1, MPI_COMM_WORLD, &status);
      } else {
        MPI_Recv(recvBuffer, msgSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(sendBuffer, msgSize, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
      TIMER_STOP(0);

      timeTaken += TIMER_ELAPSED(0);

#ifdef CUDA
      TIMER_START(1);
      checkCudaErrors( cudaMemcpy(dev_recvBuffer, recvBuffer, msgSize*sizeof(char), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(1);

      timeTakenCUDA += TIMER_ELAPSED(1);
      interror = check_result( msgSize, me, dev_recvBuffer, ((me==0)?1:0) );
      ADD_INTERROR_EXPERIMENT(5, interror);

      checkCudaErrors( cudaMemset(dev_recvBuffer, 0, msgSize*sizeof(char)) );
#endif

    }

    DBG_CHECK(1)

    if (0 == me) {
      printf("# Statistics:\n");

      const double bytesXchng = ((double)msgSize) * 2.0 * ((double)repeats);
      const double MbytesXchng = bytesXchng / (1024.0 * 1024.0);
      const double msgsXchng = ((double)repeats) * 2.0;
      const double KMsgsXchng = msgsXchng / 1000.0;

      printf("#%10s %9s %11s %17s %14s %16s %14s\n", "Type", "MsgSize", "Time", "KMsgs",
             "MB", "KMsg/S", "MB/S");
      printf("%10s  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", "MPI", (double)msgSize,
             timeTaken, KMsgsXchng, MbytesXchng, KMsgsXchng / timeTaken,
             MbytesXchng / timeTaken);
#ifdef CUDA
      printf("%10s  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", "memcpy", (double)msgSize,
             timeTakenCUDA, KMsgsXchng, MbytesXchng, KMsgsXchng / timeTakenCUDA,
             MbytesXchng / timeTakenCUDA);
      TotalTimeTaken = timeTaken + timeTakenCUDA;
      printf("%10s  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", "MPI+memcpy", (double)msgSize,
             TotalTimeTaken, KMsgsXchng, MbytesXchng, KMsgsXchng / TotalTimeTaken,
             MbytesXchng / TotalTimeTaken);
#endif
      ADD_TIME_EXPERIMENT(5, TotalTimeTaken)
    }
  }

  free(sendBuffer);
  free(recvBuffer);
  DBG_CHECK(1)

  if (0 == me) printf("# ---------------- Start NCCL ----------------\n");
  SET_EXPERIMENT(1, "NCCL")
  MPI_Barrier(MPI_COMM_WORLD);
  DBG_CHECK(3)

#ifdef NCCL
  interror = 0ULL;
  timeTaken = 0.0;
  checkCudaErrors( cudaMemset(dev_sendBuffer, 0, msgSize*sizeof(char)) );
  checkCudaErrors( cudaMemset(dev_recvBuffer, 0, msgSize*sizeof(char)) );
  init_kernel<<<grid_size, block_size>>>(msgSize, dev_sendBuffer, me+1);
  checkCudaErrors( cudaDeviceSynchronize() );
  DBG_CHECK(3)

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
      ADD_INTERROR_EXPERIMENT(1, interror);
      ADD_TIME_EXPERIMENT(1, timeTaken);
    }
    DBG_CHECK(3)
  }

  if (0 == me) printf("2node layout...\n");
  SET_EXPERIMENT(6, "l2NCCL")
  MPI_Barrier(MPI_COMM_WORLD);
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
      ADD_INTERROR_EXPERIMENT(6, interror);
      ADD_TIME_EXPERIMENT(6, timeTaken);
    }
    DBG_CHECK(3)
  }
  MPI_Barrier(MPI_COMM_WORLD);

#endif

  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == me) printf("# ---------------- GPU Direct ----------------\n");
  SET_EXPERIMENT(2, "GPUDirect")
  DBG_CHECK(1)

#ifdef GPUDIRECT

  interror = 0ULL;
  timeTaken = 0.0;
  CUdeviceptr devptrSend, devptrRecv;
  checkCudaResult( cuMemAlloc ( &devptrSend, msgSize*sizeof(char) ) );
  checkCudaResult( cuMemAlloc ( &devptrRecv, msgSize*sizeof(char) ) );

  checkCudaErrors( cudaFree(dev_sendBuffer) );
  checkCudaErrors( cudaFree(dev_recvBuffer) );
  dev_sendBuffer = (char*) devptrSend;
  dev_recvBuffer = (char*) devptrRecv;

  init_kernel<<<grid_size, block_size>>>(msgSize, dev_sendBuffer, me+1);
  checkCudaErrors( cudaDeviceSynchronize() );
  DBG_CHECK(2)


  unsigned int flag = 1;
  checkCudaResult( cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, devptrSend) );
  DBG_CHECK(2)

  nvidia_p2p_page_table *page_table;
  // do proper alignment, as required by NVIDIA kernel driver
  uint64_t virt_start = devptrSend & GPU_BOUND_MASK;
  size_t pin_size = devptrSend + msgSize*sizeof(char) - virt_start;
  if (msgSize == 0)
      return -EINVAL;
  int ret = nvidia_p2p_get_pages(0, 0, virt_start, pin_size, &page_table, NULL, dev_sendBuffer);
  if (ret == 0) {
      printf("Succesfully pinned, page_table can be accessed");
  } else {
      fprintf(stderr, "Pinning failed");
      exit(42);
  }
  DBG_CHECK(2)

#else
  if (0 == me) printf("# the GPUDIRECT macro is disabled\n");
#endif


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
