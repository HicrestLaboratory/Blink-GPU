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

#define DEBUG 2

#include "../../include/experiment_utils.h"
#include "../../include/debug_utils.h"
#include "../../include/helper_cuda.h"

#define BLK_SIZE 256
#define GRD_SIZE 4
#define TID_DIGITS 10000

#define dtype float
#define NCCL_dtype ncclFloat

// for nccl
#include <nccl.h>
// --------

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
void GPU_compute (int n, dtype *input, dtype *output) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < n)
    if (input[tid] > output[tid])
      output[tid] = input[tid];
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
  double timeTakenMPI = 0.0, timeTakenCUDA = 0.0, TotalTimeTaken = 0.0, timeTakenCOMPUTE = 0.0;

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  // ------------------------------------------ DBG print ------------------------------------------

  DBG_PRINT(1, MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("%s\n", COLOUR(AC_RED, CHECKPOINTS)); ) )

  DBG_PRINT(2, MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("%s\n", COLOUR(AC_RED, ALL_RESULT_VECTORS)); ) )

  // -----------------------------------------------------------------------------------------------

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


  const int xSize = nx * kba * vars, ySize = ny * kba * vars;

  // ---------------------------------------------------------------------------------------
  dtype *dev_xSendBuffer, *dev_xRecvBuffer, *dev_ySendBuffer, *dev_yRecvBuffer;
  checkCudaErrors( cudaMalloc(&dev_xSendBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_xRecvBuffer, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMalloc(&dev_ySendBuffer, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMalloc(&dev_yRecvBuffer, sizeof(dtype) * ySize) );

  dtype *dev_xResultBuffer[4], *dev_yResultBuffer[4];
  for(int i=0; i<4; i++) {
    checkCudaErrors( cudaMalloc(&dev_xResultBuffer[i], sizeof(dtype) * xSize) );
    checkCudaErrors( cudaMalloc(&dev_yResultBuffer[i], sizeof(dtype) * ySize) );
  }
  // ---------------------------------------------------------------------------------------

  // ---------------------------------------
  checkCudaErrors( cudaMemset(dev_xSendBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_xRecvBuffer, 0, sizeof(dtype) * xSize) );
  checkCudaErrors( cudaMemset(dev_ySendBuffer, 0, sizeof(dtype) * ySize) );
  checkCudaErrors( cudaMemset(dev_yRecvBuffer, 0, sizeof(dtype) * ySize) );

  for(int i=0; i<4; i++) {
    checkCudaErrors( cudaMemset(dev_xResultBuffer[i], 0, sizeof(dtype) * xSize) );
    checkCudaErrors( cudaMemset(dev_yResultBuffer[i], 0, sizeof(dtype) * ySize) );
  }

  {
    dim3 block_size(BLK_SIZE, 1, 1);
    dim3 grid_size(GRD_SIZE, 1, 1);
    init_kernel<<<grid_size, block_size>>>(xSize, dev_xSendBuffer, me);
    init_kernel<<<grid_size, block_size>>>(ySize, dev_ySendBuffer, me);
    checkCudaErrors( cudaDeviceSynchronize() );
  }
  // ---------------------------------------

  // --------------------- Check Input Init ---------------------
  {
    DEF_DVC
    INIT_DVC

    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);

    APPEND_DVC(dev_xSendBuffer, xSize, "dev_xSendBuffer")
    APPEND_DVC(dev_ySendBuffer, ySize, "dev_ySendBuffer")

    STR_COLL_DEF
    STR_COLL_INIT
    STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "extracted tid = %d\n", x); )
    for (int i=0; i<DVC_LEN; i+=2) {
      DVC_TOCPU(i)
      STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "%s = %6.4f, ", DVC_CPUBUFFNAM, DVC_CPUBUFF[x]); )
      DVC_TOCPU(i+1)
      STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "%s = %6.4f\n", DVC_CPUBUFFNAM, DVC_CPUBUFF[x]); )

    }
    MPI_ALL_PRINT( fprintf(fp, "%s\n", STR_COLL_GIVE); )
    STR_COLL_FREE
    FREE_DVC
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ------------------------------------------------------------

  // ---------------------------------------
  // PICO init nccl comm
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

  INIT_EXPS
  TIMER_DEF(0);
  SET_EXPERIMENT_NAME(0, "sweep3d")
  SET_EXPERIMENT_TYPE(0, "nccl")
  SET_EXPERIMENT(0, "CUDA")

  SET_EXPERIMENT_NAME(1, "sweep3d")
  SET_EXPERIMENT_TYPE(1, "nccl")
  SET_EXPERIMENT(1, "TOTAL")

  SET_EXPERIMENT_NAME(2, "sweep3d")
  SET_EXPERIMENT_TYPE(2, "nccl")
  SET_EXPERIMENT(2, "Compu")

  if (nnodes > 1) {
    SET_EXPERIMENT_LAYOUT(0, "interNodes")
    SET_EXPERIMENT_LAYOUT(1, "interNodes")
    SET_EXPERIMENT_LAYOUT(2, "interNodes")
  } else {
    SET_EXPERIMENT_LAYOUT(0, "intraNode")
    SET_EXPERIMENT_LAYOUT(1, "intraNode")
    SET_EXPERIMENT_LAYOUT(2, "intraNode")
  }

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
        ncclRecv(dev_xRecvBuffer, xSize, NCCL_dtype, xDown, NCCL_COMM_WORLD, NULL);
      }

      if (yDown > -1) {
        ncclRecv(dev_yRecvBuffer, ySize, NCCL_dtype, yDown, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

      TIMER_START(0);
      {
        dim3 block_size(BLK_SIZE, 1, 1);
        dim3 grid_size(GRD_SIZE, 1, 1);
        GPU_compute<<<grid_size, block_size>>>(xSize, dev_xRecvBuffer, dev_xResultBuffer[0]);
        GPU_compute<<<grid_size, block_size>>>(ySize, dev_yRecvBuffer, dev_yResultBuffer[0]);
        checkCudaErrors( cudaDeviceSynchronize() );
      }
      TIMER_STOP(0);
      timeTakenCOMPUTE += TIMER_ELAPSED(0);

      checkCudaErrors( cudaMemset(dev_xRecvBuffer, 0, sizeof(dtype) * xSize) );
      checkCudaErrors( cudaMemset(dev_yRecvBuffer, 0, sizeof(dtype) * ySize) );
      checkCudaErrors( cudaDeviceSynchronize() );

      TIMER_START(0);
      if (xUp > -1) {
        ncclSend(dev_xSendBuffer, xSize, NCCL_dtype, xUp, NCCL_COMM_WORLD, NULL);
      }
      if (yUp > -1) {
        ncclSend(dev_ySendBuffer, ySize, NCCL_dtype, yUp, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
    }
    // =================================================================================================================


    // =================================================================================================================
    // Recreate communication pattern of sweep from (Px,0) towards (0,Py)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xUp > -1) {
        ncclRecv(dev_xRecvBuffer, xSize, NCCL_dtype, xUp, NCCL_COMM_WORLD, NULL);
      }

      if (yDown > -1) {
        ncclRecv(dev_yRecvBuffer, ySize, NCCL_dtype, yDown, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

      TIMER_START(0);
      {
        dim3 block_size(BLK_SIZE, 1, 1);
        dim3 grid_size(GRD_SIZE, 1, 1);
        GPU_compute<<<grid_size, block_size>>>(xSize, dev_xRecvBuffer, dev_xResultBuffer[1]);
        GPU_compute<<<grid_size, block_size>>>(ySize, dev_yRecvBuffer, dev_yResultBuffer[1]);
        checkCudaErrors( cudaDeviceSynchronize() );
      }
      TIMER_STOP(0);
      timeTakenCOMPUTE += TIMER_ELAPSED(0);

      checkCudaErrors( cudaMemset(dev_xRecvBuffer, 0, sizeof(dtype) * xSize) );
      checkCudaErrors( cudaMemset(dev_yRecvBuffer, 0, sizeof(dtype) * ySize) );
      checkCudaErrors( cudaDeviceSynchronize() );

      TIMER_START(0);
      if (xDown > -1) {
        ncclSend(dev_xSendBuffer, xSize, NCCL_dtype, xDown, NCCL_COMM_WORLD, NULL);
      }
      if (yUp > -1) {
        ncclSend(dev_ySendBuffer, ySize, NCCL_dtype, yUp, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
    }
    // =================================================================================================================


    // =================================================================================================================
    // Recreate communication pattern of sweep from (Px,Py) towards (0,0)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xUp > -1) {
        ncclRecv(dev_xRecvBuffer, xSize, NCCL_dtype, xUp, NCCL_COMM_WORLD, NULL);
      }

      if (yUp > -1) {
        ncclRecv(dev_yRecvBuffer, ySize, NCCL_dtype, yUp, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

      TIMER_START(0);
      {
        dim3 block_size(BLK_SIZE, 1, 1);
        dim3 grid_size(GRD_SIZE, 1, 1);
        GPU_compute<<<grid_size, block_size>>>(xSize, dev_xRecvBuffer, dev_xResultBuffer[2]);
        GPU_compute<<<grid_size, block_size>>>(ySize, dev_yRecvBuffer, dev_yResultBuffer[2]);
        checkCudaErrors( cudaDeviceSynchronize() );
      }
      TIMER_STOP(0);
      timeTakenCOMPUTE += TIMER_ELAPSED(0);

      checkCudaErrors( cudaMemset(dev_xRecvBuffer, 0, sizeof(dtype) * xSize) );
      checkCudaErrors( cudaMemset(dev_yRecvBuffer, 0, sizeof(dtype) * ySize) );
      checkCudaErrors( cudaDeviceSynchronize() );

      TIMER_START(0);
      if (xDown > -1) {
        ncclSend(dev_xSendBuffer, xSize, NCCL_dtype, xDown, NCCL_COMM_WORLD, NULL);
      }
      if (yDown > -1) {
        ncclSend(dev_ySendBuffer, ySize, NCCL_dtype, yDown, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
    }
    // =================================================================================================================


    // =================================================================================================================
    // Recreate communication pattern of sweep from (0,Py) towards (Px,0)
    for (int k = 0; k < nz; k += kba) {
      TIMER_START(0);
      if (xDown > -1) {
        ncclRecv(dev_xRecvBuffer, xSize, NCCL_dtype, xDown, NCCL_COMM_WORLD, NULL);
      }

      if (yUp > -1) {
        ncclRecv(dev_yRecvBuffer, ySize, NCCL_dtype, yUp, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);

      TIMER_START(0);
      {
        dim3 block_size(BLK_SIZE, 1, 1);
        dim3 grid_size(GRD_SIZE, 1, 1);
        GPU_compute<<<grid_size, block_size>>>(xSize, dev_xRecvBuffer, dev_xResultBuffer[3]);
        GPU_compute<<<grid_size, block_size>>>(ySize, dev_yRecvBuffer, dev_yResultBuffer[3]);
        checkCudaErrors( cudaDeviceSynchronize() );
      }
      TIMER_STOP(0);
      timeTakenCOMPUTE += TIMER_ELAPSED(0);

      checkCudaErrors( cudaMemset(dev_xRecvBuffer, 0, sizeof(dtype) * xSize) );
      checkCudaErrors( cudaMemset(dev_yRecvBuffer, 0, sizeof(dtype) * ySize) );
      checkCudaErrors( cudaDeviceSynchronize() );

      TIMER_START(0);
      if (xUp > -1) {
        ncclSend(dev_xSendBuffer, xSize, NCCL_dtype, xUp, NCCL_COMM_WORLD, NULL);
      }
      if (yDown > -1) {
        ncclSend(dev_ySendBuffer, ySize, NCCL_dtype, yDown, NCCL_COMM_WORLD, NULL);
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
    }
    // =================================================================================================================
  }

  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&end, NULL);
  ADD_TIME_EXPERIMENT(0, timeTakenCUDA)
  ADD_TIME_EXPERIMENT(1, timeTakenCUDA)
  ADD_TIME_EXPERIMENT(2, timeTakenCOMPUTE)

  MPI_Barrier(MPI_COMM_WORLD);

  // ----------------------- Check Output -----------------------
  {
    DEF_DVC
    INIT_DVC

    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);

    APPEND_DVC(dev_xResultBuffer[0], xSize, "dev_xResultBuffer[0]")
    APPEND_DVC(dev_yResultBuffer[0], ySize, "dev_yResultBuffer[0]")
    APPEND_DVC(dev_xResultBuffer[1], xSize, "dev_xResultBuffer[1]")
    APPEND_DVC(dev_yResultBuffer[1], ySize, "dev_yResultBuffer[1]")
    APPEND_DVC(dev_xResultBuffer[2], xSize, "dev_xResultBuffer[2]")
    APPEND_DVC(dev_yResultBuffer[2], ySize, "dev_yResultBuffer[2]")
    APPEND_DVC(dev_xResultBuffer[3], xSize, "dev_xResultBuffer[3]")
    APPEND_DVC(dev_yResultBuffer[3], ySize, "dev_yResultBuffer[3]")

    STR_COLL_DEF
    STR_COLL_INIT
    STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "extracted tid = %d\n", x); )
    STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "xUp = %d, xDown = %d, yUp = %d, yDown = %d\n", xUp, xDown, yUp, yDown); )

    for (int i=0; i<DVC_LEN; i+=2) {
      DVC_TOCPU(i)
      STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "%s = %6.4f, ", DVC_CPUBUFFNAM, DVC_CPUBUFF[x]); )
      DVC_TOCPU(i+1)
      STR_COLL_APPEND( sprintf(STR_COLL_BUFF, "%s = %6.4f\n", DVC_CPUBUFFNAM, DVC_CPUBUFF[x]); )
    }

    MPI_ALL_PRINT( fprintf(fp, "%s\n", STR_COLL_GIVE); )

    STR_COLL_FREE
    FREE_DVC
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_DBG_CHECK(MPI_COMM_WORLD, 1)
  // ------------------------------------------------------------

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
